#!/usr/bin/env python3
"""Step 1: AnySplat Gaussians from context cameras, rendered at every held-out camera.

One encode per (setup, crop, input size); every render variant in `gs.renders` reuses it.
Outputs per setup/variant (aligned image domain, comparable to RIFE results and GT):
  work/gs/<setup>/gs_<crop><size>_<render>/cam{k}.png, cam{k}_valid.png, cam{k}_alpha.png, meta.json
  work/gs/<setup>/gs_<crop><size>_poses.json
work/poses/<setup>.json holds the t shared with run_flow.py; it is written by the first encode only if absent.
With work/tokens/<setup> present (detect_tokens.py), each render also gets a *_local variant: only the
Gaussians inside the token reconstruction region, plus cam{k}_alpha_core.png = footprint of the token core.

Round 7: the token crop is encoded once with a fixed window (union of every token variant's
reconstruction region, gs.token_window: fixed), so all token variants share the same Gaussians and only
select different subsets (gs_tok<size>_* = full render, gs_tok<variant><size>_*_local = that variant's
local render). The local Gaussians of a target between context cameras a and b come from the tokens
detected in pair (a, b) only. gs.token_window: per_variant restores the round-6 behaviour (one encode
per variant with its own window).

Round 9 (gs.reproject): for aligned sources, the virtual view's depth comes from a 3D model and its
colours from the two neighbouring input images (lgs/reproject.py); geometry gs = rendered Gaussian
depth, vggt = AnySplat's VGGT depth maps forward-splatted. Written next to each token variant's local
render as gs_tok<variant><size>_<render>_al_rp<geometry>_local (same token core, different colours).

Round 10 (gs.tto): the Gaussians of the token reconstruction region are refined on the input views at
full crop resolution (lgs/tto.py); each checkpoint is rendered like the feed-forward Gaussians, with
suffix _tto<iterations> before _local (e.g. gs_tokrife448_nat_dil_al_tto300_local).
Round 11 (gs.tto.crop): supervision only inside a box around the trainable Gaussians, early checkpoints,
wall time per checkpoint in meta.json (tto.log[*].time_s).
Round 12 (lean): gs.local_variants limits local renders and TTO to the deployed token variant (window
unchanged), gs.render_full: false skips the full-window renders, gs.tto.crop.per_view gives every input
view its own box.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lgs.anysplat_wrap import (AnySplatRunner, apply_sim3, distort_maps, interp_pose, paste_and_warp,  # noqa: E402
                               rot_angle_deg, sim3_from_cams, token_keep)
from lgs.common import (bbox, cam_path, cube_mask, imread_rgb, imwrite, load_cfg, load_json, read_mask,  # noqa: E402
                        save_json, setup_cams, neighbors, write_mask)
from lgs.aligned import aligned_reference, crop_K_to_raw, fit_plane, plane_homography  # noqa: E402
from lgs.compat import anysplat_source  # noqa: E402
from lgs.reproject import grid_points, reproject_view, zbuffer_depth  # noqa: E402
from lgs.tto import optimize as tto_optimize  # noqa: E402


def square_around(box, raw_shape, pad, min_side):
    h, w = raw_shape
    x0, y0, x1, y1 = box
    side = int(min(max(x1 - x0 + 2 * pad, y1 - y0 + 2 * pad, min_side), min(h, w)))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    return (int(np.clip(cx - side // 2, 0, w - side)), int(np.clip(cy - side // 2, 0, h - side)), side)


def crop_window(mode, raw_shape, ctx_raws, fg_masks, gcfg, ck, token_masks=None, token_pad=0):
    h, w = raw_shape
    if mode == "center":
        side = min(h, w)
        return ((w - side) // 2, (h - side) // 2, side)
    if mode == "fg":
        x0, y0, x1, y1 = bbox(np.logical_or.reduce(fg_masks))
        return square_around((x0, y0, x1, y1), raw_shape, int(0.03 * max(x1 - x0, y1 - y0)), 0)
    if mode == "token":  # union of the context cameras' token reconstruction regions (core + margin)
        if token_masks is None or not any(m.any() for m in token_masks):
            raise RuntimeError("token 裁剪需要先运行 scripts/detect_tokens.py，且至少一台输入相机有标记")
        return square_around(bbox(np.logical_or.reduce(token_masks)), raw_shape, token_pad, gcfg["roi_min_side"])
    if mode == "roi":
        if gcfg.get("roi_window"):
            return tuple(int(v) for v in gcfg["roi_window"])
        masks = [cube_mask(im, ck) for im in ctx_raws]
        if not all(m.any() for m in masks):
            raise RuntimeError("未在所有输入相机中检测到魔方，请在配置 gs.roi_window 中手动指定 [x0, y0, side]")
        return square_around(bbox(np.logical_or.reduce(masks)), raw_shape, gcfg["roi_pad_px"], gcfg["roi_min_side"])
    raise ValueError(mode)


POSE_SUFFIX = {"vggt": "", "interp": "_int", "aligned_int": "_al", "aligned_vggt": "_alv", "calib": "_cal", "calibK": "_calK"}


def load_rig(work):
    p = work / "calib" / "rig.json"
    if not p.exists():
        return None
    cams = load_json(p)["cameras"]
    return {int(k): {"c2w": np.array(v["c2w"]), "K": np.array(v["K"]), "dist": np.array(v["dist"])}
            for k, v in cams.items()}


def target_poses(state, ctx, targets, win, rig, sources, t_by_target=None):
    """Target camera poses/intrinsics in the Gaussian frame for each requested pose source.

    vggt:   joint VGGT camera head (round 1-2).
    interp: virtual camera interpolated between the two neighbouring context cameras in the Gaussian
            frame at the same t the flow branch uses (one shared camera definition for both branches).
    calib:  offline rig calibration, mapped into the Gaussian frame by a Sim(3) fitted on the
            context cameras; intrinsics still from VGGT (consistent with how the Gaussians were built).
    calibK: as calib, plus calibrated focal/principal point and lens distortion.
    Returns {source: (c2w, intr_norm, use_distortion, diagnostics)}.
    """
    out = {"vggt": (state["tgt_c2w"], state["tgt_intr"], False, None)}
    if "interp" in sources:
        c2w, intr = [], []
        for k in targets:
            a, b = neighbors(ctx, k)
            ia, ib = ctx.index(a), ctx.index(b)
            p, kk = interp_pose(state["ctx_c2w"][ia], state["ctx_c2w"][ib], state["ctx_intr"][ia],
                                state["ctx_intr"][ib], float(t_by_target[k]))
            c2w.append(p)
            intr.append(kk)
        out["interp"] = (np.stack(c2w), np.stack(intr), False, None)
    if not ({"calib", "calibK"} & set(sources)):
        return {k: v for k, v in out.items() if k in sources}
    if rig is None:
        raise RuntimeError("pose_sources 含 calib/calibK，但缺少 work/calib/rig.json，请先运行 scripts/calibrate_rig.py")
    cal_ctx = np.stack([rig[k]["c2w"] for k in ctx])
    s, r, t = sim3_from_cams(cal_ctx, state["ctx_c2w"])
    al_ctx = apply_sim3(cal_ctx, s, r, t)
    base = np.linalg.norm(state["ctx_c2w"][-1, :3, 3] - state["ctx_c2w"][0, :3, 3]) + 1e-12
    pdiag = {"sim3_scale": float(s),
             "ctx_rot_err_deg": [rot_angle_deg(al_ctx[i, :3, :3], state["ctx_c2w"][i, :3, :3]) for i in range(len(ctx))],
             "ctx_center_err_rel_baseline": [float(np.linalg.norm(al_ctx[i, :3, 3] - state["ctx_c2w"][i, :3, 3]) / base)
                                             for i in range(len(ctx))]}
    tgt = apply_sim3(np.stack([rig[k]["c2w"] for k in targets]), s, r, t)
    out["calib"] = (tgt, state["tgt_intr"], False, pdiag)
    x0, y0, side = win
    intr = []
    for k in targets:
        K = rig[k]["K"]
        intr.append(np.array([[K[0, 0] / side, 0, (K[0, 2] - x0) / side],
                              [0, K[1, 1] / side, (K[1, 2] - y0) / side], [0, 0, 1]]))
    out["calibK"] = (tgt, np.stack(intr), True, pdiag)
    return {k: v for k, v in out.items() if k in sources}


def aligned_setup(runner, state, ctx, targets, win, hs, gcfg, shared_t, aligned_shape):
    """Aligned-domain reference camera, zero-parallax plane and target centres in the Gaussian frame.

    Uses only the context cameras (their AnySplat poses/intrinsics and their known homographies H).
    Target centres: aligned_int = interpolated between the neighbouring context centres at the flow's t;
    aligned_vggt = the joint VGGT estimate of the target centre (mapped into the Gaussian frame).
    """
    ref = gcfg["aligned_ref"]
    if ref not in ctx:
        raise RuntimeError(f"aligned_ref={ref} 必须是输入相机之一（对齐参考相机）")
    K_raw = [crop_K_to_raw(k, win) for k in state["ctx_intr"]]
    ir = ctx.index(ref)
    K_al, c2w_al = aligned_reference(K_raw[ir], state["ctx_c2w"][ir], hs[ref])
    # rough plane distance from the reference camera's rendered depth (initialisation only)
    size = state["size"]
    Ks = np.array(state["ctx_intr"][ir], np.float64).copy()
    Ks[0] *= size
    Ks[1] *= size
    _, a, dep = runner.render_px(state, state["ctx_c2w"][ir:ir + 1], Ks[None], size, size, depth=True)
    depth0 = float(np.median(dep[0][a[0] > 0.5])) if (a[0] > 0.5).any() else 1.0
    x0, y0, side = win
    grid = np.stack(np.meshgrid(np.linspace(x0 + 0.1 * side, x0 + 0.9 * side, 9),
                                np.linspace(y0 + 0.1 * side, y0 + 0.9 * side, 9)), -1).reshape(-1, 2)
    cams = [(K_raw[i], state["ctx_c2w"][i], hs[c], grid) for i, c in enumerate(ctx) if c != ref]
    n, d, rms = fit_plane(cams, K_al, c2w_al, depth0)
    centers = {"aligned_int": [], "aligned_vggt": []}
    for i, k in enumerate(targets):
        a_, b_ = neighbors(ctx, k)
        t = float(shared_t[k])
        ca, cb = state["ctx_c2w"][ctx.index(a_)][:3, 3], state["ctx_c2w"][ctx.index(b_)][:3, 3]
        centers["aligned_int"].append((1 - t) * ca + t * cb)
        centers["aligned_vggt"].append(state["tgt_c2w"][i][:3, 3])
    return {"ref": ref, "K_al": K_al, "c2w_al": c2w_al, "n": n, "d": d, "plane_rms_px": rms,
            "centers": centers, "shape": aligned_shape, "pad": int(gcfg.get("aligned_pad_px", 160))}


def render_aligned(runner, state, al, src, keep, kw):
    """Off-axis views in the flow (aligned) domain: reference rotation/intrinsics at each target
    centre, then the zero-parallax-plane homography. `keep` is None, one mask, or one mask per target.
    Returns [(rgb, alpha, valid)] per target."""
    h, w = al["shape"]
    p = al["pad"]
    K = al["K_al"].copy()
    K[0, 2] += p
    K[1, 2] += p
    shift = np.array([[1, 0, -p], [0, 1, -p], [0, 0, 1.0]])
    out = []
    for i, c in enumerate(al["centers"][src]):
        cam = al["c2w_al"].copy()
        cam[:3, 3] = c
        kp = keep[i] if isinstance(keep, list) else keep
        rgb, alpha, _ = runner.render_px(state, cam[None], K[None], w + 2 * p, h + 2 * p, keep=kp, **kw)
        G = plane_homography(al["K_al"], al["c2w_al"], c, al["n"], al["d"], (w, h)) @ shift
        img = cv2.warpPerspective(rgb[0], G, (w, h), flags=cv2.INTER_LINEAR)
        a = cv2.warpPerspective(alpha[0], G, (w, h), flags=cv2.INTER_LINEAR)
        valid = cv2.warpPerspective(np.ones((h + 2 * p, w + 2 * p), np.uint8), G, (w, h),
                                    flags=cv2.INTER_NEAREST) > 0
        out.append((np.clip(img, 0, 1), a, valid))
    return out


def reproject_targets(runner, state, al, src, ctx, targets, shared_t, raws, win, geom, rcfg, gs_ctx_depth):
    """Round 9: per target, (rgb, alpha, valid) in the aligned domain with colours taken from the two
    neighbouring input images and depth from `geom` (gs | vggt); plus diagnostics."""
    h, w = al["shape"]
    p = al["pad"]
    W, H = w + 2 * p, h + 2 * p
    K = al["K_al"].copy()
    K[0, 2] += p
    K[1, 2] += p
    shift = np.array([[1, 0, -p], [0, 1, -p], [0, 0, 1.0]])
    size = state["size"]
    x0, y0, side = win
    if geom == "vggt":
        if "vggt_depth" not in state:
            raise RuntimeError("AnySplat 编码输出里没有 VGGT 深度（depth_dict），无法做 vggt 几何重投影")
        src_depth = state["vggt_depth"]
    else:
        src_depth = gs_ctx_depth
    imgs = {}  # float raw images of the context cameras actually used
    out, stats = [], []
    for i, (k, c) in enumerate(zip(targets, al["centers"][src])):
        a, b = neighbors(ctx, k)
        t = float(shared_t[k])
        cam = al["c2w_al"].copy()
        cam[:3, 3] = c
        if geom == "gs":
            _, alpha, dep = runner.render_px(state, cam[None], K[None], W, H, depth=True)
            depth, dvalid = dep[0], alpha[0] > 0.5
        else:
            pts = []
            for kk in (a, b):
                j = ctx.index(kk)
                Kc = np.array(state["ctx_intr"][j], np.float64).copy()
                Kc[:2] *= size
                X, ok = grid_points(Kc, state["ctx_c2w"][j], src_depth[j])
                pts.append(X[ok])
            depth, dvalid = zbuffer_depth(np.concatenate(pts), K, cam, W, H,
                                          scale=float(rcfg.get("splat_scale") or min(1.0, size / side)))
        for kk in (a, b):
            if kk not in imgs:
                imgs[kk] = raws[kk].astype(np.float32) / 255.0
        sources = [dict(img=imgs[kk], K=crop_K_to_raw(state["ctx_intr"][ctx.index(kk)], win),
                        c2w=state["ctx_c2w"][ctx.index(kk)], depth=src_depth[ctx.index(kk)], window=win, weight=wt)
                   for kk, wt in ((a, 1 - t), (b, t))]
        rgb, ralpha, nvis = reproject_view(K, cam, depth, dvalid, sources, rcfg.get("vis_tol", 0.05))
        G = plane_homography(al["K_al"], al["c2w_al"], c, al["n"], al["d"], (w, h)) @ shift
        img = cv2.warpPerspective(rgb, G, (w, h), flags=cv2.INTER_LINEAR)
        av = cv2.warpPerspective(ralpha, G, (w, h), flags=cv2.INTER_LINEAR)
        valid = cv2.warpPerspective(np.ones((H, W), np.uint8), G, (w, h), flags=cv2.INTER_NEAREST) > 0
        out.append((np.clip(img, 0, 1), av, valid))
        nd = max(int(dvalid.sum()), 1)
        stats.append({"target": k, "depth_valid_px": int(dvalid.sum()),
                      "seen_by_both": float(((nvis == 2) & dvalid).sum() / nd),
                      "seen_by_one": float(((nvis == 1) & dvalid).sum() / nd),
                      "seen_by_none": float(((nvis == 0) & dvalid).sum() / nd)})
    return out, stats


def depth_agreement(state, gs_ctx_depth, size):
    """Median relative difference between VGGT depth and rendered Gaussian depth on the input views."""
    if "vggt_depth" not in state:
        return None
    res = []
    for g, v in zip(gs_ctx_depth, state["vggt_depth"]):
        m = (g > 0) & (v > 0)
        res.append(float(np.median(np.abs(v[m] / g[m] - 1))) if m.any() else float("nan"))
    return res


def load_token_sets(tok_root, ctx, tcfg):
    """{variant: {"core": [raw masks], "recon": [raw masks], "pairs": {(a, b): {"core": [...], "recon": [...]}}}}
    from detect_tokens.py output; per-pair lists hold one mask per context camera (empty off the pair).

    Supports the round-7 layout (with pair*/), round 6 tokens/<setup>/<variant>/ and the flat tokens/<setup>/."""
    if not tok_root.exists():
        return {}

    def load(d):
        out = {kind: [read_mask(d / f"cam{k}_{kind}_raw.png") for k in ctx] for kind in ("core", "recon")}
        pairs = {}
        for a, b in zip(ctx[:-1], ctx[1:]):
            pd = d / f"pair{a}-{b}"
            if not pd.is_dir():
                continue
            empty = np.zeros_like(out["core"][0])
            pairs[(a, b)] = {kind: [read_mask(pd / f"cam{k}_{kind}_raw.png") if k in (a, b) else empty
                                    for k in ctx] for kind in ("core", "recon")}
        out["pairs"] = pairs
        return out

    names = [v for v in (tcfg.get("variants") or {}) if (tok_root / v).is_dir()]
    if names:
        return {v: load(tok_root / v) for v in names}
    if (tok_root / f"cam{ctx[0]}_core_raw.png").exists():
        return {"flat": load(tok_root)}
    return {}


def group_windows(masks, raw_shape, mcfg, min_side):
    """Round 14 multi-window encode: split the union of the context cameras' recon masks (raw frame) into
    connected components and merge them greedily (smallest merged square first) while the merged square stays
    within max_side px, is no larger than the bigger of the two, or there are more than max_windows groups.
    Returns [(window (x0, y0, side), region mask)]; region = the components of that window, so every token
    pixel belongs to exactly one window. A single component larger than max_side keeps its own large window."""
    union = np.logical_or.reduce(masks)
    if not union.any():
        return []
    n, lab, stats, _ = cv2.connectedComponentsWithStats(union.astype(np.uint8), connectivity=8)
    groups = [{"box": (s[0], s[1], s[0] + s[2], s[1] + s[3]), "comps": [i]} for i, s in enumerate(stats) if i]

    def side(b):
        return max(b[2] - b[0], b[3] - b[1])

    def merged(a, b):
        return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

    max_side, max_windows = int(mcfg.get("max_side", 640)), int(mcfg.get("max_windows", 4))
    while len(groups) > 1:
        best = min(((side(merged(g["box"], h["box"])), i, j) for i, g in enumerate(groups)
                    for j, h in enumerate(groups) if j > i))
        s, i, j = best
        free = s <= max(side(groups[i]["box"]), side(groups[j]["box"]))
        if not (s <= max_side or free or len(groups) > max_windows):
            break
        groups[i] = {"box": merged(groups[i]["box"], groups[j]["box"]), "comps": groups[i]["comps"] + groups[j]["comps"]}
        del groups[j]
    pad = int(mcfg.get("pad_px", 16))
    return [(square_around(g["box"], raw_shape, pad, min_side), np.isin(lab, g["comps"])) for g in groups]


def restrict_tokens(ts, region):
    """Token set limited to one window's region (per camera, per pair)."""
    out = {kind: [m & region for m in ts[kind]] for kind in ("core", "recon")}
    out["pairs"] = {pr: {kind: [m & region for m in v[kind]] for kind in ("core", "recon")}
                    for pr, v in (ts.get("pairs") or {}).items()}
    return out


def read_gray(path):
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0


def composite_windows(work, setup, win_roots, targets):
    """Round 14: merge the per-window renders of a multi-window variant into one gs/<setup>/<name> directory
    (pixel taken from the window with the largest alpha; alpha / core alpha = max; valid = any). Meta sums the
    encode / TTO / render times and Gaussian counts over the windows and lists every window."""
    names = sorted({p.name for r in win_roots for p in (r / setup).iterdir() if p.is_dir()} if win_roots else set())
    for name in names:
        dirs = [r / setup / name for r in win_roots if (r / setup / name / "meta.json").exists()]
        if not dirs:
            continue
        metas = [load_json(d / "meta.json") for d in dirs]
        out = work / "gs" / setup / name
        for k in targets:
            imgs = [imread_rgb(d / f"cam{k}.png").astype(np.float32) for d in dirs]
            alphas = np.stack([read_gray(d / f"cam{k}_alpha.png") for d in dirs])
            best = alphas.argmax(0)
            img = np.take_along_axis(np.stack(imgs), best[None, ..., None], 0)[0]
            imwrite(out / f"cam{k}.png", img.astype(np.uint8))
            imwrite(out / f"cam{k}_alpha.png", alphas.max(0))
            write_mask(out / f"cam{k}_valid.png", np.logical_or.reduce([read_mask(d / f"cam{k}_valid.png") for d in dirs]))
            cores = [d / f"cam{k}_alpha_core.png" for d in dirs if (d / f"cam{k}_alpha_core.png").exists()]
            if cores:
                imwrite(out / f"cam{k}_alpha_core.png", np.stack([read_gray(c) for c in cores]).max(0))
        m = dict(metas[0])
        sides = [x["window"][2] for x in metas]
        m.update(window=metas[int(np.argmax(sides))]["window"], windows=[x["window"] for x in metas],
                 num_gaussians=sum(x["num_gaussians"] for x in metas),
                 time_encode_s=sum(x["time_encode_s"] for x in metas),
                 time_render_per_view_s=sum(x["time_render_per_view_s"] for x in metas),
                 local_gaussians=[x["local_gaussians"] for x in metas])
        ttos = [x["tto"] for x in metas if x.get("tto")]
        if ttos:
            log = {}
            for t in ttos:
                for e in t["log"]:
                    log.setdefault(e["iter"], []).append(e)
            merged = []
            for it in sorted(log):
                es = log[it]
                e = {"iter": it, "loss": float(np.mean([x["loss"] for x in es])),
                     "ctx_psnr": np.mean([x["ctx_psnr"] for x in es], 0).tolist()}
                if all("time_s" in x for x in es) and len(es) == len(ttos):
                    e["time_s"] = float(sum(x["time_s"] for x in es))
                boxes = [x["box"] for x in es if x.get("box")]
                if boxes:
                    e["box"] = dict(boxes[0], area_frac=float(np.mean([b["area_frac"] for b in boxes])))
                merged.append(e)
            m["tto"] = {"train_gaussians": sum(t["train_gaussians"] for t in ttos), "res": max(t["res"] for t in ttos),
                        "time_s": sum(t["time_s"] for t in ttos), "log": merged, "windows": len(ttos)}
        save_json(out / "meta.json", m)
        print(f"  [{setup}] multi-window {name}: {len(dirs)} windows, sides {sides}")


def local_keep(runner, state, tokens, win, size, ctx, targets, gcfg, depth):
    """Per-target (recon, core) keep masks. With per-pair tokens a target between a and b keeps only the
    Gaussians selected by pair (a, b); otherwise every target uses the union over all context cameras."""
    x0, y0, side = win

    def sel(mask_list):
        masks = [cv2.resize(m[y0:y0 + side, x0:x0 + side].astype(np.uint8), (size, size),
                            interpolation=cv2.INTER_NEAREST) > 0 for m in mask_list]
        if not any(m.any() for m in masks):
            return np.zeros(len(state["gaussians"].means[0]), bool)
        if gcfg.get("keep_mode", "visible") == "visible":
            return runner.visible_keep(state, masks, gcfg.get("visible_tol", 0.03), depth=depth).cpu().numpy()
        return token_keep(state["gaussians"].means[0], state["ctx_c2w"], state["ctx_intr"], masks).cpu().numpy()

    if tokens.get("pairs"):
        by_pair = {pr: (sel(m["recon"]), sel(m["core"])) for pr, m in tokens["pairs"].items()}
        per_t = [by_pair[neighbors(ctx, k)] for k in targets]
        counts = {f"{a}-{b}": {"recon": int(r.sum()), "core": int(c.sum())} for (a, b), (r, c) in by_pair.items()}
    else:
        both = (sel(tokens["recon"]), sel(tokens["core"]))
        per_t = [both] * len(targets)
        counts = {"all": {"recon": int(both[0].sum()), "core": int(both[1].sum())}}
    return [r for r, _ in per_t], [c for _, c in per_t], counts


def rig_positions(c2w_by_cam, ctx):
    """Camera positions projected on the context baseline: first context cam = 0, last = 1."""
    c = {k: v[:3, 3] for k, v in c2w_by_cam.items()}
    d = c[ctx[-1]] - c[ctx[0]]
    length = np.linalg.norm(d)
    d = d / length
    pos, off = {}, {}
    for k, ck in c.items():
        r = ck - c[ctx[0]]
        pos[k] = float(r @ d / length)
        off[k] = float(np.linalg.norm(r - (r @ d) * d) / length)
    return pos, off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--setups", nargs="+", default=None)
    ap.add_argument("--sizes", nargs="+", type=int, default=None, help="覆盖配置 gs.sizes")
    ap.add_argument("--crops", nargs="+", default=None, help="覆盖配置 gs.crops")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    gcfg = cfg["gs"]
    sizes = args.sizes or gcfg["sizes"]
    crops = args.crops or gcfg["crops"]
    work = Path(cfg["work_dir"])
    hs = np.load(work / "prepare" / "H_raw2aligned.npy")
    compat_packages = work / "compat_packages"
    if compat_packages.exists():
        sys.path.insert(0, str(compat_packages))
    runner = AnySplatRunner(anysplat_source(cfg["anysplat_dir"], work), cfg["anysplat_ckpt"])
    rig = load_rig(work)
    dmaps = {}

    tcfg = cfg.get("tokens", {})

    for setup in args.setups or list(cfg["setups"]):
        ctx, targets = setup_cams(cfg, setup)
        raws = {k: imread_rgb(cam_path(cfg, "raw", k)) for k in ctx + targets}
        raw_shape = raws[ctx[0]].shape[:2]
        aligned_shape = imread_rgb(cam_path(cfg, "aligned", ctx[0])).shape[:2]
        fg_ctx = [read_mask(work / "prepare" / f"fg_raw_{k:02d}.png") for k in ctx]
        token_sets = load_token_sets(work / "tokens" / setup, ctx, tcfg)
        # Shared t with the flow branch: reuse the existing pose file (e.g. copied from round 2) if present.
        pose_file = work / "poses" / f"{setup}.json"
        shared_t = {int(k): v for k, v in load_json(pose_file)["t_pose"].items()} if pose_file.exists() else None
        # token crop, fixed window (default): one encode whose window covers every variant's recon region;
        # each variant only selects its own local Gaussians -> label "tok", locals "tok<variant>".
        # token crop, per_variant (round 6): one encode per variant with that variant's own window.
        # Other crops use the primary variant's masks for their local Gaussians.
        jobs = []  # (crop, label, window masks, {local label: (variant, token set)}, fixed window, output root)
        mcfg = gcfg.get("multi_window") or {}
        win_roots = []
        for crop in crops:
            if crop == "token" and token_sets:
                live = {}
                for v, ts in token_sets.items():
                    if any(m.any() for m in ts["recon"]):
                        live[v] = ts
                    else:
                        print(f"[{setup}] token 变体 {v} 在所有输入相机上都没有标记：不触发 GS，等价于纯光流")
                if not live:
                    continue
                # gs.local_variants (round 12): only these variants get local renders / TTO; the encode window
                # still covers every detected variant, so the Gaussians match the earlier rounds.
                lv = gcfg.get("local_variants")
                sel = {v: ts for v, ts in live.items() if not lv or v in lv}
                # gs.window_variants (round 13): which variants define the encode window (default: all)
                wv = gcfg.get("window_variants")
                wsets = [ts for v, ts in live.items() if not wv or v in wv] or list(live.values())
                if gcfg.get("token_window", "fixed") == "fixed":
                    union = [np.logical_or.reduce([ts["recon"][i] for ts in wsets]) for i in range(len(ctx))]
                    jobs.append((crop, "tok", union, {f"tok{v}": (v, ts) for v, ts in sel.items()}, None, None))
                else:
                    jobs += [(crop, f"tok{v}", ts["recon"], {f"tok{v}": (v, ts)}, None, None) for v, ts in sel.items()]
                # gs.multi_window (round 14): the same variants again, encoded window by window (one small window
                # per group of token components, so the 448 px input is barely upsampled); locals "tok<v>_mw",
                # rendered under gs_win/<v>_w<i>/ and composited into gs/<setup>/ afterwards
                if mcfg.get("enabled"):
                    for v, ts in sel.items():
                        groups = group_windows(ts["recon"], raw_shape, mcfg, gcfg["roi_min_side"])
                        print(f"[{setup}] token 变体 {v}: {len(groups)} 个编码窗口，边长 {[g[0][2] for g in groups]}")
                        for i, (w, region) in enumerate(groups):
                            root = work / "gs_win" / f"{v}_w{i}"
                            win_roots.append(root)
                            jobs.append((crop, f"tok{v}_mw{i}_", None, {f"tok{v}_mw": (v, restrict_tokens(ts, region))},
                                         w, root))
            else:
                prim = token_sets.get(tcfg.get("primary")) if token_sets else None
                jobs.append((crop, crop, None, {crop: (tcfg.get("primary"), prim)} if prim else {}, None, None))
        for crop, label, win_masks, locals_, win_fixed, out_root in jobs:
            out_root = out_root or work / "gs"
            win = win_fixed or crop_window(crop, raw_shape, [raws[k] for k in ctx], fg_ctx, gcfg, cfg["chroma"],
                                           win_masks, tcfg.get("crop_pad_px", 0))
            x0, y0, side = win
            for size in sizes:
                assert size % 14 == 0, "AnySplat/VGGT 需要 14 的倍数"
                base = f"gs_{label}{size}"
                crops_img = {k: cv2.resize(raws[k][y0:y0 + side, x0:x0 + side], (size, size),
                                           interpolation=cv2.INTER_AREA) for k in raws}
                print(f"[{setup}] {base}: window x0={x0} y0={y0} side={side} (x{side / size:.2f} upsampling)")
                state, diag = runner.encode([crops_img[k] for k in ctx], [crops_img[k] for k in targets], size)
                print(f"  gaussians={diag['num_gaussians']}  encode={diag['time_encode_s']:.3f}s  "
                      f"ctx self-PSNR={np.mean(diag['ctx_self_psnr']):.2f}dB  "
                      f"pose align rot err={max(diag['ctx_rot_err_deg']):.2f}deg")

                order = ctx + targets
                c2w = {k: state["all_c2w_joint"][i] for i, k in enumerate(order)}
                pos, off = rig_positions(c2w, ctx)
                t_pose = {k: (pos[k] - pos[neighbors(ctx, k)[0]]) /
                          (pos[neighbors(ctx, k)[1]] - pos[neighbors(ctx, k)[0]]) for k in targets}
                poses = {
                    "context": ctx, "targets": targets,
                    "baseline_pos": {k: pos[k] for k in sorted(pos)},
                    "offset_from_baseline_rel": {k: off[k] for k in sorted(off)},
                    "t_pose": t_pose,
                    "t_uniform": {k: (k - neighbors(ctx, k)[0]) / (neighbors(ctx, k)[1] - neighbors(ctx, k)[0])
                                  for k in targets},
                    "c2w_joint": {k: c2w[k].tolist() for k in sorted(c2w)},
                }
                save_json(out_root / setup / f"{base}_poses.json", poses)
                if shared_t is None:
                    save_json(pose_file, dict(poses, from_variant=base))
                    shared_t = t_pose
                print("  t_pose (this encode):", {k: round(v, 3) for k, v in t_pose.items()},
                      " t shared with flow:", {k: round(shared_t[k], 3) for k in targets})

                keeps, local_counts = {}, {}
                depth = runner.ctx_depth(state) if locals_ else None
                rcfg = gcfg.get("reproject") or {}
                rp_cache = {}
                if rcfg.get("geometries") and locals_:
                    diag["vggt_vs_gs_depth_rel"] = depth_agreement(state, depth, size)
                    print(f"  VGGT vs GS depth on inputs (median |ratio-1|): {diag['vggt_vs_gs_depth_rel']}")
                for lab, (tv, ts) in locals_.items():
                    kr, kc, counts = local_keep(runner, state, ts, win, size, ctx, targets, gcfg, depth)
                    keeps[lab] = (tv, kr, kc)
                    local_counts[lab] = counts
                    print(f"  local GS {lab} ({gcfg.get('keep_mode', 'visible')}, of {diag['num_gaussians']}): "
                          + ", ".join(f"{pr} recon={c['recon']} core={c['core']}" for pr, c in counts.items()))

                # Camera-domain sources render the raw target view and warp it with that camera's H.
                sources = [s for s in gcfg["pose_sources"] if not s.startswith("aligned")]
                poses_by_src = target_poses(state, ctx, targets, win, rig, sources, shared_t)
                # Aligned-domain sources render the off-axis view at a centre directly in the flow domain.
                al = None
                if any(s.startswith("aligned") for s in gcfg["pose_sources"]):
                    al = aligned_setup(runner, state, ctx, targets, win, hs, gcfg, shared_t, aligned_shape)
                    print(f"  aligned domain: ref cam {al['ref']}, zero-parallax plane fit rms={al['plane_rms_px']:.3f}px")
                    for s in gcfg["pose_sources"]:
                        if s.startswith("aligned"):
                            poses_by_src[s] = (None, None, False, {"plane_rms_px": al["plane_rms_px"]})

                # Round 10: test-time optimisation of the token-region Gaussians on the input views;
                # every checkpoint is rendered like the feed-forward Gaussians, with suffix _tto<iters>.
                stages = [("", state, 0)]
                ocfg = gcfg.get("tto") or {}
                if ocfg.get("checkpoints") and keeps:
                    if ocfg.get("scope", "local") == "local":
                        train = np.logical_or.reduce([m for (_, kr, _) in keeps.values() for m in kr])
                    else:
                        train = np.ones(diag["num_gaussians"], bool)
                    R = int(min(side, ocfg.get("max_res", 1008)))
                    imgs = [cv2.resize(raws[k][y0:y0 + side, x0:x0 + side], (R, R), interpolation=cv2.INTER_AREA)
                            .astype(np.float32) / 255.0 for k in ctx]
                    t1 = time.time()
                    snaps, olog = tto_optimize(runner, state, imgs, R, sorted(ocfg["checkpoints"]), train, ocfg,
                                               eps2d=0.3 * (R / size) ** 2)
                    diag["tto"] = {"train_gaussians": int(train.sum()), "res": R, "time_s": time.time() - t1,
                                   "log": olog}
                    box = olog[0].get("box")
                    print(f"  TTO: {int(train.sum())} of {diag['num_gaussians']} Gaussians at {R}px, "
                          + (f"supervised box {box['w']}x{box['h']} ({box['area_frac'] * 100:.0f}% of view), "
                             if box else "full view, ")
                          + f"{time.time() - t1:.1f}s total; " + ", ".join(
                              f"it{e['iter']}: PSNR {np.mean(e['ctx_psnr']):.2f}"
                              + (f" @ {e['time_s']:.2f}s" if "time_s" in e else "") for e in olog
                              if e["iter"] == 1 or e["iter"] in snaps))
                    stages += [(f"_tto{it}", dict(state, gaussians=g_), it) for it, g_ in sorted(snaps.items())]

                for stage, gst, tto_iter in stages:
                    for src, (tgt_c2w, tgt_intr, use_dist, pdiag) in poses_by_src.items():
                        if pdiag and "ctx_rot_err_deg" in pdiag:
                            print(f"  pose source {src}: ctx rot err after Sim3={max(pdiag['ctx_rot_err_deg']):.2f}deg "
                                  f"center err/baseline={max(pdiag['ctx_center_err_rel_baseline']):.3f}")
                        for rv in gcfg["renders"]:
                            if src.startswith("aligned"):
                                res = None
                                scale = al["K_al"][0, 0] / (state["ctx_intr"][ctx.index(al["ref"])][0, 0] * size)
                            else:
                                res = size if rv["res"] == "input" else side
                                scale = res / size
                            eps2d = rv["eps2d"]
                            if eps2d == "auto":  # keep the 0.3 px dilation of the input grid when rendering denser
                                eps2d = 0.3 * scale ** 2
                            kw = dict(mode=rv["mode"], eps2d=eps2d, filter3d=rv["filter3d"],
                                      filter3d_var=gcfg["filter3d_var"], filter3d_comp=gcfg.get("filter3d_comp", False))
                            tail = f"{size}_{rv['name']}{POSE_SUFFIX[src]}"
                            # full: all Gaussians of the crop; local: only the token reconstruction region
                            outputs = ({f"{base}_{rv['name']}{POSE_SUFFIX[src]}{stage}": (None, None, None, None)}
                                       if gcfg.get("render_full", True) or not keeps else {})
                            for lab, (tv, kr, kc) in keeps.items():
                                outputs[f"gs_{lab}{tail}{stage}_local"] = (tv, kr, kc, local_counts[lab])
                            for variant, (tok_variant, kp, kp_core, counts) in outputs.items():
                                out = out_root / setup / variant
                                t0 = time.time()
                                if src.startswith("aligned"):
                                    views = render_aligned(runner, gst, al, src, kp, kw)
                                    cores = render_aligned(runner, gst, al, src, kp_core, kw) if kp_core is not None \
                                        else None
                                else:
                                    views = []
                                    cores = [] if kp_core is not None else None
                                    for i, k in enumerate(targets):
                                        sl = slice(i, i + 1)
                                        rgb, alpha = runner.render(gst, tgt_c2w[sl], tgt_intr[sl], res,
                                                                   keep=kp[i] if kp is not None else None, **kw)
                                        dmap = None
                                        if use_dist:
                                            dmap = dmaps.setdefault(k, distort_maps(rig[k]["K"], rig[k]["dist"], raw_shape))
                                        views.append(paste_and_warp(rgb[0], alpha[0], win, raw_shape, hs[k],
                                                                    aligned_shape, dmap))
                                        if kp_core is not None:
                                            acore = runner.render(gst, tgt_c2w[sl], tgt_intr[sl], res, keep=kp_core[i],
                                                                  **kw)[1]
                                            cores.append(paste_and_warp(rgb[0], acore[0], win, raw_shape, hs[k],
                                                                        aligned_shape, dmap))
                                t_render = (time.time() - t0) / len(targets)
                                for i, k in enumerate(targets):
                                    img, a, valid = views[i]
                                    imwrite(out / f"cam{k}.png", img)
                                    imwrite(out / f"cam{k}_alpha.png", np.clip(a, 0, 1))
                                    write_mask(out / f"cam{k}_valid.png", valid)
                                    if cores is not None:
                                        imwrite(out / f"cam{k}_alpha_core.png", np.clip(cores[i][1], 0, 1))
                                meta = dict(diag, window=list(win), size=size, crop=crop, token_variant=tok_variant,
                                            token_window=gcfg.get("token_window", "fixed") if crop == "token" else None,
                                            local_gaussians=counts, render=rv, render_res=res,
                                            eps2d_used=float(eps2d), time_render_per_view_s=t_render, pose_source=src,
                                            pose_diag=pdiag, local=kp is not None, tto_iters=tto_iter)
                                save_json(out / "meta.json", meta)
                                print(f"  {variant}: res={res or 'aligned'} mode={rv['mode']} eps2d={eps2d:.3f} "
                                      f"filter3d={rv['filter3d']} render/view={t_render * 1000:.1f}ms")
                                # Round 9: same token core, colours from the input images, depth from gs / vggt
                                if stage or kp is None or cores is None or src not in rcfg.get("sources", ["aligned_int"]):
                                    continue
                                for geom in rcfg.get("geometries") or []:
                                    if (src, geom) not in rp_cache:
                                        t1 = time.time()
                                        rp_cache[(src, geom)] = reproject_targets(
                                            runner, state, al, src, ctx, targets, shared_t, raws, win, geom, rcfg, depth)
                                        rp_cache[(src, geom)] += ((time.time() - t1) / len(targets),)
                                        st = rp_cache[(src, geom)][1]
                                        print(f"  reproject[{src}/{geom}]: "
                                              + ", ".join(f"cam{x['target']} both={x['seen_by_both']:.2f} "
                                                          f"none={x['seen_by_none']:.2f}" for x in st))
                                    views_rp, st, t_rp = rp_cache[(src, geom)]
                                    out_rp = out_root / setup / variant.replace("_local", f"_rp{geom}_local")
                                    for i, k in enumerate(targets):
                                        img, a, valid = views_rp[i]
                                        imwrite(out_rp / f"cam{k}.png", img)
                                        imwrite(out_rp / f"cam{k}_alpha.png", np.clip(a, 0, 1))
                                        write_mask(out_rp / f"cam{k}_valid.png", valid)
                                        imwrite(out_rp / f"cam{k}_alpha_core.png", np.clip(cores[i][1], 0, 1))
                                    save_json(out_rp / "meta.json", dict(meta, reproject=geom, reproject_stats=st,
                                                                         time_render_per_view_s=t_rp))
                del state
        composite_windows(work, setup, sorted(set(win_roots)), targets)


if __name__ == "__main__":
    main()
