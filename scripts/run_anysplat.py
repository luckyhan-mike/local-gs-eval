#!/usr/bin/env python3
"""Step 1: AnySplat Gaussians from context cameras, rendered at every held-out camera.

One encode per (setup, crop, input size); every render variant in `gs.renders` reuses it.
Outputs per setup/variant (aligned image domain, comparable to RIFE results and GT):
  work/gs/<setup>/gs_<crop><size>_<render>/cam{k}.png, cam{k}_valid.png, cam{k}_alpha.png, meta.json
  work/gs/<setup>/gs_<crop><size>_poses.json
work/poses/<setup>.json holds the t shared with run_flow.py; it is written by the first encode only if absent.
With work/tokens/<setup> present (detect_tokens.py), each render also gets a *_local variant: only the
Gaussians inside the token reconstruction region, plus cam{k}_alpha_core.png = footprint of the token core.
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
    centre, then the zero-parallax-plane homography. Returns [(rgb, alpha, valid)] per target."""
    h, w = al["shape"]
    p = al["pad"]
    K = al["K_al"].copy()
    K[0, 2] += p
    K[1, 2] += p
    shift = np.array([[1, 0, -p], [0, 1, -p], [0, 0, 1.0]])
    out = []
    for c in al["centers"][src]:
        cam = al["c2w_al"].copy()
        cam[:3, 3] = c
        rgb, alpha, _ = runner.render_px(state, cam[None], K[None], w + 2 * p, h + 2 * p, keep=keep, **kw)
        G = plane_homography(al["K_al"], al["c2w_al"], c, al["n"], al["d"], (w, h)) @ shift
        img = cv2.warpPerspective(rgb[0], G, (w, h), flags=cv2.INTER_LINEAR)
        a = cv2.warpPerspective(alpha[0], G, (w, h), flags=cv2.INTER_LINEAR)
        valid = cv2.warpPerspective(np.ones((h + 2 * p, w + 2 * p), np.uint8), G, (w, h),
                                    flags=cv2.INTER_NEAREST) > 0
        out.append((np.clip(img, 0, 1), a, valid))
    return out


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
        tok_dir = work / "tokens" / setup
        tokens = None
        if tok_dir.exists():
            tokens = {kind: [read_mask(tok_dir / f"cam{k}_{kind}_raw.png") for k in ctx] for kind in ("core", "recon")}
        # Shared t with the flow branch: reuse the existing pose file (e.g. copied from round 2) if present.
        pose_file = work / "poses" / f"{setup}.json"
        shared_t = {int(k): v for k, v in load_json(pose_file)["t_pose"].items()} if pose_file.exists() else None
        for crop in crops:
            win = crop_window(crop, raw_shape, [raws[k] for k in ctx], fg_ctx, gcfg, cfg["chroma"],
                              tokens["recon"] if tokens else None, tcfg.get("crop_pad_px", 0))
            x0, y0, side = win
            for size in sizes:
                assert size % 14 == 0, "AnySplat/VGGT 需要 14 的倍数"
                base = f"gs_{crop}{size}"
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
                save_json(work / "gs" / setup / f"{base}_poses.json", poses)
                if shared_t is None:
                    save_json(pose_file, dict(poses, from_variant=base))
                    shared_t = t_pose
                print("  t_pose (this encode):", {k: round(v, 3) for k, v in t_pose.items()},
                      " t shared with flow:", {k: round(shared_t[k], 3) for k in targets})

                keep = {}
                if tokens:
                    for kind in ("core", "recon"):
                        masks = [cv2.resize(m[y0:y0 + side, x0:x0 + side].astype(np.uint8), (size, size),
                                            interpolation=cv2.INTER_NEAREST) > 0 for m in tokens[kind]]
                        if gcfg.get("keep_mode", "visible") == "visible":
                            keep[kind] = runner.visible_keep(state, masks, gcfg.get("visible_tol", 0.03))
                        else:
                            keep[kind] = token_keep(state["gaussians"].means[0], state["ctx_c2w"],
                                                    state["ctx_intr"], masks)
                    diag["local_gaussians"] = {kind: int(v.sum()) for kind, v in keep.items()}
                    print(f"  local GS ({gcfg.get('keep_mode', 'visible')}): recon={diag['local_gaussians']['recon']}"
                          f" core={diag['local_gaussians']['core']} of {diag['num_gaussians']} Gaussians")

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
                        name = f"{base}_{rv['name']}{POSE_SUFFIX[src]}"
                        # full: all Gaussians of the crop; local: only the token reconstruction region
                        outputs = {name: (None, None)}
                        if keep:
                            outputs[f"{name}_local"] = (keep["recon"], keep["core"])
                        for variant, (kp, kp_core) in outputs.items():
                            out = work / "gs" / setup / variant
                            t0 = time.time()
                            if src.startswith("aligned"):
                                views = render_aligned(runner, state, al, src, kp, kw)
                                cores = render_aligned(runner, state, al, src, kp_core, kw) if kp_core is not None \
                                    else None
                            else:
                                rgb, alpha = runner.render(state, tgt_c2w, tgt_intr, res, keep=kp, **kw)
                                acore = runner.render(state, tgt_c2w, tgt_intr, res, keep=kp_core, **kw)[1] \
                                    if kp_core is not None else None
                                views = []
                                cores = [] if acore is not None else None
                                for i, k in enumerate(targets):
                                    dmap = None
                                    if use_dist:
                                        dmap = dmaps.setdefault(k, distort_maps(rig[k]["K"], rig[k]["dist"], raw_shape))
                                    views.append(paste_and_warp(rgb[i], alpha[i], win, raw_shape, hs[k],
                                                                aligned_shape, dmap))
                                    if acore is not None:
                                        cores.append(paste_and_warp(rgb[i], acore[i], win, raw_shape, hs[k],
                                                                    aligned_shape, dmap))
                            t_render = (time.time() - t0) / len(targets)
                            for i, k in enumerate(targets):
                                img, a, valid = views[i]
                                imwrite(out / f"cam{k}.png", img)
                                imwrite(out / f"cam{k}_alpha.png", np.clip(a, 0, 1))
                                write_mask(out / f"cam{k}_valid.png", valid)
                                if cores is not None:
                                    imwrite(out / f"cam{k}_alpha_core.png", np.clip(cores[i][1], 0, 1))
                            meta = dict(diag, window=list(win), size=size, crop=crop, render=rv, render_res=res,
                                        eps2d_used=float(eps2d), time_render_per_view_s=t_render, pose_source=src,
                                        pose_diag=pdiag, local=kp is not None)
                            save_json(out / "meta.json", meta)
                            print(f"  {variant}: res={res or 'aligned'} mode={rv['mode']} eps2d={eps2d:.3f} "
                                  f"filter3d={rv['filter3d']} render/view={t_render * 1000:.1f}ms")
                del state


if __name__ == "__main__":
    main()
