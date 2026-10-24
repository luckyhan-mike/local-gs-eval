#!/usr/bin/env python3
"""Step 1: AnySplat Gaussians from context cameras, rendered at every held-out camera.

One encode per (setup, crop, input size); every render variant in `gs.renders` reuses it.
Outputs per setup/variant (aligned image domain, comparable to RIFE results and GT):
  work/gs/<setup>/gs_<crop><size>_<render>/cam{k}.png, cam{k}_valid.png, cam{k}_alpha.png, meta.json
  work/gs/<setup>/gs_<crop><size>_poses.json
The first (crop, size) also writes work/poses/<setup>.json, which run_flow.py uses for t_pose.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lgs.anysplat_wrap import (AnySplatRunner, apply_sim3, distort_maps, paste_and_warp,  # noqa: E402
                               rot_angle_deg, sim3_from_cams)
from lgs.common import (bbox, cam_path, cube_mask, imread_rgb, imwrite, load_cfg, load_json, read_mask,  # noqa: E402
                        save_json, setup_cams, neighbors, write_mask)
from lgs.compat import anysplat_source  # noqa: E402


def square_around(box, raw_shape, pad, min_side):
    h, w = raw_shape
    x0, y0, x1, y1 = box
    side = int(min(max(x1 - x0 + 2 * pad, y1 - y0 + 2 * pad, min_side), min(h, w)))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    return (int(np.clip(cx - side // 2, 0, w - side)), int(np.clip(cy - side // 2, 0, h - side)), side)


def crop_window(mode, raw_shape, ctx_raws, fg_masks, gcfg, ck):
    h, w = raw_shape
    if mode == "center":
        side = min(h, w)
        return ((w - side) // 2, (h - side) // 2, side)
    if mode == "fg":
        x0, y0, x1, y1 = bbox(np.logical_or.reduce(fg_masks))
        return square_around((x0, y0, x1, y1), raw_shape, int(0.03 * max(x1 - x0, y1 - y0)), 0)
    if mode == "roi":
        if gcfg.get("roi_window"):
            return tuple(int(v) for v in gcfg["roi_window"])
        masks = [cube_mask(im, ck) for im in ctx_raws]
        if not all(m.any() for m in masks):
            raise RuntimeError("未在所有输入相机中检测到魔方，请在配置 gs.roi_window 中手动指定 [x0, y0, side]")
        return square_around(bbox(np.logical_or.reduce(masks)), raw_shape, gcfg["roi_pad_px"], gcfg["roi_min_side"])
    raise ValueError(mode)


POSE_SUFFIX = {"vggt": "", "calib": "_cal", "calibK": "_calK"}


def load_rig(work):
    p = work / "calib" / "rig.json"
    if not p.exists():
        return None
    cams = load_json(p)["cameras"]
    return {int(k): {"c2w": np.array(v["c2w"]), "K": np.array(v["K"]), "dist": np.array(v["dist"])}
            for k, v in cams.items()}


def target_poses(state, ctx, targets, win, rig, sources):
    """Target camera poses/intrinsics in the Gaussian frame for each requested pose source.

    vggt:   joint VGGT camera head (round 1-2).
    calib:  offline rig calibration, mapped into the Gaussian frame by a Sim(3) fitted on the
            context cameras; intrinsics still from VGGT (consistent with how the Gaussians were built).
    calibK: as calib, plus calibrated focal/principal point and lens distortion.
    Returns {source: (c2w, intr_norm, use_distortion, diagnostics)}.
    """
    out = {"vggt": (state["tgt_c2w"], state["tgt_intr"], False, None)}
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

    for setup in args.setups or list(cfg["setups"]):
        ctx, targets = setup_cams(cfg, setup)
        raws = {k: imread_rgb(cam_path(cfg, "raw", k)) for k in ctx + targets}
        raw_shape = raws[ctx[0]].shape[:2]
        aligned_shape = imread_rgb(cam_path(cfg, "aligned", ctx[0])).shape[:2]
        fg_ctx = [read_mask(work / "prepare" / f"fg_raw_{k:02d}.png") for k in ctx]
        first = True
        for crop in crops:
            win = crop_window(crop, raw_shape, [raws[k] for k in ctx], fg_ctx, gcfg, cfg["chroma"])
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

                poses_by_src = target_poses(state, ctx, targets, win, rig, gcfg["pose_sources"])
                for src, (tgt_c2w, tgt_intr, use_dist, pdiag) in poses_by_src.items():
                    if pdiag:
                        print(f"  pose source {src}: ctx rot err after Sim3={max(pdiag['ctx_rot_err_deg']):.2f}deg "
                              f"center err/baseline={max(pdiag['ctx_center_err_rel_baseline']):.3f}")
                    for rv in gcfg["renders"]:
                        res = size if rv["res"] == "input" else side
                        eps2d = rv["eps2d"]
                        if eps2d == "auto":  # keep the 0.3 px dilation of the input grid when rendering denser
                            eps2d = 0.3 * (res / size) ** 2
                        variant = f"{base}_{rv['name']}{POSE_SUFFIX[src]}"
                        out = work / "gs" / setup / variant
                        t0 = time.time()
                        rgb, alpha = runner.render(state, tgt_c2w, tgt_intr, res, mode=rv["mode"], eps2d=eps2d,
                                                   filter3d=rv["filter3d"], filter3d_var=gcfg["filter3d_var"],
                                                   filter3d_comp=gcfg.get("filter3d_comp", False))
                        t_render = (time.time() - t0) / len(targets)
                        for i, k in enumerate(targets):
                            dmap = dmaps.setdefault(k, distort_maps(rig[k]["K"], rig[k]["dist"], raw_shape)) \
                                if use_dist else None
                            img, a, valid = paste_and_warp(rgb[i], alpha[i], win, raw_shape, hs[k],
                                                           aligned_shape, dmap)
                            imwrite(out / f"cam{k}.png", img)
                            imwrite(out / f"cam{k}_alpha.png", np.clip(a, 0, 1))
                            write_mask(out / f"cam{k}_valid.png", valid)
                        meta = dict(diag, window=list(win), size=size, crop=crop, render=rv, render_res=res,
                                    eps2d_used=float(eps2d), time_render_per_view_s=t_render, pose_source=src,
                                    pose_diag=pdiag)
                        save_json(out / "meta.json", meta)
                        print(f"  {variant}: res={res} mode={rv['mode']} eps2d={eps2d:.3f} "
                              f"filter3d={rv['filter3d']} render/view={t_render * 1000:.1f}ms")

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
                if first:
                    save_json(work / "poses" / f"{setup}.json", dict(poses, from_variant=base))
                    first = False
                print("  t_pose:", {k: round(v, 3) for k, v in t_pose.items()})
                del state


if __name__ == "__main__":
    main()
