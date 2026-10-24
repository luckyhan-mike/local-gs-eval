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
from lgs.anysplat_wrap import AnySplatRunner, paste_and_warp  # noqa: E402
from lgs.common import (bbox, cam_path, cube_mask, imread_rgb, imwrite, load_cfg, read_mask,  # noqa: E402
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

                for rv in gcfg["renders"]:
                    res = size if rv["res"] == "input" else side
                    eps2d = rv["eps2d"]
                    if eps2d == "auto":  # keep the 0.3 px dilation of the input grid when rendering denser
                        eps2d = 0.3 * (res / size) ** 2
                    variant = f"{base}_{rv['name']}"
                    out = work / "gs" / setup / variant
                    t0 = time.time()
                    rgb, alpha = runner.render(state, state["tgt_c2w"], state["tgt_intr"], res,
                                               mode=rv["mode"], eps2d=eps2d, filter3d=rv["filter3d"],
                                               filter3d_var=gcfg["filter3d_var"])
                    t_render = (time.time() - t0) / len(targets)
                    for i, k in enumerate(targets):
                        img, a, valid = paste_and_warp(rgb[i], alpha[i], win, raw_shape, hs[k], aligned_shape)
                        imwrite(out / f"cam{k}.png", img)
                        imwrite(out / f"cam{k}_alpha.png", np.clip(a, 0, 1))
                        write_mask(out / f"cam{k}_valid.png", valid)
                    meta = dict(diag, window=list(win), size=size, crop=crop, render=rv, render_res=res,
                                eps2d_used=float(eps2d), time_render_per_view_s=t_render)
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
