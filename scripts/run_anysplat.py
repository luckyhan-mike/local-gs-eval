#!/usr/bin/env python3
"""Step 1: AnySplat Gaussians from context cameras, rendered at every held-out camera.

Outputs per setup/variant (aligned image domain, comparable to RIFE results and GT):
  work/gs/<setup>/<variant>/cam{k}.png, cam{k}_valid.png, cam{k}_alpha.png, meta.json, poses.json
The first variant also writes work/poses/<setup>.json, which run_flow.py uses for t_pose.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lgs.anysplat_wrap import AnySplatRunner, paste_and_warp  # noqa: E402
from lgs.common import (bbox, cam_path, imread_rgb, imwrite, load_cfg, read_mask,  # noqa: E402
                        save_json, setup_cams, neighbors, write_mask)


def crop_window(mode, raw_shape, fg_masks):
    h, w = raw_shape
    side_max = min(h, w)
    if mode == "center":
        return ((w - side_max) // 2, (h - side_max) // 2, side_max)
    if mode == "fg":
        x0, y0, x1, y1 = bbox(np.logical_or.reduce(fg_masks))
        side = int(min(max(x1 - x0, y1 - y0) * 1.06, side_max))
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        return (int(np.clip(cx - side // 2, 0, w - side)), int(np.clip(cy - side // 2, 0, h - side)), side)
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
    ap.add_argument("--sizes", nargs="+", type=int, default=[448, 518])
    ap.add_argument("--crops", nargs="+", default=["center", "fg"])
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    work = Path(cfg["work_dir"])
    hs = np.load(work / "prepare" / "H_raw2aligned.npy")
    runner = AnySplatRunner(cfg["anysplat_dir"], cfg["anysplat_ckpt"])

    for setup in args.setups or list(cfg["setups"]):
        ctx, targets = setup_cams(cfg, setup)
        raws = {k: imread_rgb(cam_path(cfg, "raw", k)) for k in ctx + targets}
        raw_shape = raws[ctx[0]].shape[:2]
        aligned_shape = imread_rgb(cam_path(cfg, "aligned", ctx[0])).shape[:2]
        fg_ctx = [read_mask(work / "prepare" / f"fg_raw_{k:02d}.png") for k in ctx]
        first = True
        for crop in args.crops:
            win = crop_window(crop, raw_shape, fg_ctx)
            x0, y0, side = win
            for size in args.sizes:
                assert size % 14 == 0, "AnySplat/VGGT 需要 14 的倍数"
                variant = f"gs_{crop}{size}"
                out = work / "gs" / setup / variant
                crops = {k: cv2.resize(raws[k][y0:y0 + side, x0:x0 + side], (size, size),
                                       interpolation=cv2.INTER_AREA) for k in raws}
                print(f"[{setup}] {variant}: window x0={x0} y0={y0} side={side}")
                res = runner.run([crops[k] for k in ctx], [crops[k] for k in targets], size)
                for i, k in enumerate(targets):
                    rgb, a, valid = paste_and_warp(res["tgt_rgb"][i], res["tgt_alpha"][i], win,
                                                   raw_shape, hs[k], aligned_shape)
                    imwrite(out / f"cam{k}.png", rgb)
                    imwrite(out / f"cam{k}_alpha.png", np.clip(a, 0, 1))
                    write_mask(out / f"cam{k}_valid.png", valid)
                    imwrite(out / f"cam{k}_crop_render.png", res["tgt_rgb"][i])

                order = ctx + targets
                c2w = {k: res["all_c2w_joint"][i] for i, k in enumerate(order)}
                pos, off = rig_positions(c2w, ctx)
                t_pose = {}
                for k in targets:
                    a, b = neighbors(ctx, k)
                    t_pose[k] = (pos[k] - pos[a]) / (pos[b] - pos[a])
                poses = {
                    "context": ctx, "targets": targets,
                    "baseline_pos": {k: pos[k] for k in sorted(pos)},
                    "offset_from_baseline_rel": {k: off[k] for k in sorted(off)},
                    "t_pose": t_pose,
                    "t_uniform": {k: (k - neighbors(ctx, k)[0]) / (neighbors(ctx, k)[1] - neighbors(ctx, k)[0])
                                  for k in targets},
                    "c2w_joint": {k: c2w[k].tolist() for k in sorted(c2w)},
                }
                save_json(out / "poses.json", poses)
                if first:
                    save_json(work / "poses" / f"{setup}.json", dict(poses, from_variant=variant))
                    first = False
                meta = {k: v for k, v in res.items() if k not in ("tgt_rgb", "tgt_alpha", "all_c2w_joint")}
                meta.update(window=list(win), size=size, crop=crop)
                save_json(out / "meta.json", meta)
                print(f"  gaussians={meta['num_gaussians']}  encode={meta['time_encode_s']:.3f}s  "
                      f"render/view={meta['time_render_per_view_s'] * 1000:.1f}ms  "
                      f"ctx self-PSNR={np.mean(meta['ctx_self_psnr']):.2f}dB  "
                      f"pose align rot err={max(meta['ctx_rot_err_deg']):.2f}deg")
                print("  t_pose:", {k: round(v, 3) for k, v in t_pose.items()})


if __name__ == "__main__":
    main()
