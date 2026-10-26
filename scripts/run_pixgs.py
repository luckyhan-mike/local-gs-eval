#!/usr/bin/env python3
"""Step 2b (round 15): pixel-wise Gaussians in the aligned space for the token region (lgs/pixgs.py).

For every setup, every token variant in gs.pixgs.variants and every target k between context cameras a and b:
Gaussians from the recon pixels of a and b (tokens/<setup>/<variant>/pair{a}-{b}/cam*_recon.png, aligned frame), moved by
the stereo correspondences (gs.pixgs.disp, exported with tools/export_fs_disp.py) to the target's t (the same
poses/<setup>.json t as the flow branch), rendered at the aligned resolution. The core pixels alone give the paste-back
alpha. Output gs/<setup>/gs_tok<variant>_pix<size>_pix_al_local/ in the same format as run_anysplat.py's local renders,
so evaluate.py / run_scenes.py treat it as one more local GS variant (<variant>_pix in the summary).
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lgs.common import cam_path, imread_rgb, imwrite, load_cfg, load_json, read_mask, save_json, setup_cams, neighbors, write_mask  # noqa: E402
from lgs.pixgs import depth_sign, fill_background, load_disp, make_set, render_view  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--setups", nargs="+", default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    pcfg = cfg["gs"].get("pixgs") or {}
    if not pcfg.get("enabled"):
        print("gs.pixgs 未启用，跳过")
        return
    work = Path(cfg["work_dir"])
    repo = Path(__file__).resolve().parents[1]
    droot = Path(pcfg["disp"].format(shot=cfg.get("shot", "").replace("/", "_")))
    droot = droot if droot.is_absolute() else repo / droot
    size = cfg["gs"]["sizes"][0]
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    for setup in args.setups or list(cfg["setups"]):
        ctx, targets = setup_cams(cfg, setup)
        t_of = {int(k): v for k, v in load_json(work / "poses" / f"{setup}.json")["t_pose"].items()}
        img = {k: imread_rgb(cam_path(cfg, "aligned", k)).astype(np.float32) / 255.0 for k in ctx}
        shape = img[ctx[0]].shape[:2]
        fg = {k: read_mask(work / "prepare" / f"fg_aligned_{k:02d}.png") for k in ctx}
        disp = {(a, b): load_disp(droot / f"pair{a}-{b}.npz", shape) for a, b in zip(ctx[:-1], ctx[1:])}
        for v in pcfg["variants"]:
            tdir = work / "tokens" / setup / v
            if not tdir.is_dir():
                print(f"[{setup}] 没有 token 变体 {v}，跳过")
                continue
            out = work / "gs" / setup / f"gs_tok{v}_pix{size}_pix_al_local"
            n_total, t_render, stats = 0, [], {}
            for k in targets:
                a, b = neighbors(ctx, k)
                t = float(t_of[k])
                d = disp[(a, b)]
                pdir = tdir / f"pair{a}-{b}"
                rec = {s: read_mask(pdir / f"cam{s}_recon.png") for s in (a, b)}
                core = {s: read_mask(pdir / f"cam{s}_core.png") for s in (a, b)}
                ga, gb = d["ga"], d["gb"]
                sign = depth_sign(ga[0], fg[a], np.isfinite(ga[0]))
                t0 = time.time()
                ga_f = fill_background(ga, rec[a], sign, pcfg.get("fill_px", 96))
                gb_f = fill_background(gb, rec[b], -sign, pcfg.get("fill_px", 96))   # key of b = -sign * g_b,x
                sig, mx = float(pcfg.get("sigma_px", 0.5)), float(pcfg.get("max_stretch_px", 3.0))
                full = [make_set(img[a], ga_f, rec[a], t, sign, sig, mx), make_set(img[b], gb_f, rec[b], 1 - t, -sign, sig, mx)]
                cores = [make_set(img[a], ga_f, core[a], t, sign, sig, mx), make_set(img[b], gb_f, core[b], 1 - t, -sign, sig, mx)]
                rgb, alpha, backend = render_view(full, t, shape, pcfg, device)
                _, acore, _ = render_view(cores, t, shape, pcfg, device, backend)
                t_render.append(time.time() - t0)
                n = sum(len(s[0]) for s in full)
                n_total += n
                stats[str(k)] = {"pair": f"{a}-{b}", "t": t, "gaussians": n,
                                 "missing_a": float((rec[a] & ~np.isfinite(ga_f[0])).sum() / max(rec[a].sum(), 1)),
                                 "missing_b": float((rec[b] & ~np.isfinite(gb_f[0])).sum() / max(rec[b].sum(), 1)),
                                 "depth_sign": sign}
                imwrite(out / f"cam{k}.png", rgb)
                imwrite(out / f"cam{k}_alpha.png", alpha)
                imwrite(out / f"cam{k}_alpha_core.png", acore)
                write_mask(out / f"cam{k}_valid.png", np.ones(shape, bool))
            dmeta = load_json(droot / "meta.json") if (droot / "meta.json").exists() else {}
            meta = {"num_gaussians": n_total, "time_encode_s": float(sum(x.get("time_stereo_s", 0) for x in dmeta.values()
                                                                        if isinstance(x, dict))),
                    "time_encode_note": "stereo (FoundationStereo rec_pp) per pair, measured on the Windows RTX 3090 at export",
                    "ctx_self_psnr": [float("nan")], "ctx_rot_err_deg": [0.0], "ctx_center_err_rel_baseline": [0.0],
                    "window": [0, 0, shape[1]], "size": size, "crop": "token", "token_variant": v, "local": True,
                    "render": {"name": "pix", "res": "aligned", "mode": backend, "eps2d": pcfg.get("eps2d", 0.3), "filter3d": False},
                    "render_res": None, "eps2d_used": float(pcfg.get("eps2d", 0.3)),
                    "time_render_per_view_s": float(np.mean(t_render)), "pose_source": "aligned_int", "pose_diag": None,
                    "tto_iters": 0, "pixgs": stats, "pixgs_cfg": pcfg}
            save_json(out / "meta.json", meta)
            print(f"[{setup}] {out.name}: {n_total} Gaussians over {len(targets)} views, "
                  f"{np.mean(t_render) * 1000:.0f} ms/view ({backend}); "
                  + ", ".join(f"cam{k} t={s['t']:.2f} n={s['gaussians']} miss a/b={s['missing_a']:.2f}/{s['missing_b']:.2f}"
                              for k, s in stats.items()))


if __name__ == "__main__":
    main()
