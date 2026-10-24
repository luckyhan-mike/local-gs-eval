#!/usr/bin/env python3
"""Step 2: RIFE synthesis of every held-out camera from its two nearest context cameras.

t modes: uniform (index fraction, what the current pipeline does), pose (from AnySplat/VGGT
camera centers, run_anysplat.py must run first), posecal (offline rig calibration, calibrate_rig.py),
oracle (grid search against GT; upper bound).
Outputs: work/flow/<setup>/flow_s<scale>_<tmode>/cam{k}.png, cam{k}_signals.npz, meta.json
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lgs.common import (cam_path, imread_rgb, imwrite, load_cfg, load_json, neighbors,  # noqa: E402
                        read_mask, save_json, setup_cams, write_mask)
from lgs.metrics import psnr  # noqa: E402
from lgs.rife import RIFE  # noqa: E402
from lgs.risk import risk_mask, signals  # noqa: E402


def oracle_t(rife, i0, i1, gt, m, scale):
    def score(t):
        return psnr(rife.synth(i0, i1, t, scale)["pred"], gt, m)

    coarse = np.linspace(0.0, 1.0, 21)
    best = max(coarse, key=score)
    fine = np.clip(np.linspace(best - 0.05, best + 0.05, 11), 0, 1)
    return float(max(fine, key=score))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--setups", nargs="+", default=None)
    ap.add_argument("--scales", nargs="+", type=float, default=[1.0, 0.5])
    ap.add_argument("--tmodes", nargs="+", default=["uniform", "pose", "posecal", "oracle"])
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    work = Path(cfg["work_dir"])
    rife = RIFE(cfg["rife_dir"])

    for setup in args.setups or list(cfg["setups"]):
        ctx, targets = setup_cams(cfg, setup)
        t_src = {}
        for mode, name, hint in (("pose", f"{setup}.json", "run_anysplat.py"),
                                 ("posecal", f"{setup}_calib.json", "calibrate_rig.py")):
            f = work / "poses" / name
            if f.exists():
                t_src[mode] = load_json(f)["t_pose"]
            elif mode in args.tmodes:
                print(f"[{setup}] 未找到 {f}，跳过 t_{mode}（请先运行 {hint}）")
        tmodes = [m for m in args.tmodes if m not in ("pose", "posecal") or m in t_src]
        imgs = {k: imread_rgb(cam_path(cfg, "aligned", k)) for k in range(ctx[0], ctx[-1] + 1)}
        for scale in args.scales:
            for tmode in tmodes:
                variant = f"flow_s{scale}_{tmode}"
                out = work / "flow" / setup / variant
                meta = {}
                for k in targets:
                    a, b = neighbors(ctx, k)
                    gt = imgs[k].astype(np.float32) / 255.0
                    fg = read_mask(work / "prepare" / f"fg_aligned_{k:02d}.png")
                    if tmode == "uniform":
                        t = (k - a) / (b - a)
                    elif tmode in t_src:
                        t = float(np.clip(t_src[tmode][str(k)], 0, 1))
                    else:
                        t = oracle_t(rife, imgs[a], imgs[b], gt, fg, scale)
                    res = rife.synth(imgs[a], imgs[b], t, scale)
                    sig = signals(res, t)
                    risk = risk_mask(sig, res, cfg["risk"], cfg["chroma"])
                    imwrite(out / f"cam{k}.png", res["pred"])
                    write_mask(out / f"cam{k}_risk.png", risk)
                    np.savez_compressed(out / f"cam{k}_signals.npz",
                                        **{n: v.astype(np.float16) for n, v in sig.items()},
                                        flow=res["flow"].astype(np.float16))
                    meta[k] = {"pair": [a, b], "t": t, "fg_psnr": psnr(res["pred"], gt, fg),
                               "risk_frac_of_fg": float((risk & fg).sum() / max(fg.sum(), 1))}
                    print(f"[{setup}] {variant} cam{k} pair={a}-{b} t={t:.3f} "
                          f"fgPSNR={meta[k]['fg_psnr']:.2f} risk={meta[k]['risk_frac_of_fg']:.3f}")
                save_json(out / "meta.json", meta)


if __name__ == "__main__":
    main()
