#!/usr/bin/env python3
"""Export stereo correspondences (aligned frame) for the pixel-GS variant of local-gs-eval (round 15).

Runs on the Windows machine that holds pregs_exp (FoundationStereo, P3 recommended config "rec_pp": ViT-L, scale 0.375,
8 iterations, one direction + forward-projected b, edge-aware upsampling), with risk_exp's venv:
    D:/hb/risk_exp/.venv/Scripts/python.exe tools/export_fs_disp.py --shots changjing1/shot_0021 ...
For each adjacent pair (a, b) of the input cameras: g_a = q_b - q_a for every aligned pixel of a and g_b = q_a - q_b for
b (2 x 1088 x 1920, aligned px, nan = no estimate, e.g. b pixels hidden in a). Calibration-free: the per-rig
rectification correction (pregs_exp P2) comes from image features only.
Only a band around the round-14 Cn_wide GS regions (data/tokens_cnw, both detectors, dilated by --band px) is kept, so
the files stay small: data/disp_fs/<scene>_<shot>/pair{a}-{b}.npz with box (x0, y0, x1, y1) and ga / gb inside it as
int16 quarter pixels (value / 4 = px; -32768 = nan).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, r"D:/hb/pregs_exp/src")
sys.path.insert(0, r"D:/hb/risk_exp/src")
from common import imread  # noqa: E402
from r7_common import DS, H, W, align, valid_mask  # noqa: E402
from p2_rect import rect  # noqa: E402
from p3_eval import cfg_of  # noqa: E402
import p3_core as C3  # noqa: E402

TOK = REPO / "data" / "tokens_cnw"
OUT = REPO / "data" / "disp_fs"


def band(shot, a, b, k, hom, px):
    m = np.zeros((H, W), np.uint8)
    for det in ("cnn", "f5"):
        f = TOK / shot.replace("/", "_") / det / f"pair{a}-{b}" / f"cam{k}.png"
        raw = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        m |= cv2.warpPerspective(raw, hom, (W, H), flags=cv2.INTER_NEAREST)
    return cv2.dilate((m > 127).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1,) * 2)) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", nargs="+", required=True)
    ap.add_argument("--cams", nargs="+", type=int, default=[0, 3, 5])
    ap.add_argument("--band", type=int, default=72)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    from r7_common import HOM7
    cfg = cfg_of("rec_pp")
    rect_cache = {}
    for shot in args.shots:
        al = {c: align(imread(os.path.join(DS, shot, f"view_{c}.png")), c) for c in args.cams}
        root = OUT / shot.replace("/", "_")
        root.mkdir(parents=True, exist_ok=True)
        meta = {}
        for a, b in zip(args.cams[:-1], args.cams[1:]):
            if (a, b) not in rect_cache:
                vm = [valid_mask(i).astype(np.uint8) for i in (a, b)]
                rect_cache[(a, b)] = rect(cv2.cvtColor(al[a], cv2.COLOR_BGR2GRAY), cv2.cvtColor(al[b], cv2.COLOR_BGR2GRAY),
                                          vm[0], vm[1], a, b)
            rp = rect_cache[(a, b)]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            ra = cv2.warpAffine(al[a], rp["Aa"], (W, H), flags=cv2.INTER_CUBIC)
            rb = cv2.warpAffine(al[b], rp["Ab"], (W, H), flags=cv2.INTER_CUBIC)
            ga, gb, _, _, _, _ = C3.stereo_pair(ra, rb, rp, cfg)
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            keep = band(shot, a, b, a, HOM7[a], args.band) | band(shot, a, b, b, HOM7[b], args.band)
            if not keep.any():
                keep[:1, :1] = True
            ys, xs = np.nonzero(keep)
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            out = {}
            for name, g, k in (("ga", ga, a), ("gb", gb, b)):
                g = g.cpu().numpy()
                g[:, ~band(shot, a, b, k, HOM7[k], args.band)] = np.nan
                g = g[:, y0:y1, x0:x1]
                out[name] = np.where(np.isfinite(g), np.clip(np.round(np.nan_to_num(g) * 4), -32767, 32767), -32768).astype(np.int16)
            np.savez_compressed(root / f"pair{a}-{b}.npz", box=np.array([x0, y0, x1, y1]), **out)
            fa = out["ga"][0] != -32768; fb = out["gb"][0] != -32768
            meta[f"{a}-{b}"] = {"time_stereo_s": dt, "box": [x0, y0, x1, y1], "valid_a": int(fa.sum()), "valid_b": int(fb.sum()),
                                "median_dx_a": float(np.median(out["ga"][0][fa]) / 4) if fa.any() else None}
        meta.update(shot=shot, cams=args.cams, config="pregs_exp P3 rec_pp (FoundationStereo ViT-L, scale 0.375, 8 iters, one direction)",
                    band_px=args.band, gpu=torch.cuda.get_device_name())
        (root / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(shot, {k: (round(v["time_stereo_s"], 2), v["valid_a"], v["valid_b"]) for k, v in meta.items() if "-" in k}, flush=True)


if __name__ == "__main__":
    main()
