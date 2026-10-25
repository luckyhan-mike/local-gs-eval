#!/usr/bin/env python3
"""Step 0: estimate raw->aligned homographies and foreground masks for every camera.

With cfg["homographies"] (a calibrated (N, 3, 3) raw->aligned .npy relative to data_root, e.g. the
six-camera dataset) the aligned images are generated here by warping the raw views to
cfg["aligned_size"] (default: the raw size), instead of being read from disk and matched with SIFT."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lgs.common import (cam_path, chroma_fg, imread_rgb, imwrite, load_cfg,  # noqa: E402
                        save_json, write_mask)


def estimate_h(raw, aligned):
    sift = cv2.SIFT_create(nfeatures=10000)
    g0 = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
    g1 = cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY)
    k0, d0 = sift.detectAndCompute(g0, None)
    k1, d1 = sift.detectAndCompute(g1, None)
    matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64)).knnMatch(d0, d1, k=2)
    good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n.distance]
    p0 = np.float32([k0[m.queryIdx].pt for m in good])
    p1 = np.float32([k1[m.trainIdx].pt for m in good])
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    h, inl = cv2.findHomography(p0, p1, method, 1.0, maxIters=20000, confidence=0.9999)
    return h, int(inl.sum()), len(good)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    out = Path(cfg["work_dir"]) / "prepare"
    hs, report = [], {}
    given = np.load(Path(cfg["data_root"]) / cfg["homographies"]) if cfg.get("homographies") else None
    for k in range(cfg["num_cams"]):
        raw = imread_rgb(cam_path(cfg, "raw", k))
        if given is not None:
            h, ninl, nmatch = given[k].astype(np.float64), 0, 0
            w_al, h_al = cfg.get("aligned_size") or (raw.shape[1], raw.shape[0])
            al = cv2.warpPerspective(raw, h, (int(w_al), int(h_al)), flags=cv2.INTER_LINEAR)
            imwrite(cam_path(cfg, "aligned", k), al)
        else:
            al = imread_rgb(cam_path(cfg, "aligned", k))
            h, ninl, nmatch = estimate_h(raw, al)
        hh, ww = al.shape[:2]
        warped = cv2.warpPerspective(raw, h, (ww, hh), flags=cv2.INTER_LINEAR)
        valid = cv2.warpPerspective(np.ones(raw.shape[:2], np.uint8), h, (ww, hh)) > 0
        valid &= cv2.erode((al.max(-1) > 8).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        mse = np.mean((warped[valid].astype(np.float32) - al[valid]) ** 2) / 255.0 ** 2
        psnr = float(-10 * np.log10(max(mse, 1e-10)))
        hs.append(h)
        fg_raw, fg_al = chroma_fg(raw, cfg["chroma"]), chroma_fg(al, cfg["chroma"])
        write_mask(out / f"fg_raw_{k:02d}.png", fg_raw)
        write_mask(out / f"fg_aligned_{k:02d}.png", fg_al)
        imwrite(out / f"check_warp_{k:02d}.png", np.concatenate([warped, al], 1)[::2, ::2])
        report[k] = {"matches": nmatch, "inliers": ninl, "warp_psnr_db": psnr,
                     "fg_frac_aligned": float(fg_al.mean()),
                     "raw_size": list(raw.shape[:2]), "aligned_size": list(al.shape[:2])}
        flag = "" if psnr > 30 else "   <-- 低于 30 dB，请检查 check_warp 图"
        print(f"cam {k}: inliers {ninl}/{nmatch}, raw->aligned PSNR {psnr:.2f} dB{flag}")
    np.save(out / "H_raw2aligned.npy", np.stack(hs))
    save_json(out / "prepare.json", report)


if __name__ == "__main__":
    main()
