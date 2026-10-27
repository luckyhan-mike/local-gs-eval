#!/usr/bin/env python3
"""Export two more correspondence sources for round 16 (same format as tools/export_fs_disp.py, same band):

  fss  FoundationStereo ViT-S, pregs_exp P3 fast config "c_s25_it8" (scale 0.25, 8 iterations, one direction + forward-
       projected b, edge-aware upsampling), ~0.19 s per pair on the RTX 3090 -> data/disp_fss/
  flw  RIFE optical flow of the generated view nearest t = 0.5 (rt_token_exp R1 "reuse ... m"): for every pixel x of that
       view, a-position x + F_t->0, displacement F_t->1 - F_t->0, splatted to a's grid (z-buffer, nearest pixel); b from
       the same view. No extra network run in deployment (the view is generated anyway) -> data/disp_flw/

Run with risk_exp's venv on the Windows machine that holds pregs_exp / rt_token_exp:
    D:/hb/risk_exp/.venv/Scripts/python.exe tools/export_disp_more.py --shots changjing1/shot_0021 ... [--only fss|flw]
Unfilled: pixels without an estimate stay nan (lgs fills them, as for disp_fs). meta.json records the time per pair.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_fs_disp as E  # noqa: E402  (sets sys.path for pregs_exp / risk_exp)
from export_fs_disp import H, W, DS, align, imread, rect, cfg_of, C3, band  # noqa: E402

REPO = E.REPO


def pack(ga, gb, shot, a, b, hom, px):
    keep = band(shot, a, b, a, hom[a], px) | band(shot, a, b, b, hom[b], px)
    if not keep.any():
        keep[:1, :1] = True
    ys, xs = np.nonzero(keep)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    out = {}
    for name, g, k in (("ga", ga, a), ("gb", gb, b)):
        g = g.copy()
        g[:, ~band(shot, a, b, k, hom[k], px)] = np.nan
        g = g[:, y0:y1, x0:x1]
        out[name] = np.where(np.isfinite(g), np.clip(np.round(np.nan_to_num(g) * 4), -32767, 32767), -32768).astype(np.int16)
    return out, [x0, y0, x1, y1]


def flow_disp(fl):
    """fl (4,H,W) = F_t->0, F_t->1 of the mid view -> D_a on a's grid, D_b on b's grid (nan = not hit)."""
    pix = torch.stack(torch.meshgrid(torch.arange(W, device=fl.device), torch.arange(H, device=fl.device), indexing="xy"))
    pix = pix.reshape(2, -1).float()
    F0, F1 = fl[:2].reshape(2, -1), fl[2:].reshape(2, -1)
    res = []
    for to_src in (0, 1):
        pos = pix + (F0 if to_src == 0 else F1)
        D = (F1 - F0) if to_src == 0 else (F0 - F1)
        sig = float(torch.sign(D[0].median()))
        key = -sig * D[0]
        x, y = torch.round(pos[0]).long(), torch.round(pos[1]).long()
        v = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        idx = torch.where(v, y * W + x, torch.full_like(x, H * W))
        zb = torch.full((H * W + 1,), -float("inf"), device=fl.device)
        zb.scatter_reduce_(0, idx, torch.where(v, key, torch.full_like(key, -float("inf"))), reduce="amax")
        win = v & (key >= zb[idx] - 0.5)
        out = torch.full((2, H * W + 1), float("nan"), device=fl.device)
        out[:, idx[win]] = D[:, win]
        res.append(out[:, :H * W].reshape(2, H, W).cpu().numpy())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", nargs="+", required=True)
    ap.add_argument("--cams", nargs="+", type=int, default=[0, 3, 5])
    ap.add_argument("--band", type=int, default=72)
    ap.add_argument("--only", choices=["fss", "flw"], default=None)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    from r7_common import HOM7
    from common import load_rife, pad64, to_tensor
    srcs = [args.only] if args.only else ["fss", "flw"]
    cfg = cfg_of("c_s25_it8")
    rife = load_rife() if "flw" in srcs else None
    rect_cache = {}
    for shot in args.shots:
        al = {c: align(imread(os.path.join(DS, shot, f"view_{c}.png")), c) for c in args.cams}
        for src in srcs:
            root = REPO / "data" / f"disp_{src}" / shot.replace("/", "_")
            root.mkdir(parents=True, exist_ok=True)
            meta = {}
            for a, b in zip(args.cams[:-1], args.cams[1:]):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                if src == "fss":
                    if (a, b) not in rect_cache:
                        from r7_common import valid_mask
                        vm = [valid_mask(i).astype(np.uint8) for i in (a, b)]
                        rect_cache[(a, b)] = rect(cv2.cvtColor(al[a], cv2.COLOR_BGR2GRAY), cv2.cvtColor(al[b], cv2.COLOR_BGR2GRAY),
                                                  vm[0], vm[1], a, b)
                    rp = rect_cache[(a, b)]
                    t0 = time.perf_counter()
                    ra = cv2.warpAffine(al[a], rp["Aa"], (W, H), flags=cv2.INTER_CUBIC)
                    rb = cv2.warpAffine(al[b], rp["Ab"], (W, H), flags=cv2.INTER_CUBIC)
                    ga, gb, _, _, _, _ = C3.stereo_pair(ra, rb, rp, cfg)
                    ga, gb = ga.cpu().numpy(), gb.cpu().numpy()
                else:
                    fl = rife.inference_extra(pad64(to_tensor(al[a])), pad64(to_tensor(al[b])), 0.5)["flow"][0, :, :H, :W]
                    ga, gb = flow_disp(fl.float())
                torch.cuda.synchronize(); dt = time.perf_counter() - t0
                out, box = pack(ga, gb, shot, a, b, HOM7, args.band)
                np.savez_compressed(root / f"pair{a}-{b}.npz", box=np.array(box), **out)
                fa = out["ga"][0] != -32768; fb = out["gb"][0] != -32768
                meta[f"{a}-{b}"] = {"time_stereo_s": dt, "box": box, "valid_a": int(fa.sum()), "valid_b": int(fb.sum()),
                                    "median_dx_a": float(np.median(out["ga"][0][fa]) / 4) if fa.any() else None}
            meta.update(shot=shot, cams=args.cams, band_px=args.band, gpu=torch.cuda.get_device_name(),
                        config={"fss": "pregs_exp P3 c_s25_it8 (FoundationStereo ViT-S, scale 0.25, 8 iters, one direction)",
                                "flw": "RIFE flow of the t=0.5 view, F_t->1 - F_t->0 splatted to each source (rt_token_exp R1 reuse-m); "
                                       "time includes the RIFE run, which deployment gets for free"}[src])
            (root / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
            print(src, shot, {k: (round(v["time_stereo_s"], 3), v["valid_a"], v["valid_b"]) for k, v in meta.items() if "-" in k}, flush=True)


if __name__ == "__main__":
    main()
