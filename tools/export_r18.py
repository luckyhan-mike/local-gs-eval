#!/usr/bin/env python3
"""Round 18 exports (Windows machine with rt_token_exp; risk_exp venv):

1. F4R-q tokens (rt_token_exp R4 recommendation: R1 flow geometry, reused probe views, CNN retrained on 1/4-res features,
   5-fold CV by shot). Every shot uses the net of the fold it was HELD OUT of, with (theta_big, tau) chosen on that fold's
   training shots (r4_choice.json; the in035 g2 pair (3,5) uses the g3 choice, as in R4). No VGGT, no calibration.
   Saved like tools/export_cnw_tokens.py: data/tokens_f4rq/<scene>_<shot>/f4rq/pair{a}-{b}/cam{k}.png on the RAW frame
   (aligned mask warped back with the inverse round-7 homography), so scripts/detect_tokens.py "external" reads it unchanged.
2. Boot background plane (rt_token_exp R3 "bgboot"): DIS on one boot frame of the room, affine fit of the far 60 % of the
   displacement, per (room, a, b) and (room, b, a). Static wall and rig; computed once per room, reused by every shot.
   data/bgplane/<room>/pair{a}-{b}.npz with ga, gb (2 x H x W float16, aligned px, a->b and b->a).

    D:/hb/risk_exp/.venv/Scripts/python.exe tools/export_r18.py --shots changjing1/shot_0001 ...
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
sys.path.insert(0, r"D:/hb/rt_token_exp/src")
import r1_common  # noqa: E402,F401  (sets sys.path for pregs_exp / risk_exp)
from r1_common import HOM7, RAW_H, RAW_W  # noqa: E402
from r4_tokfn import make_tok_r4  # noqa: E402
from r3_render import boot_plane, BOOT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", nargs="+", required=True)
    ap.add_argument("--cams", nargs="+", type=int, default=[0, 3, 5])
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    tok = make_tok_r4({"version": "F4R-q"})
    pairs = list(zip(args.cams[:-1], args.cams[1:]))
    for shot in args.shots:
        root = REPO / "data" / "tokens_f4rq" / shot.replace("/", "_") / "f4rq"
        meta = {"shot": shot, "cams": args.cams, "scheme": "rt_token_exp R4 F4R-q (held-out fold net, CV-chosen theta/tau)"}
        for a, b in pairs:
            g = abs(b - a)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            ma, mb = tok(shot, g, a, b)
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            for k, m in ((a, ma), (b, mb)):
                raw = cv2.warpPerspective(m.cpu().numpy().astype(np.uint8) * 255, np.linalg.inv(HOM7[k]), (RAW_W, RAW_H),
                                          flags=cv2.INTER_NEAREST)
                (root / f"pair{a}-{b}").mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(root / f"pair{a}-{b}" / f"cam{k}.png"), raw)
                meta[f"{a}-{b}/cam{k}_area_aligned"] = float(m.float().mean())
            meta[f"{a}-{b}/time_s_incl_probe_views"] = dt
        (root / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(shot, {k: round(v, 4) for k, v in meta.items() if isinstance(v, float)}, flush=True)
    for room in sorted({s.split("/")[0] for s in args.shots}):
        out = REPO / "data" / "bgplane" / room
        out.mkdir(parents=True, exist_ok=True)
        for a, b in pairs:
            ga, gb = boot_plane(f"{room}/any", a, b), boot_plane(f"{room}/any", b, a)   # keyed by room only
            np.savez_compressed(out / f"pair{a}-{b}.npz", ga=ga.cpu().numpy().astype(np.float16), gb=gb.cpu().numpy().astype(np.float16))
        (out / "meta.json").write_text(json.dumps({"room": room, "boot_frame": BOOT[room],
                                                   "method": "rt_token_exp R3 boot_plane: DIS on the boot frame, affine fit of the far 60 %"},
                                                  indent=1), encoding="utf-8")
        print("bgplane", room, BOOT[room], flush=True)


if __name__ == "__main__":
    main()
