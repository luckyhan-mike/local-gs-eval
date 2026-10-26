#!/usr/bin/env python3
"""Export the round-7 deployed GS region (Cn_wide, risk_exp/src/r7_e2e.py) as external tokens for local-gs-eval.

Runs on the Windows machine that holds risk_exp (VGGT, RIFE, the probe CNN), with risk_exp's venv:
    D:/hb/risk_exp/.venv/Scripts/python.exe tools/export_cnw_tokens.py --shots changjing2/shot_0000 ...
Same steps and parameters as r7_e2e.run_once (Cn_wide = (A_wide ∩ B) ∪ Fn, M1 VGGT on the input views,
probes t = 0.25 / 0.5 / 0.75, CNN tau 0.98 or f5 tau 0.16, theta_g by camera gap), except that the pairs are the
adjacent pairs of the given input cameras and the mask is kept per pair (a view between a and b may only
use the region found by that pair). No calibration and no target image is used.
The CNN was trained on the 40 non-pilot shots; use the 8 round-7 pilot shots for a clean test.
Output: data/tokens_cnw/<scene>_<shot>/<det>/pair{a}-{b}/cam{k}.png (raw 2304x1296 masks) and meta.json.
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
import torch.nn.functional as F

RISK = Path(os.environ.get("RISK_EXP_SRC", r"D:/hb/risk_exp/src"))
sys.path.insert(0, str(RISK))
import r7_e2e as E  # noqa: E402
from common import imread, load_rife, pad64, to_tensor  # noqa: E402
from make_samples import features  # noqa: E402
from r4_models import VGGT_H, VGGT_W, VGGTRunner, scale_K  # noqa: E402
from r7_common import DS, RAW_H, RAW_W, K_to_raw, align, dev, upsample_raw  # noqa: E402
from r7_probe import LOG_IDX, cnn_prob, load_cnn  # noqa: E402
from r7_tokens import Geo  # noqa: E402
from r7_tokens3 import footprint, gather_max  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "tokens_cnw"


def run(shot, cams, models, dets):
    vggt, rife, (net, mu, sd_) = models
    pairs = list(zip(cams[:-1], cams[1:]))
    tm = E.Timer()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    raw = {c: imread(os.path.join(DS, shot, f"view_{c}.png")) for c in cams}
    al = {c: align(raw[c], c) for c in cams}
    x = torch.stack([torch.from_numpy(cv2.resize(raw[c], (VGGT_W, VGGT_H), interpolation=cv2.INTER_AREA)[..., ::-1].copy())
                     for c in cams]).permute(0, 3, 1, 2).float().div(255).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = vggt.model(x)
    ext, K = vggt.p2e(pred["pose_enc"], (VGGT_H, VGGT_W))
    depth = pred["depth"][0, ..., 0].float()
    Kr = K_to_raw(scale_K(K[0].float(), VGGT_W, VGGT_H).cpu().numpy())
    w2c = torch.eye(4, device=dev).repeat(len(cams), 1, 1); w2c[:, :3, :4] = ext[0].float()
    d = upsample_raw(depth, "edge")
    geo = Geo("model", {c: d[i] for i, c in enumerate(cams)}, {c: Kr[i] for i, c in enumerate(cams)},
              {c: w2c[i].cpu().numpy() for i, c in enumerate(cams)})
    side = {(a, b): (E.geom(geo, a, b), E.geom(geo, b, a)) for a, b in pairs}
    tens = {c: pad64(to_tensor(al[c])) for c in cams}
    out = {det: {} for det in dets}
    for a, b in pairs:
        sa, sb = side[(a, b)]
        AF = [E.candidates(geo, a, sa, raw[a], b - a, tm), E.candidates(geo, b, sb, raw[b], b - a, tm)]
        score = {det: [None, None] for det in dets}
        for tp in E.PROBES:
            o = rife.inference_extra(tens[a], tens[b], tp)
            for det in dets:
                if det == "cnn":
                    f = features(o, tp)[:, :9, :1088, :1920]
                    f[:, LOG_IDX] = torch.log1p(f[:, LOG_IDX].clamp(min=0))
                    mp = cnn_prob(net, F.avg_pool2d(f, 2)[0], mu, sd_)
                else:
                    f5 = (o["warped_img0"] - o["warped_img1"]).abs().mean(1, keepdim=True)[:, :, :1088, :1920]
                    mp = F.avg_pool2d(F.pad(f5, (2, 2, 2, 2), mode="replicate"), 5, 1)[0, 0]
                mp = F.max_pool2d(mp[None, None], 7, 1, 3)[0, 0].reshape(-1)
                for i, (sdd, w) in enumerate(((sa, tp), (sb, 1 - tp))):
                    v = gather_max(footprint(sdd["q_s"], sdd["q_o"], sdd["z_s"], sdd["z_o"], w), mp)
                    score[det][i] = v if score[det][i] is None else torch.maximum(score[det][i], v)
        for det in dets:
            for i, c in enumerate((a, b)):
                A, Fn = AF[i]
                m = ((A & (score[det][i] > E.TAU[det])) | Fn) & side[(a, b)][i]["src_valid"]
                out[det][(a, b, c)] = m.reshape(RAW_H, RAW_W).cpu().numpy()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", nargs="+", required=True)
    ap.add_argument("--cams", nargs="+", type=int, default=[0, 3, 5])
    ap.add_argument("--dets", nargs="+", default=["cnn", "f5"])
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    models = (VGGTRunner(), load_rife(), load_cnn())
    for shot in args.shots:
        masks, dt = run(shot, args.cams, models, args.dets)
        for det in args.dets:
            root = OUT / shot.replace("/", "_") / det
            area = {}
            for (a, b, c), m in masks[det].items():
                p = root / f"pair{a}-{b}"
                p.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(p / f"cam{c}.png"), m.astype(np.uint8) * 255)
                area[f"{a}-{b}/cam{c}"] = float(m.mean())
            meta = {"shot": shot, "cams": args.cams, "det": det, "scheme": "Cn_wide (r7_e2e: A_wide & B | Fn)",
                    "tau": E.TAU[det], "theta_big": E.TH_BIG, "theta_g": {f"{a}-{b}": E.TH_G[b - a] for a, b in
                                                                         zip(args.cams[:-1], args.cams[1:])},
                    "probes": E.PROBES, "area_frac_of_raw": area,
                    "time_s_all_dets_incl_read": dt, "gpu": torch.cuda.get_device_name()}
            (root / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
            print(shot, det, {k: round(v * 100, 2) for k, v in area.items()}, f"{dt:.2f}s", flush=True)


if __name__ == "__main__":
    main()
