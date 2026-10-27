#!/usr/bin/env python3
"""Round 16 stage C: wall-clock time of a dense set of virtual views with the token repair pasted in (one GPU, idle).

For each shot (its round-16 work dir must exist: aligned images, tokens/<setup>/<token>), each adjacent input pair (a, b)
of the setup and N views per pair at t = k / (N + 1):
  once per pair   correspondences loaded (in deployment: stereo / flow, time taken from the export meta.json),
                  displacement prep on the GPU (lgs/pixgs.py prep_side_torch), primitives built once (PixSet2 /
                  stereo_synth.prepare)
  per view        RIFE at t (network only, tensors stay on the GPU), repair render (pix2: gsplat or the torch splat;
                  p3h: stereo_synth.render), paste = feathered core blend into the RIFE frame on the GPU
Every timed block is bracketed by torch.cuda.synchronize(). Nothing is written except the report:
work_root/bench_dense_<host>.json and .md. Token detection is not run here (R1 F4M: 28.7 ms per frame for 3 cameras,
measured in rt_token_exp); it is listed from that report in the .md.

    python scripts/bench_dense.py --config configs/dataset6_lean.yaml --host linux-4090 --views 41 60
"""
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lgs.common import imread_rgb, read_mask  # noqa: E402
from lgs.pixgs import PixSet2, depth_sign, load_disp, prep_side_torch, render_view2  # noqa: E402
from lgs.rife import RIFE  # noqa: E402
from lgs import stereo_synth  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Clock:
    def __init__(self):
        self.t = {}

    def add(self, name, dt):
        self.t.setdefault(name, []).append(dt)


def rife_frame(rife, x0, x1, t, h, w):
    """IFNet at t on padded GPU tensors -> (3, h, w) frame, kept on the GPU (as lgs/rife.py synth, scale 1)."""
    flow_list, mask_logit, _ = rife.net(torch.cat((x0, x1), 1), float(t), [16, 8, 4, 2, 1])
    flow = flow_list[-1]
    m = torch.sigmoid(mask_logit)
    pred = rife.warp(x0, flow[:, :2]) * m + rife.warp(x1, flow[:, 2:4]) * (1 - m)
    return pred[0, :, :h, :w].clamp(0, 1)


def paste(frame, rgb, core, alpha, feather):
    """Feathered core blend as evaluate.py (paste_weight), on the GPU."""
    c = (core > 0.5) & (alpha > 0.5)
    k = feather // 2 * 2 + 1
    dil = F.max_pool2d(c.float()[None, None], k, 1, k // 2)
    wgt = torch.maximum(F.avg_pool2d(dil, k, 1, k // 2), c.float()[None, None])[0, 0] * (alpha > 0.5)
    return frame * (1 - wgt) + rgb.permute(2, 0, 1) * wgt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--host", default=None, help="take the shots of dataset6.hosts[host] (first --n of them)")
    ap.add_argument("--shots", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--views", nargs="+", type=int, default=[41, 60])
    ap.add_argument("--token", default="cnw_cnn")
    ap.add_argument("--methods", nargs="+", default=["pix2", "p3h"])
    ap.add_argument("--disp", default="data/disp_fs/{shot}")
    ap.add_argument("--setup", default="in035")
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()
    base = yaml.safe_load(open(args.config, encoding="utf-8"))
    dcfg = base["dataset6"]
    root = Path(dcfg.get("work_root", "work/dataset6"))
    root = root if root.is_absolute() else REPO / root
    shots = args.shots or (dcfg["hosts"][args.host][:args.n] if args.host else dcfg["shots"][:args.n])
    pcfg = base["gs"]["pixgs"]
    ctx = [int(c) for c in base["setups"][args.setup]["context"]]
    data_root = Path(os.environ.get("LGS_DATA_ROOT", base["data_root"]))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)
    rife = RIFE(str(data_root / base["rife_dir"]) if not os.path.isabs(base["rife_dir"]) else base["rife_dir"], dev)
    try:
        import gsplat  # noqa: F401
        backend = "gsplat" if dev == "cuda" else "torch"
    except ImportError:
        backend = "torch"
    feather = int(base["tokens"]["paste_feather_px"])
    report = {"host": args.host or socket.gethostname(), "gpu": torch.cuda.get_device_name() if dev == "cuda" else "cpu",
              "backend_pix2": backend, "token": args.token, "disp": args.disp, "setup": args.setup, "shots": {}}
    for shot in shots:
        work = root / shot.replace("/", "_")
        droot = Path(args.disp.format(shot=shot.replace("/", "_")))
        droot = droot if droot.is_absolute() else REPO / droot
        img = {k: imread_rgb(work / "aligned" / f"aligned_{k:02d}.png").astype(np.float32) / 255.0 for k in ctx}
        h, w = img[ctx[0]].shape[:2]
        fg = {k: read_mask(work / "prepare" / f"fg_aligned_{k:02d}.png") for k in ctx}
        dmeta = json.loads((droot / "meta.json").read_text(encoding="utf-8")) if (droot / "meta.json").exists() else {}
        mult = 128
        pad = lambda x: F.pad(torch.as_tensor(x, device=dev).permute(2, 0, 1)[None],
                              (0, (mult - w % mult) % mult, 0, (mult - h % mult) % mult), mode="replicate")
        res = {}
        for N in args.views:
            ts = [k / (N + 1) for k in range(1, N + 1)]
            for method in args.methods:
                ck = Clock()
                for a, b in zip(ctx[:-1], ctx[1:]):
                    pdir = work / "tokens" / args.setup / args.token / f"pair{a}-{b}"
                    rec = {s: read_mask(pdir / f"cam{s}_recon.png") for s in (a, b)}
                    core = {s: read_mask(pdir / f"cam{s}_core.png") for s in (a, b)}
                    d = load_disp(droot / f"pair{a}-{b}.npz", (h, w))
                    sign = depth_sign(d["ga"][0], fg[a], np.isfinite(d["ga"][0]))
                    x0, x1 = pad(img[a]), pad(img[b])
                    for _ in range(args.warmup):   # cuDNN / gsplat warm-up, not timed
                        rife_frame(rife, x0, x1, 0.5, h, w)
                    sync(); t0 = time.perf_counter()
                    ga, oka = prep_side_torch(d["ga"], sign, dev)
                    gb, okb = prep_side_torch(d["gb"], -sign, dev)
                    ga, gb, oka, okb = (x.cpu().numpy() for x in (ga, gb, oka, okb))
                    if method == "pix2":
                        sig, mx, e = float(pcfg.get("sigma_px", 0.5)), float(pcfg.get("max_stretch_px", 3.0)), float(pcfg.get("edge_px", 1.0))
                        S = [PixSet2(img[a], ga, oka, core[a], rec[a], sign, sig, mx, e, dev),
                             PixSet2(img[b], gb, okb, core[b], rec[b], -sign, sig, mx, e, dev)]
                    else:
                        P = stereo_synth.prepare(img[a], img[b], ga, gb, oka, okb, core[a], core[b], rec[a], rec[b], sign, dev)
                    sync(); ck.add("once_prep_build_s", time.perf_counter() - t0)
                    ck.add("once_correspondence_export_s", float(dmeta.get(f"{a}-{b}", {}).get("time_stereo_s", float("nan"))))
                    for i, t in enumerate([0.5] * args.warmup + ts):
                        sync(); t0 = time.perf_counter()
                        frame = rife_frame(rife, x0, x1, t, h, w)
                        sync(); t1 = time.perf_counter()
                        if method == "pix2":
                            rgb, alpha, acore, _ = render_view2([S[0].at(t), S[1].at(1 - t)], t, (h, w), pcfg, dev, backend,
                                                                to_numpy=False)
                        else:
                            rgb, alpha, acore = stereo_synth.render(P, t, to_numpy=False)
                        sync(); t2 = time.perf_counter()
                        out = paste(frame, rgb, acore, alpha, feather)
                        sync(); t3 = time.perf_counter()
                        if i >= args.warmup:
                            ck.add("view_rife_s", t1 - t0); ck.add("view_repair_s", t2 - t1); ck.add("view_paste_s", t3 - t2)
                    del out
                n_views = len(ts) * (len(ctx) - 1)
                once = float(np.sum(ck.t["once_prep_build_s"]))
                corr = float(np.nansum(ck.t["once_correspondence_export_s"]))
                per = {k: float(np.mean(v)) for k, v in ck.t.items() if k.startswith("view_")}
                tot = once + n_views * sum(per.values())
                res[f"{method}_N{N}"] = {"views": n_views, "once_prep_build_s": once, "once_correspondence_s_windows3090": corr,
                                         **{f"{k}_mean": v for k, v in per.items()},
                                         **{f"{k}_p90": float(np.quantile(ck.t[k], 0.9)) for k in per},
                                         "total_s_excl_correspondence": tot, "rife_only_s": n_views * per["view_rife_s"]}
                print(f"[{shot}] {method} N={N}/pair ({n_views} views): once {once:.3f} s + per view "
                      f"rife {per['view_rife_s'] * 1e3:.1f} / repair {per['view_repair_s'] * 1e3:.1f} / paste {per['view_paste_s'] * 1e3:.1f} ms"
                      f" -> {tot:.2f} s (RIFE alone {res[f'{method}_N{N}']['rife_only_s']:.2f} s; correspondences {corr:.2f} s on Windows)",
                      flush=True)
        report["shots"][shot] = res
    out = root / f"bench_dense_{report['host']}"
    out.with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    keys = sorted({k for r in report["shots"].values() for k in r})
    lines = [f"# 密集视点计时（{report['host']}，{report['gpu']}，pix2 渲染 {backend}，token {args.token}，对应 {args.disp}）", "",
             "每对输入相机生成 N 个视点（in035 共 2 对）。一次性 = 每对一次的位移整理和建高斯 / 建立合成数据（GPU）；"
             "对应关系（立体 / 光流）本身在 Windows RTX 3090 上导出，单列；token 标记见 rt_token_exp R1（3 台相机 28.7 ms / 帧），不在此计时。", "",
             "| 方法 | N/对 | 视点数 | 一次性 (s) | RIFE (ms/视点) | 修补渲染 (ms/视点) | 贴回 (ms/视点) | 合计 (s) | 只用 RIFE (s) | 对应关系 (s, Windows) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in keys:
        rs = [r[k] for r in report["shots"].values() if k in r]
        m = lambda f: float(np.median([r[f] for r in rs]))
        method, n = k.split("_N")
        lines.append(f"| {method} | {n} | {rs[0]['views']} | {m('once_prep_build_s'):.3f} | {m('view_rife_s_mean') * 1e3:.1f} | "
                     f"{m('view_repair_s_mean') * 1e3:.1f} | {m('view_paste_s_mean') * 1e3:.1f} | {m('total_s_excl_correspondence'):.2f} | "
                     f"{m('rife_only_s'):.2f} | {m('once_correspondence_s_windows3090'):.2f} |")
    lines += ["", f"各列为 {len(report['shots'])} 组场景的中位数；逐场景数据见 {out.name}.json。"]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
