#!/usr/bin/env python3
"""Dense-view wall clock (round 18): N virtual views between every adjacent input pair, token repair pasted in, one idle
GPU. Everything stays on the GPU; every timed block is bracketed by torch.cuda.synchronize().

Per shot and pair (a, b) of the setup, views at t = k / (N + 1):
  RIFE      seq32: one view at a time in fp32 (rounds 16/17); b16: views in batches of --batch with fp16 autocast
  once      displacement prep + fused set build on the GPU (lgs/pixfuse.py, bg plane if --bgplane), then the joint
            optimisation (--steps; 0 = pix3n), cropped to the recon box
  per view  repair render (cropped to the Gaussians' box) and the feathered paste into the RIFE frame
Not timed here: token detection (rt_token_exp R4 F4R-q: about 28 ms per frame for 3 cameras, RTX 3090) and the
correspondences (FoundationStereo, exported on the Windows RTX 3090; their time is read from the export meta.json).
Output: <work_root>/bench_dense_<host>.json / .md

    python scripts/bench_dense.py --config configs/dataset6_lean.yaml --host linux-4090 --views 30
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
from lgs.pixgs import depth_sign, load_disp  # noqa: E402
from lgs.rife import RIFE  # noqa: E402
from lgs import pixfuse  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def rife_batch(rife, x0, x1, ts, h, w, half):
    """IFNet at several t in one batch (timestep tensor [B,1,1,1]) -> (B, 3, h, w) frames on the GPU."""
    B = len(ts)
    x = torch.cat((x0, x1), 1).expand(B, -1, -1, -1)
    tt = torch.tensor(ts, device=x.device, dtype=x.dtype).view(B, 1, 1, 1)
    with torch.autocast("cuda", dtype=torch.float16, enabled=half):
        flow_list, mask_logit, _ = rife.net(x, tt, [16, 8, 4, 2, 1])
    flow = flow_list[-1].float()
    m = torch.sigmoid(mask_logit.float())
    xb0, xb1 = x0.expand(B, -1, -1, -1), x1.expand(B, -1, -1, -1)
    pred = rife.warp(xb0, flow[:, :2]) * m + rife.warp(xb1, flow[:, 2:4]) * (1 - m)
    return pred[:, :, :h, :w].clamp(0, 1)


def paste(frame, rgb, core, alpha, feather):
    """Feathered core blend as evaluate.py (paste_weight), on the GPU. frame (3, H, W), rgb (H, W, 3)."""
    c = (core > 0.5) & (alpha > 0.5)
    k = feather // 2 * 2 + 1
    dil = F.max_pool2d(c.float()[None, None], k, 1, k // 2)
    wgt = torch.maximum(F.avg_pool2d(dil, k, 1, k // 2), c.float()[None, None])[0, 0] * (alpha > 0.5)
    return frame * (1 - wgt) + rgb.permute(2, 0, 1) * wgt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--host", default=None, help="take the shots of dataset6.hosts[host]")
    ap.add_argument("--shots", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--views", nargs="+", type=int, default=[30])
    ap.add_argument("--token", default="f4rq")
    ap.add_argument("--steps", nargs="+", type=int, default=[0, 50, 100], help="joint-optimisation steps (0 = pix3n)")
    ap.add_argument("--disp", default="data/disp_fs/{shot}")
    ap.add_argument("--bgplane", default="data/bgplane/{room}", help="'' to switch the background plane off")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--setup", default="in035")
    ap.add_argument("--warmup", type=int, default=2)
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
    torch.backends.cudnn.benchmark = True
    rife = RIFE(str(data_root / base["rife_dir"]) if not os.path.isabs(base["rife_dir"]) else base["rife_dir"], dev)
    try:
        import gsplat  # noqa: F401
        backend = "gsplat" if dev == "cuda" else "torch"
    except ImportError:
        backend = "torch"
    eps2d = float(pcfg.get("eps2d", 0.3))
    feather = int(base["tokens"]["paste_feather_px"])
    lr = pcfg.get("opt_lr")
    report = {"host": args.host or socket.gethostname(), "gpu": torch.cuda.get_device_name() if dev == "cuda" else "cpu",
              "backend": backend, "token": args.token, "disp": args.disp, "bgplane": args.bgplane, "batch": args.batch, "shots": {}}
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
            T = {}
            add = lambda k, v: T.setdefault(k, []).append(v)
            for a, b in zip(ctx[:-1], ctx[1:]):
                x0, x1 = pad(img[a]), pad(img[b])
                for _ in range(args.warmup):
                    rife_batch(rife, x0, x1, [0.5], h, w, False); rife_batch(rife, x0, x1, ts[:args.batch], h, w, True)
                sync(); t0 = time.perf_counter()
                for t in ts:
                    rife_batch(rife, x0, x1, [t], h, w, False)
                sync(); add("rife_seq32_s", time.perf_counter() - t0)
                frames = []
                sync(); t0 = time.perf_counter()
                for i in range(0, N, args.batch):
                    frames.append(rife_batch(rife, x0, x1, ts[i:i + args.batch], h, w, True))
                frames = torch.cat(frames)
                sync(); add("rife_b16_s", time.perf_counter() - t0)
                add("corr_export_s", float(dmeta.get(f"{a}-{b}", {}).get("time_stereo_s", float("nan"))))
                pdir = work / "tokens" / args.setup / args.token / f"pair{a}-{b}"
                rec = {s: read_mask(pdir / f"cam{s}_recon.png") for s in (a, b)}
                core = {s: read_mask(pdir / f"cam{s}_core.png") for s in (a, b)}
                d = load_disp(droot / f"pair{a}-{b}.npz", (h, w))
                sign = depth_sign(d["ga"][0], fg[a], np.isfinite(d["ga"][0]))
                bg = None
                if args.bgplane:
                    z = np.load(REPO / args.bgplane.format(room=shot.split("/")[0]) / f"pair{a}-{b}.npz")
                    bg = (torch.as_tensor(z["ga"].astype(np.float32), device=dev), torch.as_tensor(z["gb"].astype(np.float32), device=dev))
                for steps in args.steps:
                    tag = f"pix3_s{steps}" if steps else "pix3n"
                    sync(); t0 = time.perf_counter()
                    G = pixfuse.build(img[a], img[b], d["ga"], d["gb"], rec[a], rec[b], core[a], core[b], sign,
                                      float(pcfg.get("edge_px", 1.0)), True, dev, bg=bg)
                    S = pixfuse.make(G, float(pcfg.get("sigma_px", 0.5)), float(pcfg.get("max_stretch_px", 3.0)), dev)
                    sync(); add(f"{tag}|build_s", time.perf_counter() - t0)
                    ol, _ = pixfuse.optimise(S, img[a], img[b], (h, w), steps, lr, eps2d, backend)
                    add(f"{tag}|opt_s", ol["time_s"])
                    for i, t in enumerate([0.5] * args.warmup + ts):
                        sync(); t1 = time.perf_counter()
                        rgb, alpha, acore, _ = pixfuse.render_view(S, t, (h, w), eps2d, backend, to_numpy=False)
                        sync(); t2 = time.perf_counter()
                        out = paste(frames[max(i - args.warmup, 0)], rgb, acore, alpha, feather)
                        sync(); t3 = time.perf_counter()
                        if i >= args.warmup:
                            add(f"{tag}|render_s", t2 - t1); add(f"{tag}|paste_s", t3 - t2)
                    del out, S, G
                del frames
            nv = N * (len(ctx) - 1)
            r = {"views": nv, "rife_seq32_s": float(np.sum(T["rife_seq32_s"])), "rife_b16_s": float(np.sum(T["rife_b16_s"])),
                 "corr_export_s": float(np.nansum(T["corr_export_s"]))}
            for steps in args.steps:
                tag = f"pix3_s{steps}" if steps else "pix3n"
                r[tag] = {"build_s": float(np.sum(T[f"{tag}|build_s"])), "opt_s": float(np.sum(T[f"{tag}|opt_s"])),
                          "render_ms": float(np.mean(T[f"{tag}|render_s"]) * 1e3), "paste_ms": float(np.mean(T[f"{tag}|paste_s"]) * 1e3)}
                q = r[tag]
                q["total_s"] = r["rife_b16_s"] + q["build_s"] + q["opt_s"] + nv * (q["render_ms"] + q["paste_ms"]) / 1e3
            res[f"N{N}"] = r
            print(f"[{shot}] N={N}/pair ({nv} views): RIFE seq fp32 {r['rife_seq32_s']:.2f} s -> batch fp16 {r['rife_b16_s']:.2f} s; "
                  + "; ".join(f"{k}: build {v['build_s']:.2f} + opt {v['opt_s']:.2f} s, {v['render_ms']:.1f}+{v['paste_ms']:.1f} ms/view, "
                              f"total {v['total_s']:.2f} s" for k, v in r.items() if isinstance(v, dict)), flush=True)
        report["shots"][shot] = res
    out = root / f"bench_dense_{report['host']}"
    out.with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    med = lambda f: float(np.median([f(r) for r in report["shots"].values()]))
    lines = [f"# 密集视点计时（{report['host']}，{report['gpu']}，渲染 {backend}，token {args.token}，对应 {args.disp}，"
             f"背景平面 {args.bgplane or '关'}）", "",
             f"每对相邻输入相机之间 N 个虚拟视点（in035 共 2 对）；{len(report['shots'])} 组场景的中位数。全部在 GPU 上；"
             "token（R4 F4R-q，3 台相机约 28 ms / 帧）和对应关系（FoundationStereo，Windows RTX 3090 导出，单列）不在本表计时内。", ""]
    for N in args.views:
        k = f"N{N}"
        lines += [f"## N = {N} / 对（共 {med(lambda r: r[k]['views']):.0f} 个虚拟视点）", "",
                  f"RIFE：逐个视点 fp32 {med(lambda r: r[k]['rife_seq32_s']):.2f} s → 批量（{args.batch}）fp16 "
                  f"**{med(lambda r: r[k]['rife_b16_s']):.2f} s**；对应关系（立体，Windows 导出）{med(lambda r: r[k]['corr_export_s']):.2f} s", "",
                  "| 方法 | 建立 (s) | 联合优化 (s) | 渲染 (ms/视点) | 贴回 (ms/视点) | 合计含批量 RIFE (s) |", "|---|---:|---:|---:|---:|---:|"]
        for steps in args.steps:
            tag = f"pix3_s{steps}" if steps else "pix3n"
            lines.append(f"| {tag} | {med(lambda r: r[k][tag]['build_s']):.3f} | {med(lambda r: r[k][tag]['opt_s']):.3f} | "
                         f"{med(lambda r: r[k][tag]['render_ms']):.2f} | {med(lambda r: r[k][tag]['paste_ms']):.2f} | "
                         f"{med(lambda r: r[k][tag]['total_s']):.2f} |")
        lines.append("")
    lines.append(f"逐场景数据见 {out.name}.json。")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
