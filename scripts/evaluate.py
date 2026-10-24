#!/usr/bin/env python3
"""Step 3: region-wise comparison of flow, local GS and flow+GS hybrids against held-out GT.

CPU-light (LPIPS uses the GPU if available). Re-run freely after changing the `risk` / `eval`
sections of the config; nothing upstream needs to be recomputed.
Outputs: work/eval/metrics.csv, auc.csv, summary.md, <setup>/cam{k}_montage.png
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lgs.common import (bbox, cam_path, chroma_fg, dilate, imread_rgb, imwrite, load_cfg,  # noqa: E402
                        load_json, read_mask, setup_cams, to_u8)
from lgs.metrics import LPIPSRegion, auc, psnr, ssim_map  # noqa: E402

REGIONS = ["fg", "oracle_fail", "risk", "edge", "rest"]
REGION_DESC = {
    "fg": "整个前景（人+魔方）",
    "oracle_fail": "光流失败区（用真值误差定义，等价于完美检测器）",
    "risk": "自检风险区（仅用光流自身信号、固定阈值）",
    "edge": "深度边缘带（视差水平梯度）",
    "rest": "前景中其余区域（光流应当可靠）",
}
SIGNALS = ["r_warp", "r_col", "g_disp", "mask_ent", "mask_grad"]


def load_rgb(p):
    return imread_rgb(str(p)).astype(np.float32) / 255.0


def risk_from_signals(sig, pred, rcfg, ck):
    raw = (sig["r_warp"] > rcfg["warp_thresh"]) | (sig["r_col"] > rcfg["col_thresh"])
    for name, thr in (rcfg.get("extra_rules") or {}).items():
        raw |= sig[name] > thr
    near_fg = dilate(chroma_fg(pred, ck), rcfg["fg_dilate_px"])
    return dilate(raw & near_fg, rcfg["dilate_px"]) & near_fg


def blend(flow, gs, region, gs_ok, sigma, gain_ring=0):
    gs = gs.copy()
    if gain_ring > 0:
        ring = dilate(region, gain_ring) & ~region & (gs_ok > 0)
        if ring.sum() > 100:
            gain = flow[ring].mean(0) / np.maximum(gs[ring].mean(0), 1e-4)
            gs = np.clip(gs * np.clip(gain, 0.8, 1.25), 0, 1)
    w = cv2.GaussianBlur(region.astype(np.float32), (0, 0), sigma) if sigma > 0 else region.astype(np.float32)
    w = np.maximum(w, region.astype(np.float32)) * gs_ok
    return flow * (1 - w[..., None]) + gs * w[..., None]


def montage(gt, panels, regions_overlay, box, height=300):
    x0, y0, x1, y1 = box
    tiles = []
    for name, img in [("GT", gt)] + panels:
        tiles.append((name, img[y0:y1, x0:x1]))
    ov = gt.copy()
    for m, color in regions_overlay:
        ov[m] = 0.5 * ov[m] + 0.5 * np.array(color, np.float32)
    tiles.append(("red=oracle_fail blue=risk", ov[y0:y1, x0:x1]))
    out = []
    for name, t in tiles:
        s = height / t.shape[0]
        t = cv2.resize(to_u8(np.clip(t, 0, 1)), (max(int(t.shape[1] * s), 1), height))
        cv2.putText(t, name, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        out.append(t)
    row = len(out) // 2 + len(out) % 2
    w = max(t.shape[1] for t in out)
    pad = [np.pad(t, ((0, 0), (0, w - t.shape[1]), (0, 0))) for t in out]
    while len(pad) < 2 * row:
        pad.append(np.zeros_like(pad[0]))
    return np.concatenate([np.concatenate(pad[:row], 1), np.concatenate(pad[row:], 1)], 0)


def fmt(v, nd=2):
    return "—" if v is None or np.isnan(v) else f"{v:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--setups", nargs="+", default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    work = Path(cfg["work_dir"])
    ecfg, ck = cfg["eval"], cfg["chroma"]
    out_dir = work / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    lp = LPIPSRegion("cuda" if torch.cuda.is_available() else "cpu")
    rows, auc_rows, t_rows, gs_rows = [], [], [], []

    for setup in args.setups or list(cfg["setups"]):
        ctx, targets = setup_cams(cfg, setup)
        flow_dir, gs_dir = work / "flow" / setup, work / "gs" / setup
        flow_vars = sorted(p.name for p in flow_dir.iterdir() if p.is_dir()) if flow_dir.exists() else []
        gs_vars = sorted(p.name for p in gs_dir.iterdir() if p.is_dir()) if gs_dir.exists() else []
        ref = ecfg["ref_flow"]
        if ref not in flow_vars:
            print(f"[{setup}] 缺少参考光流 {ref}，跳过")
            continue
        for gv in gs_vars:
            m = load_json(gs_dir / gv / "meta.json")
            gs_rows.append([setup, gv, m["num_gaussians"], m["time_encode_s"], m["time_render_per_view_s"],
                            float(np.mean(m["ctx_self_psnr"])), max(m["ctx_rot_err_deg"]),
                            max(m["ctx_center_err_rel_baseline"])])
        flow_meta = {fv: load_json(flow_dir / fv / "meta.json") for fv in flow_vars}
        for k in targets:
            t_rows.append([setup, k] + [flow_meta.get(f"flow_s1.0_{m}", {}).get(str(k), {}).get("t", float("nan"))
                                        for m in ("uniform", "pose", "oracle")])

        for k in targets:
            gt = load_rgb(cam_path(cfg, "aligned", k))
            fg_gt = read_mask(work / "prepare" / f"fg_aligned_{k:02d}.png")
            preds = {fv: load_rgb(flow_dir / fv / f"cam{k}.png") for fv in flow_vars}
            gs_ok, valid = {}, np.ones(gt.shape[:2], bool)
            for gv in gs_vars:
                preds[gv] = load_rgb(gs_dir / gv / f"cam{k}.png")
                v = read_mask(gs_dir / gv / f"cam{k}_valid.png")
                a = cv2.imdecode(np.fromfile(str(gs_dir / gv / f"cam{k}_alpha.png"), np.uint8), 0) / 255.0
                gs_ok[gv] = (v & (a > 0.5)).astype(np.float32)
                valid &= v
            sig = {n: np.load(flow_dir / ref / f"cam{k}_signals.npz")[n].astype(np.float32) for n in SIGNALS}

            err_ref = cv2.GaussianBlur(np.abs(preds[ref] - gt).mean(-1), (0, 0), 1.0)
            fg = fg_gt & valid
            reg = {"fg": fg}
            reg["oracle_fail"] = dilate((err_ref > ecfg["fail_err_thresh"]) & fg, 3) & fg
            reg["risk"] = risk_from_signals(sig, preds[ref], cfg["risk"], ck) & valid & dilate(fg_gt, 15)
            reg["edge"] = dilate(sig["g_disp"] > ecfg["edge_thresh"], ecfg["edge_dilate_px"]) & fg
            reg["rest"] = fg & ~reg["risk"] & ~reg["oracle_fail"]

            for gv in gs_vars:
                for rname in ("risk", "oracle_fail"):
                    preds[f"hyb_{rname}[{gv}]"] = blend(preds[ref], preds[gv], reg[rname], gs_ok[gv],
                                                        ecfg["feather_sigma"])
                preds[f"hyb_risk_gain[{gv}]"] = blend(preds[ref], preds[gv], reg["risk"], gs_ok[gv],
                                                      ecfg["feather_sigma"], ecfg["ring_px"])

            nfg = max(fg.sum(), 1)
            for meth, p in preds.items():
                smap = ssim_map(p, gt)
                for rname in REGIONS:
                    m = reg[rname]
                    rows.append([setup, k, meth, rname, psnr(p, gt, m),
                                 float(smap[m].mean()) if m.any() else float("nan"),
                                 lp(p, gt, m), int(m.sum()), float(m.sum() / nfg)])

            label = err_ref > ecfg["fail_err_thresh"]
            for n in SIGNALS:
                auc_rows.append([setup, k, n, auc(sig[n][fg], label[fg]), float(label[fg].mean())])

            box = bbox(reg["oracle_fail"] | reg["risk"], 48) or bbox(fg, 16)
            x0, y0, x1, y1 = box
            if x1 - x0 < 256:
                c = (x0 + x1) // 2
                x0, x1 = max(c - 128, 0), min(c + 128, gt.shape[1])
            if y1 - y0 < 256:
                c = (y0 + y1) // 2
                y0, y1 = max(c - 128, 0), min(c + 128, gt.shape[0])
            panels = [(ref, preds[ref])]
            if gs_vars:
                best_gv = max(gs_vars, key=lambda g: psnr(preds[g], gt, reg["oracle_fail"]) if reg["oracle_fail"].any() else 0)
                panels += [(best_gv, preds[best_gv]), (f"hyb_risk[{best_gv}]", preds[f"hyb_risk[{best_gv}]"]),
                           ("|err| flow x3", np.repeat(np.abs(preds[ref] - gt).mean(-1, keepdims=True) * 3, 3, -1)),
                           ("|err| gs x3", np.repeat(np.abs(preds[best_gv] - gt).mean(-1, keepdims=True) * 3, 3, -1))]
            imwrite(out_dir / setup / f"cam{k}_montage.png",
                    montage(gt, panels, [(reg["oracle_fail"], (1, 0, 0)), (reg["risk"], (0, 0.3, 1))], (x0, y0, x1, y1)))
            print(f"[{setup}] cam{k}: fg={fg.sum()} oracle_fail={reg['oracle_fail'].sum() / nfg:.3f} "
                  f"risk={reg['risk'].sum() / nfg:.3f}")

    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["setup", "target", "method", "region", "psnr", "ssim", "lpips", "area_px", "area_frac_fg"])
        w.writerows(rows)
    with open(out_dir / "auc.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["setup", "target", "signal", "auc", "fail_rate_in_fg"])
        w.writerows(auc_rows)
    write_summary(out_dir / "summary.md", rows, auc_rows, t_rows, gs_rows)
    print(f"done -> {out_dir / 'summary.md'}")


def write_summary(path, rows, auc_rows, t_rows, gs_rows):
    agg = defaultdict(list)
    area = defaultdict(list)
    for s, k, meth, r, p, ss, l, apx, afr in rows:
        agg[(s, r, meth)].append((p, ss, l))
        area[(s, r)].append(afr)
    lines = ["# 局部 GS 质量验证结果汇总", "",
             "指标为各留出相机的平均值；PSNR/SSIM 越高越好，LPIPS 越低越好。", ""]
    for s in sorted({r[0] for r in rows}):
        lines += [f"## 设置 {s}", ""]
        for r in REGIONS:
            fr = np.nanmean(area[(s, r)]) if area[(s, r)] else float("nan")
            lines += [f"### 区域 `{r}`：{REGION_DESC[r]}（平均占前景 {fmt(fr * 100, 1)}%）", "",
                      "| 方法 | PSNR | SSIM | LPIPS |", "|---|---:|---:|---:|"]
            meths = sorted(m for (ss, rr, m) in agg if ss == s and rr == r)
            for m in meths:
                v = np.array(agg[(s, r, m)], dtype=np.float64)
                lines.append(f"| {m} | {fmt(np.nanmean(v[:, 0]))} | {fmt(np.nanmean(v[:, 1]), 4)} | "
                             f"{fmt(np.nanmean(v[:, 2]), 4)} |")
            lines.append("")
    lines += ["## t 估计对比（scale=1.0）", "", "| 设置 | 目标相机 | t_uniform | t_pose | t_oracle |",
              "|---|---:|---:|---:|---:|"]
    lines += [f"| {s} | {k} | {fmt(a, 3)} | {fmt(b, 3)} | {fmt(c, 3)} |" for s, k, a, b, c in t_rows]
    lines += ["", "## AnySplat 运行统计", "",
              "| 设置 | 变体 | 高斯数 | 编码耗时 s | 渲染 ms/视点 | 输入视点自渲染 PSNR | 位姿对齐最大旋转误差° | 最大光心误差/基线 |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {s} | {g} | {n} | {fmt(te, 3)} | {fmt(tr * 1000, 1)} | {fmt(cp)} | {fmt(re)} | {fmt(ce, 3)} |"
              for s, g, n, te, tr, cp, re, ce in gs_rows]
    lines += ["", "## 光流自身信号预测“光流失败”的 AUC（标签 = oracle_fail，0.5 = 随机）", "",
              "| 设置 | 信号 | 平均 AUC | 前景失败率 |", "|---|---|---:|---:|"]
    by = defaultdict(list)
    for s, k, n, a, fr in auc_rows:
        by[(s, n)].append((a, fr))
    for (s, n), v in sorted(by.items()):
        v = np.array(v, dtype=np.float64)
        lines.append(f"| {s} | {n} | {fmt(np.nanmean(v[:, 0]), 3)} | {fmt(np.nanmean(v[:, 1]), 3)} |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
