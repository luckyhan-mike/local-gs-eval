#!/usr/bin/env python3
"""Step 3: region-wise comparison of flow, local GS and flow+GS hybrids against held-out GT.

Re-run freely after changing the `risk` / `eval` sections of the config; nothing upstream needs to
be recomputed. Methods whose render covers only part of the frame (ROI crops) are scored on the
covered part of each region; the `coverage` column says how much that is.
Outputs: work/eval/metrics.csv, auc.csv, registration.csv, risk_sweep.csv, summary.md,
         <setup>/cam{k}_montage.png
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
from lgs.common import (bbox, cam_path, chroma_fg, cube_mask, dilate, imread_rgb, imwrite,  # noqa: E402
                        load_cfg, load_json, read_mask, setup_cams, to_u8)
from lgs.metrics import LPIPSRegion, auc, psnr, ssim_map  # noqa: E402

REGIONS = ["object", "oracle_fail", "risk", "edge", "rest", "fg"]
REGION_DESC = {
    "object": "物体级区域（真值中的魔方 + 周围 40 px，含手和遮挡带）",
    "oracle_fail": "光流失败区（用真值误差定义，等价于完美检测器）",
    "risk": "自检风险区（仅用光流自身信号、配置中的固定阈值）",
    "edge": "深度边缘带（视差水平梯度）",
    "rest": "前景中其余区域（光流应当可靠）",
    "fg": "整个前景（人+魔方）",
}
SIGNALS = ["r_warp", "r_col", "g_disp", "mask_ent", "mask_grad"]


def load_rgb(p):
    return imread_rgb(str(p)).astype(np.float32) / 255.0


def risk_from_signals(sig, near_fg, warp_t, col_t, rcfg):
    raw = (sig["r_warp"] > warp_t) | (sig["r_col"] > col_t)
    for name, thr in (rcfg.get("extra_rules") or {}).items():
        raw |= sig[name] > thr
    return dilate(raw & near_fg, rcfg["dilate_px"]) & near_fg


def blend(flow, gs, region, gs_ok, sigma):
    w = cv2.GaussianBlur(region.astype(np.float32), (0, 0), sigma) if sigma > 0 else region.astype(np.float32)
    w = np.maximum(w, region.astype(np.float32)) * gs_ok
    return flow * (1 - w[..., None]) + gs * w[..., None]


def _ecc(g_t, g_i, m, motion, init):
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    warp = (init if motion == cv2.MOTION_HOMOGRAPHY else init[:2]).astype(np.float32)
    try:
        _, warp = cv2.findTransformECC(g_t, g_i, warp, motion, crit, m, 5)
    except cv2.error:
        return None
    return warp.astype(np.float64) if warp.shape[0] == 3 else np.vstack([warp, [0, 0, 1]]).astype(np.float64)


def register(gs, gt, mask):
    """Coarse-to-fine ECC (translation -> affine -> homography) aligning a GS render to GT on `mask`.

    Diagnostic only (uses GT). Each stage is kept only if it lowers the masked error at full
    resolution, so a flat or ambiguous mask falls back to the identity instead of drifting.
    Returns (3x3 warp in WARP_INVERSE_MAP convention, mean displacement px over the mask).
    """
    h, w = gt.shape[:2]
    if mask.sum() < 100:
        return None, float("nan")

    def gray(img, s):
        return cv2.resize(cv2.cvtColor(to_u8(img), cv2.COLOR_RGB2GRAY), (int(w * s), int(h * s)),
                          interpolation=cv2.INTER_AREA).astype(np.float32) / 255

    def cost(hm):
        return float(np.mean(np.abs(warp_by(gs, hm) - gt)[mask]))

    best, best_cost = np.eye(3), cost(np.eye(3))
    cur = np.eye(3)
    for s, motion in ((0.25, cv2.MOTION_TRANSLATION), (0.5, cv2.MOTION_AFFINE), (0.5, cv2.MOTION_HOMOGRAPHY)):
        sm = np.diag([s, s, 1.0])
        m = (cv2.resize(mask.astype(np.uint8), (int(w * s), int(h * s)), interpolation=cv2.INTER_NEAREST))
        init = sm @ cur @ np.linalg.inv(sm)
        warp = _ecc(gray(gt, s), gray(gs, s), m, motion, init)
        if warp is None:
            continue
        cand = np.linalg.inv(sm) @ warp @ sm
        c = cost(cand)
        if c < best_cost:
            best, best_cost, cur = cand, c, cand
    ys, xs = np.nonzero(mask)
    sel = np.random.default_rng(0).choice(len(xs), min(len(xs), 20000), replace=False)
    p = np.stack([xs[sel], ys[sel], np.ones(len(sel))])
    q = best @ p
    return best, float(np.linalg.norm(q[:2] / q[2] - p[:2], axis=0).mean())


def warp_by(img, hmat):
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, hmat, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)


def tile(name, img, box, height):
    x0, y0, x1, y1 = box
    t = img[y0:y1, x0:x1]
    s = height / t.shape[0]
    t = cv2.resize(to_u8(np.clip(t, 0, 1)), (max(int(t.shape[1] * s), 1), height))
    cv2.putText(t, name[:34], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    return t


def grid(tiles, cols):
    w = max(t.shape[1] for t in tiles)
    tiles = [np.pad(t, ((0, 0), (0, w - t.shape[1]), (0, 0))) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.concatenate([np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)], 0)


def fmt(v, nd=2):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def gs_base(gv):
    return gv.split("+")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--setups", nargs="+", default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    work = Path(cfg["work_dir"])
    ecfg, rcfg, ck = cfg["eval"], cfg["risk"], cfg["chroma"]
    out_dir = work / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    lp = LPIPSRegion("cuda" if torch.cuda.is_available() else "cpu")
    rows, auc_rows, t_rows, gs_rows, reg_rows, sweep_rows, tuned = [], [], [], [], [], [], {}
    ref = ecfg["ref_flow"]

    for setup in args.setups or list(cfg["setups"]):
        ctx, targets = setup_cams(cfg, setup)
        flow_dir, gs_dir = work / "flow" / setup, work / "gs" / setup
        flow_vars = sorted(p.name for p in flow_dir.iterdir() if p.is_dir()) if flow_dir.exists() else []
        gs_vars = sorted(p.name for p in gs_dir.iterdir() if p.is_dir()) if gs_dir.exists() else []
        if ref not in flow_vars:
            print(f"[{setup}] 缺少参考光流 {ref}，跳过")
            continue
        gs_meta = {gv: load_json(gs_dir / gv / "meta.json") for gv in gs_vars}
        stale = [gv for gv, m in gs_meta.items() if "render" not in m]
        if stale:
            print(f"[{setup}] 跳过第一轮的旧 GS 目录（无渲染变体信息）: {stale}")
            gs_vars = [gv for gv in gs_vars if gv not in stale]
            gs_meta = {gv: gs_meta[gv] for gv in gs_vars}
        for gv, m in gs_meta.items():
            pd = m.get("pose_diag") or m
            gs_rows.append([setup, gv, m.get("pose_source", "vggt"), m["window"][2], m["render_res"], m["render"]["mode"],
                            m["eps2d_used"], m["render"]["filter3d"], m["num_gaussians"], m["time_encode_s"],
                            m["time_render_per_view_s"], float(np.mean(m["ctx_self_psnr"])),
                            max(pd["ctx_rot_err_deg"]), max(pd["ctx_center_err_rel_baseline"])])
        full_frame = [gv for gv in gs_vars if gs_meta[gv]["crop"] != "roi"]
        flow_meta = {fv: load_json(flow_dir / fv / "meta.json") for fv in flow_vars}
        for k in targets:
            t_rows.append([setup, k] + [flow_meta.get(f"flow_s1.0_{m}", {}).get(str(k), {}).get("t", float("nan"))
                                        for m in ("uniform", "pose", "posecal", "oracle")])

        # ---------- pass 1: oracle failure regions and the risk-threshold sweep
        base = {}
        for k in targets:
            gt = load_rgb(cam_path(cfg, "aligned", k))
            fg_gt = read_mask(work / "prepare" / f"fg_aligned_{k:02d}.png")
            valid = np.ones(gt.shape[:2], bool)
            for gv in full_frame:
                valid &= read_mask(gs_dir / gv / f"cam{k}_valid.png")
            pred = load_rgb(flow_dir / ref / f"cam{k}.png")
            with np.load(flow_dir / ref / f"cam{k}_signals.npz") as z:
                sig = {n: z[n].astype(np.float32) for n in SIGNALS}
            fg = fg_gt & valid
            err = cv2.GaussianBlur(np.abs(pred - gt).mean(-1), (0, 0), 1.0)
            oracle = dilate((err > ecfg["fail_err_thresh"]) & fg, 3) & fg
            near_fg = dilate(chroma_fg(pred, ck), rcfg["fg_dilate_px"]) & valid & dilate(fg_gt, 15)
            base[k] = dict(fg_gt=fg_gt, valid=valid, fg=fg, err=err, oracle=oracle, near_fg=near_fg,
                           sig={n: sig[n] for n in ("r_warp", "r_col", "g_disp")})
            label = err > ecfg["fail_err_thresh"]
            for n in SIGNALS:
                auc_rows.append([setup, k, n, auc(sig[n][fg], label[fg]), float(label[fg].mean())])
        sweep = defaultdict(list)
        for wt in ecfg["sweep_warp"]:
            for ct in ecfg["sweep_col"]:
                for k in targets:
                    b = base[k]
                    r = risk_from_signals(b["sig"], b["near_fg"], wt, ct, rcfg)
                    inter, uni = (r & b["oracle"]).sum(), (r | b["oracle"]).sum()
                    sweep[(wt, ct)].append((inter / max(r.sum(), 1), inter / max(b["oracle"].sum(), 1),
                                            inter / max(uni, 1), r.sum() / max(b["oracle"].sum(), 1)))
        for (wt, ct), v in sweep.items():
            v = np.mean(v, 0)
            sweep_rows.append([setup, wt, ct, *v])
        best = max(sweep, key=lambda key: np.mean([x[2] for x in sweep[key]]))
        tuned[setup] = best
        print(f"[{setup}] 风险阈值扫描最优（IoU）: warp={best[0]} col={best[1]}")

        # ---------- pass 2: all methods
        for k in targets:
            b = base[k]
            gt = load_rgb(cam_path(cfg, "aligned", k))
            fg, valid = b["fg"], b["valid"]
            with np.load(flow_dir / ref / f"cam{k}_signals.npz") as z:
                sig = {n: z[n].astype(np.float32) for n in SIGNALS}
            reg = {"fg": fg, "oracle_fail": b["oracle"]}
            reg["risk"] = risk_from_signals(sig, b["near_fg"], rcfg["warp_thresh"], rcfg["col_thresh"], rcfg)
            risk_tuned = risk_from_signals(sig, b["near_fg"], *tuned[setup], rcfg)
            reg["edge"] = dilate(sig["g_disp"] > ecfg["edge_thresh"], ecfg["edge_dilate_px"]) & fg
            reg["rest"] = fg & ~reg["risk"] & ~reg["oracle_fail"]
            cube = cube_mask(gt, ck)
            reg["object"] = dilate(cube, ecfg["object_dilate_px"]) & dilate(b["fg_gt"], 3) & valid

            preds = {fv: load_rgb(flow_dir / fv / f"cam{k}.png") for fv in flow_vars}
            cover = {m: np.ones(gt.shape[:2], bool) for m in preds}
            gs_ok = {}
            for gv in gs_vars:
                p = load_rgb(gs_dir / gv / f"cam{k}.png")
                v = read_mask(gs_dir / gv / f"cam{k}_valid.png")
                a = cv2.imdecode(np.fromfile(str(gs_dir / gv / f"cam{k}_alpha.png"), np.uint8), 0) / 255.0
                ok = (v & (a > 0.5)).astype(np.float32)
                preds[gv], cover[gv], gs_ok[gv] = p, v, ok
                # Fit on the whole foreground except the scored object area, so the object gain is not fitted.
                reg_mask = fg & (ok > 0) & ~reg["object"]
                hmat, disp = register(p, gt, reg_mask)
                if hmat is not None and reg_mask.sum() >= ecfg["reg_min_px"]:
                    g2 = f"{gv}+reg"
                    preds[g2] = warp_by(p, hmat)
                    cover[g2] = warp_by(v.astype(np.float32), hmat) > 0.5
                    gs_ok[g2] = warp_by(ok, hmat)
                # ok = a registration stage was accepted (identity fallback is not counted as success)
                reg_rows.append([setup, k, gv, disp, hmat is not None and not np.allclose(hmat, np.eye(3))])

            for gv in list(gs_ok):
                hybs = {"hyb_oracle_fail": reg["oracle_fail"], "hyb_risk": reg["risk"], "hyb_risktuned": risk_tuned}
                for hname, region in hybs.items():
                    m = f"{hname}[{gv}]"
                    preds[m] = blend(preds[ref], preds[gv], region, gs_ok[gv], ecfg["feather_sigma"])
                    cover[m] = np.ones(gt.shape[:2], bool)

            nfg = max(fg.sum(), 1)
            crop = bbox(dilate(b["fg_gt"], 20), 8)
            x0, y0, x1, y1 = crop
            for meth, p in preds.items():
                smap = np.full(gt.shape[:2], np.nan, np.float32)
                smap[y0:y1, x0:x1] = ssim_map(p[y0:y1, x0:x1], gt[y0:y1, x0:x1])
                for rname in REGIONS:
                    region = reg[rname]
                    m = region & cover[meth]
                    if not m.any():
                        continue
                    rows.append([setup, k, meth, rname, psnr(p, gt, m), float(np.nanmean(smap[m])),
                                 lp(p, gt, m), int(m.sum()), float(region.sum() / nfg),
                                 float(m.sum() / max(region.sum(), 1))])

            # montage around the object / failure area
            box = bbox(reg["object"] | reg["oracle_fail"] & dilate(reg["object"], 120), 40) or bbox(fg, 16)
            x0, y0, x1, y1 = box
            if x1 - x0 < 320:
                c = (x0 + x1) // 2
                x0, x1 = max(c - 160, 0), min(c + 160, gt.shape[1])
            if y1 - y0 < 320:
                c = (y0 + y1) // 2
                y0, y1 = max(c - 160, 0), min(c + 160, gt.shape[0])
            box = (x0, y0, x1, y1)
            tiles = [tile("GT", gt, box, 260), tile(ref, preds[ref], box, 260)]
            if gs_vars:
                best_gv = max(gs_vars, key=lambda g: psnr(preds[g], gt, reg["object"] & cover[g])
                              if (reg["object"] & cover[g]).any() else -1)
                errs = [("|err| flow x3", preds[ref])]
                tiles.append(tile(best_gv, preds[best_gv], box, 260))
                errs.append((f"|err| {best_gv} x3", preds[best_gv]))
                if f"{best_gv}+reg" in preds:
                    tiles.append(tile(f"{best_gv}+reg", preds[f"{best_gv}+reg"], box, 260))
                    errs.append(("|err| +reg x3", preds[f"{best_gv}+reg"]))
                tiles.append(tile(f"hyb_oracle[{best_gv}]", preds[f"hyb_oracle_fail[{best_gv}]"], box, 260))
                tiles.append(tile(f"hyb_risktuned[{best_gv}]", preds[f"hyb_risktuned[{best_gv}]"], box, 260))
                for name, p in errs:
                    tiles.append(tile(name, np.repeat(np.abs(p - gt).mean(-1, keepdims=True) * 3, 3, -1), box, 260))
            ov = gt.copy()
            for m, color in [(reg["oracle_fail"], (1, 0, 0)), (risk_tuned, (0, 0.3, 1))]:
                ov[m] = 0.5 * ov[m] + 0.5 * np.array(color, np.float32)
            ov[dilate(reg["object"], 2) & ~reg["object"]] = (1, 1, 0)
            tiles.append(tile("red=oracle blue=risk_tuned yellow=object", ov, box, 260))
            imwrite(out_dir / setup / f"cam{k}_montage.png", grid(tiles, 5))
            print(f"[{setup}] cam{k}: oracle_fail={reg['oracle_fail'].sum() / nfg:.3f} "
                  f"risk={reg['risk'].sum() / nfg:.3f} risk_tuned={risk_tuned.sum() / nfg:.3f} "
                  f"object={reg['object'].sum() / nfg:.3f}")

    def write_csv(name, header, data):
        with open(out_dir / name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(data)

    write_csv("metrics.csv", ["setup", "target", "method", "region", "psnr", "ssim", "lpips", "area_px",
                              "area_frac_fg", "coverage"], rows)
    write_csv("auc.csv", ["setup", "target", "signal", "auc", "fail_rate_in_fg"], auc_rows)
    write_csv("registration.csv", ["setup", "target", "gs_variant", "mean_disp_px", "ok"], reg_rows)
    write_csv("risk_sweep.csv", ["setup", "warp_thresh", "col_thresh", "precision", "recall", "iou",
                                 "area_ratio"], sweep_rows)
    write_summary(out_dir / "summary.md", rows, auc_rows, t_rows, gs_rows, reg_rows, sweep_rows, tuned, rcfg, ref)
    print(f"done -> {out_dir / 'summary.md'}")


def write_summary(path, rows, auc_rows, t_rows, gs_rows, reg_rows, sweep_rows, tuned, rcfg, ref):
    agg = defaultdict(list)
    area = defaultdict(list)
    for s, k, meth, r, p, ss, l, apx, afr, cov in rows:
        agg[(s, r, meth)].append((p, ss, l, cov))
        area[(s, r)].append(afr)

    def mean(s, r, m, i):
        v = np.array(agg.get((s, r, m), []), dtype=np.float64)
        return float(np.nanmean(v[:, i])) if len(v) else float("nan")

    lines = ["# 局部 GS 质量验证结果汇总（第二轮：Mip 滤波 + ROI + 配准诊断）", "",
             "指标为各留出相机的平均值；PSNR/SSIM 越高越好，LPIPS 越低越好。"
             "`覆盖` = 该方法有渲染结果的像素占区域的比例（ROI 渲染只覆盖局部）。"
             "`+reg` 表示用真值做 ECC 单应配准后的 GS（诊断用上限，不是可部署的方法）。", ""]
    setups = sorted({r[0] for r in rows})
    def pose_src(m):
        return "calibK" if m.endswith("_calK") else "calib" if m.endswith("_cal") else "vggt"

    for s in setups:
        gsv = sorted({m for (ss, r, m) in agg if ss == s and m.startswith("gs_") and "+reg" not in m})
        best_by_src = {}
        for src in ("vggt", "calib", "calibK"):
            cands = [m for m in gsv if pose_src(m) == src]
            if cands:
                best_by_src[src] = max(cands, key=lambda m: mean(s, "object", m, 0))
        lines += [f"## 设置 {s}", "",
                  "按目标位姿来源分别选出物体区 PSNR 最好的 GS 变体：" +
                  "，".join(f"{src} → `{g}`" for src, g in best_by_src.items()) +
                  f"。调优风险阈值（扫描 IoU 最优）：warp={tuned[s][0]}, col={tuned[s][1]}。", "",
                  "### 关键对比", "", "| 方法 | " + " | ".join(f"{r} PSNR / LPIPS" for r in ("object", "oracle_fail", "rest", "fg")) + " |",
                  "|---|" + "---:|" * 4]
        key = ["flow_s1.0_uniform", "flow_s1.0_pose", "flow_s1.0_posecal", "flow_s0.5_pose", "flow_s0.5_posecal", ref]
        for g in best_by_src.values():
            key += [g, f"{g}+reg", f"hyb_oracle_fail[{g}]", f"hyb_risktuned[{g}]",
                    f"hyb_oracle_fail[{g}+reg]", f"hyb_risktuned[{g}+reg]"]
        seen = set()
        for m in key:
            if m in seen or not any((s, r, m) in agg for r in REGIONS):
                continue
            seen.add(m)
            cells = [f"{fmt(mean(s, r, m, 0))} / {fmt(mean(s, r, m, 2), 4)}" for r in ("object", "oracle_fail", "rest", "fg")]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")
        for r in REGIONS:
            fr = np.nanmean(area[(s, r)]) if area[(s, r)] else float("nan")
            lines += [f"### 区域 `{r}`：{REGION_DESC[r]}（平均占前景 {fmt(fr * 100, 1)}%）", "",
                      "| 方法 | PSNR | SSIM | LPIPS | 覆盖 |", "|---|---:|---:|---:|---:|"]
            for m in sorted(mm for (ss, rr, mm) in agg if ss == s and rr == r):
                lines.append(f"| {m} | {fmt(mean(s, r, m, 0))} | {fmt(mean(s, r, m, 1), 4)} | "
                             f"{fmt(mean(s, r, m, 2), 4)} | {fmt(mean(s, r, m, 3), 2)} |")
            lines.append("")

    lines += ["## GS 配准残差（ECC 平移→仿射→单应，前景除物体区；越大说明目标相机位姿/内参误差越大）", "",
              "| 设置 | 变体 | 平均位移 px | 成功率 |", "|---|---|---:|---:|"]
    by = defaultdict(list)
    for s, k, gv, d, ok in reg_rows:
        by[(s, gv)].append((d, ok))
    for (s, gv), v in sorted(by.items()):
        d = np.array([x[0] for x in v], dtype=np.float64)
        lines.append(f"| {s} | {gv} | {fmt(np.nanmean(d) if np.isfinite(d).any() else float('nan'))} | "
                     f"{np.mean([x[1] for x in v]):.2f} |")

    lines += ["", "## 风险阈值扫描（风险区 vs oracle_fail，按 IoU 排序，前 8 名与配置默认值）", "",
              "| 设置 | warp 阈值 | col 阈值 | 精确率 | 召回率 | IoU | 面积比 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for s in sorted({r[0] for r in sweep_rows}):
        sr = sorted([r for r in sweep_rows if r[0] == s], key=lambda r: -r[5])
        default = [r for r in sr if r[1] == rcfg["warp_thresh"] and r[2] == rcfg["col_thresh"]]
        for r in sr[:8] + [d for d in default if d not in sr[:8]]:
            tag = "（默认）" if r in default else ""
            col = "不用" if r[2] >= 1e8 else fmt(r[2], 1)
            lines.append(f"| {s} | {fmt(r[1], 2)}{tag} | {col} | {fmt(r[3], 3)} | {fmt(r[4], 3)} | "
                         f"{fmt(r[5], 3)} | {fmt(r[6], 2)} |")

    lines += ["", "## t 估计对比（scale=1.0）", "",
              "| 设置 | 目标相机 | t_uniform | t_pose (VGGT) | t_posecal (标定) | t_oracle |", "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {s} | {k} | {fmt(a, 3)} | {fmt(b, 3)} | {fmt(c, 3)} | {fmt(d, 3)} |" for s, k, a, b, c, d in t_rows]
    lines += ["", "## AnySplat 运行统计", "",
              "旋转/光心误差：vggt 行是联合估计与输入视点估计的一致性；calib/calibK 行是标定位姿经 Sim(3) 对齐到"
              "AnySplat 输入视点位姿后的残差（反映 AnySplat 自身预测的输入相机几何与真实阵列的偏差）。", "",
              "| 设置 | 变体 | 位姿来源 | 窗口边长 | 渲染分辨率 | 模式 | eps2d | 3D 滤波 | 高斯数 | 编码 s | 渲染 ms/视点 | 输入视点自渲染 PSNR | 最大旋转误差° | 最大光心误差/基线 |",
              "|---|---|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {s} | {g} | {src} | {side} | {res} | {mode} | {fmt(eps, 2)} | {f3} | {n} | {fmt(te, 3)} | "
              f"{fmt(tr * 1000, 1)} | {fmt(cp)} | {fmt(re)} | {fmt(ce, 3)} |"
              for s, g, src, side, res, mode, eps, f3, n, te, tr, cp, re, ce in gs_rows]
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
