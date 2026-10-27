#!/usr/bin/env python3
"""Run the lean local-GS experiment on several shots of the six-camera dataset and summarise them.

Every shot in cfg["dataset6"]["shots"] (e.g. changjing1/shot_0012) gets its own work directory
work_root/<scene>_<shot> with a filled-in copy of the config, the interpolation time t of every target
from the calibrated camera centres (poses/<setup>.json, so no target image is used), and then the usual
steps: prepare -> detect_tokens -> run_anysplat -> run_flow (reference flow only) -> run_pixgs -> evaluate ->
validate_results. Afterwards work_root/summary_all.md lists flow / feed-forward / TTO per shot and over
all shots, with the TTO wall times.

Usage: python scripts/run_scenes.py --config configs/dataset6_lean.yaml [--shots changjing/shot_0007 ...]
       [--steps prepare tokens anysplat flow pixgs evaluate validate] [--summary-only]
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from lgs.common import setup_cams  # noqa: E402

STEPS = {
    "prepare": ["scripts/prepare.py", "--config", "{cfg}"],
    "tokens": ["scripts/detect_tokens.py", "--config", "{cfg}"],
    "anysplat": ["scripts/run_anysplat.py", "--config", "{cfg}"],
    "flow": ["scripts/run_flow.py", "--config", "{cfg}", "--scales", "1.0", "--tmodes", "pose"],
    "pixgs": ["scripts/run_pixgs.py", "--config", "{cfg}"],   # after flow: the round-16 paste gate compares with it
    "evaluate": ["scripts/evaluate.py", "--config", "{cfg}"],
    "validate": ["scripts/validate_results.py", "{cfg}"],
}


def calib_t(calib, ctx, targets):
    """t of each target = projection of its calibrated centre onto the segment between its neighbours."""
    c = np.asarray(calib["rig"]["centers_mm"], np.float64)
    out = {}
    for k in targets:
        a, b = max(x for x in ctx if x < k), min(x for x in ctx if x > k)
        d = c[b] - c[a]
        out[str(k)] = float(np.dot(c[k] - c[a], d) / np.dot(d, d))
    return out


def run_step(name, cmd, log_path, env):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        p = subprocess.Popen([sys.executable] + cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in p.stdout:
            sys.stdout.write(line)
            log.write(line)
        p.wait()
    if p.returncode:
        raise RuntimeError(f"step {name} failed (exit {p.returncode}), see {log_path}")
    return time.time() - t0


def shot_config(base, shot, work):
    cfg = json.loads(json.dumps(base))  # deep copy
    cfg["raw_pattern"] = f"{Path(base['dataset6'].get('root', 'datasets')).as_posix()}/{shot}/view_{{}}.png"
    cfg["aligned_pattern"] = str((work / "aligned" / "aligned_{:02d}.png").as_posix())
    cfg["work_dir"] = str(work.as_posix())
    cfg["shot"] = shot  # external token masks are looked up by shot (tokens.variants.*.external)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shots", nargs="+")
    ap.add_argument("--host", help="只跑 dataset6.hosts.<host> 里的场景（如 linux-4090 / linux-3090）")
    ap.add_argument("--steps", nargs="+", default=list(STEPS))
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    data_root = Path(os.environ.get("LGS_DATA_ROOT", base["data_root"]))
    dcfg = base["dataset6"]
    root = Path(dcfg.get("work_root", "work/dataset6"))
    root = root if root.is_absolute() else REPO / root
    shots = args.shots or (dcfg["hosts"][args.host] if args.host else dcfg["shots"])
    with open(data_root / dcfg["calib"], encoding="utf-8") as f:
        calib = json.load(f)
    env = {k: v for k, v in os.environ.items() if k != "LGS_WORK_DIR"}  # every shot uses its own work_dir

    if not args.summary_only:
        for shot in shots:
            work = root / shot.replace("/", "_")
            cfg = shot_config(base, shot, work)
            work.mkdir(parents=True, exist_ok=True)
            cfg_path = work / "config.yaml"
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
            for setup in cfg["setups"]:
                ctx, targets = setup_cams(cfg, setup)
                t = calib_t(calib, ctx, targets)
                (work / "poses").mkdir(exist_ok=True)
                with open(work / "poses" / f"{setup}.json", "w", encoding="utf-8") as f:
                    json.dump({"context": ctx, "targets": targets, "t_pose": t, "source": "calibrated centres (evaluation only: defines where the held-out camera is)"},
                              f, indent=1)
                print(f"[{shot}] {setup}: t from calibration {t}")
            timing = {}
            for step in args.steps:
                print(f"\n===== {shot}: {step} =====", flush=True)
                cmd = [c.format(cfg=cfg_path.as_posix()) for c in STEPS[step]]
                timing[step] = run_step(step, cmd, work / "logs" / f"{step}.log", env)
            with open(work / "timing.json", "w", encoding="utf-8") as f:
                json.dump(timing, f, indent=1)
    summarize(root, shots, base)


def summarize(root, shots, base):
    ref = base["eval"]["ref_flow"]
    main_v = base["eval"].get("tto_main_variant", "rife")
    variants = base["gs"].get("local_variants") or [main_v]
    if (base["gs"].get("multi_window") or {}).get("enabled"):  # round 14: window-by-window encode of the same tokens
        variants = [x for v in variants for x in (v, f"{v}_mw")]
    pix = base["gs"].get("pixgs") or {}
    if pix.get("enabled"):  # round 15: pixel-wise Gaussians in the aligned space
        variants = variants + [f"{v}_pix" for v in pix.get("variants", [])]
        gated = set((pix.get("gate") or {}).get("runs", []))
        variants += [f"{tok}_{r['name']}{s}" for tok in pix.get("run_tokens", []) for r in pix.get("runs", [])
                     for s in ([""] + (["_g"] if r["name"] in gated else []))]  # round 16
    if base["eval"].get("summary_variants"):  # round 17: only the variants of the chosen direction
        variants = list(base["eval"]["summary_variants"])
    size = base["gs"]["sizes"][0]
    pats = {v: re.compile(rf"^(hyb_token|hyb_tokchk_oracle)\[gs_tok{re.escape(v)}{size}_.+_al(?:_tto(\d+))?_local\]$")
            for v in variants}
    gpat = {v: re.compile(rf"^gs_tok{re.escape(v)}{size}_.+_al(?:_tto(\d+))?_local$") for v in variants}
    per = defaultdict(dict)    # (shot, setup, variant, stage, how) -> {(target, region): (psnr, lpips)}
    area = defaultdict(dict)   # (shot, setup, variant, stage, how) -> {target: paste area / fg}
    times = {}
    for shot in shots:
        work = root / shot.replace("/", "_")
        mfile = work / "eval" / "metrics.csv"
        if not mfile.exists():
            print(f"[{shot}] 没有 eval/metrics.csv，跳过汇总")
            continue
        for r in csv.DictReader(mfile.open(encoding="utf-8")):
            if r["region"] not in ("fg", "object", "rest"):
                continue
            keys = []
            if r["method"] == ref:
                keys = [(shot, r["setup"], v, "flow", "—") for v in variants]
            for v, pat in pats.items():
                m = pat.match(r["method"])
                if m:
                    keys.append((shot, r["setup"], v, int(m.group(2) or 0),
                                 "直接" if m.group(1) == "hyb_token" else "oracle"))
            for key in keys:
                per[key][(int(r["target"]), r["region"])] = (float(r["psnr"]), float(r["lpips"]))
        cfile = work / "eval" / "tokchk.csv"
        if cfile.exists():
            for r in csv.DictReader(cfile.open(encoding="utf-8")):
                for v, gp in gpat.items():
                    m = gp.match(r["local_variant"])
                    if m and r["check"] == "oracle":
                        st = int(m.group(1) or 0)
                        area[(shot, r["setup"], v, st, "直接")][int(r["target"])] = float(r["core_frac_fg"])
                        area[(shot, r["setup"], v, st, "oracle")][int(r["target"])] = float(r["paste_frac_fg"])
        for v in variants:
            for meta in (work / "gs").glob(f"*/gs_tok{v}{size}_*_al_tto*_local/meta.json"):
                m = json.loads(meta.read_text(encoding="utf-8"))
                setup = meta.parts[-3]
                if m.get("tto"):
                    key = (shot, setup, v)
                    times[key] = {e["iter"]: e["time_s"] for e in m["tto"]["log"] if "time_s" in e}
                    times[key]["box"] = (m["tto"]["log"][0].get("box") or {}).get("area_frac")
                    times[key]["encode"] = m.get("time_encode_s")
                    times[key]["window"] = "+".join(str(w[2]) for w in m["windows"]) if m.get("windows")                         else (m.get("window") or [None, None, None])[2]
        tf = work / "timing.json"
        if tf.exists():
            times[(shot, "steps")] = json.loads(tf.read_text(encoding="utf-8"))

    def avg(d, region):
        v = [x for (k, r), x in d.items() if r == region]
        return (np.mean([x[0] for x in v]), np.mean([x[1] for x in v]), len(v)) if v else None

    def cell(a, nd=4):
        return f"{a[0]:.2f} / {a[1]:.{nd}f}" if a else "—"

    def name(stage):
        return "纯光流" if stage == "flow" else ("前馈 FF" if stage == 0 else f"TTO {stage} 步")

    def not_worse(d, fl, ks):
        return sum(d[(k, "fg")][0] >= fl[(k, "fg")][0] - 1e-6 and d[(k, "fg")][1] <= fl[(k, "fg")][1] + 1e-6
                   for k in ks if (k, "fg") in fl and (k, "fg") in d)

    lines = ["# 六路数据集：局部 GS 验证汇总", "",
             f"token 变体 {', '.join(variants)}，`_al` 对齐域；PSNR / LPIPS 为各目标视点平均；"
             "不差目标数 = fg 的 PSNR 不低于且 LPIPS 不高于纯光流的目标数；贴回区 = 实际贴回面积占前景比例；"
             "oracle 只贴 core ∩ 真值失败区（诊断用）。object = 真值中最大的高饱和色块膨胀 40 px"
             "（没有彩色物体的场景为空，记为 —）。", ""]
    setups = sorted({k[1] for k in per})
    stages = sorted({k[3] for k in per if k[3] != "flow"})
    rows = [("flow", "—")] + [(s, h) for s in stages for h in ("直接", "oracle")]
    pilot = [x for x in shots if x in set(base["dataset6"].get("pilot_shots") or [])]
    subsets = [("全部场景合计", shots)] + ([("第七轮试点场景合计（检测 CNN 未用它们训练，cnw_cnn 只看这一张表）", pilot)]
                                       if pilot and len(pilot) < len(shots) else [])
    for setup in setups:
        for title, sub in subsets:
            lines += [f"## {setup}：{title}（{len(sub)} 组场景，每个目标视点等权）", "",
                      "| token 变体 | 方法 | 贴回 | 贴回区 | fg PSNR / LPIPS | object PSNR / LPIPS | rest PSNR | 不差目标数 |",
                      "|---|---|---|---:|---:|---:|---:|---:|"]
            for v in variants:
                for stage, how in rows:
                    if stage == "flow" and v != variants[0]:
                        continue
                    d, fl, pa = {}, {}, []
                    for shot in sub:
                        fs = per.get((shot, setup, v, "flow", "—"), {})
                        ds = per.get((shot, setup, v, stage, how), {})
                        d.update({((shot, k), r): x for (k, r), x in ds.items()})
                        fl.update({((shot, k), r): x for (k, r), x in fs.items()})
                        pa += list(area.get((shot, setup, v, stage, how), {}).values())
                    if not d:
                        continue
                    ks = sorted({k for (k, r) in d if r == "fg"})
                    rest = avg(d, "rest")
                    lines.append(f"| {'—' if stage == 'flow' else v} | {name(stage)} | {how} | "
                                 f"{f'{np.mean(pa) * 100:.1f}%' if pa else '—'} | {cell(avg(d, 'fg'))} | "
                                 f"{cell(avg(d, 'object'))} | {f'{rest[0]:.2f}' if rest else '—'} | "
                                 f"{'—' if stage == 'flow' else f'{not_worse(d, fl, ks)}/{len(ks)}'} |")
            lines.append("")
        lines += [f"## {setup}：逐场景", "",
                  "| 场景 | token 变体 | 方法 | 贴回 | 贴回区 | fg PSNR / LPIPS | object PSNR / LPIPS | rest PSNR | 不差目标数 | TTO 累计耗时 |",
                  "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for shot in shots:
            for v in variants:
                fl = per.get((shot, setup, v, "flow", "—"))
                if not fl:
                    continue
                for stage, how in rows:
                    if stage == "flow" and v != variants[0]:
                        continue
                    d = per.get((shot, setup, v, stage, how))
                    if not d:
                        continue
                    ks = sorted({k for (k, r) in d if r == "fg"})
                    pa = list(area.get((shot, setup, v, stage, how), {}).values())
                    tt = times.get((shot, setup, v), {}).get(stage) if stage not in ("flow", 0) else None
                    rest = avg(d, "rest")
                    lines.append(f"| {shot} | {'—' if stage == 'flow' else v} | {name(stage)} | {how} | "
                                 f"{f'{np.mean(pa) * 100:.1f}%' if pa else '—'} | {cell(avg(d, 'fg'))} | "
                                 f"{cell(avg(d, 'object'))} | {f'{rest[0]:.2f}' if rest else '—'} | "
                                 f"{'—' if stage == 'flow' else f'{not_worse(d, fl, ks)}/{len(ks)}'} | "
                                 f"{f'{tt:.2f} s' if tt is not None else '—'} |")
        lines.append("")
    lines += ["## 耗时", "", "| 场景 | 设置 | token 变体 | 编码窗口边长 | 编码 | 监督框面积 | TTO 各检查点累计 | 各步骤墙钟时间 |",
              "|---|---|---|---:|---:|---:|---|---|"]
    for key, t in sorted(times.items(), key=lambda kv: tuple(map(str, kv[0]))):
        if len(key) != 3:
            continue
        shot, setup, v = key
        cps = ", ".join(f"{it}: {x:.2f}s" for it, x in sorted((k, x) for k, x in t.items() if isinstance(k, int)))
        st = times.get((shot, "steps"), {})
        lines.append(f"| {shot} | {setup} | {v} | {t.get('window') or '—'} | {t.get('encode') or float('nan'):.2f} s | "
                     f"{(t.get('box') or float('nan')) * 100:.0f}% | {cps} | "
                     + ", ".join(f"{k} {x:.0f}s" for k, x in st.items()) + " |")
    if base["eval"].get("summary_variants"):
        lines = summary_r17(root, shots, base, variants, size) + lines
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary_all.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"done -> {root / 'summary_all.md'}")


def summary_r17(root, shots, base, variants, size):
    """Round 17 compact table: every target equally weighted, pooled over shots; direct and oracle paste; fg plus the
    flow-failure region split by visibility (fail_vis2 = seen by both input views, fail_vis1 = by one)."""
    ref = base["eval"]["ref_flow"]
    regs = ["fg", "oracle_fail", "fail_vis2", "fail_vis1", "risk"]
    data = defaultdict(dict)   # (shot, target) -> {(method, region): (psnr, lpips)}
    for shot in shots:
        f = root / shot.replace("/", "_") / "eval" / "metrics.csv"
        if f.exists():
            for r in csv.DictReader(f.open(encoding="utf-8")):
                if r["region"] in regs:
                    data[(shot, int(r["target"]))][(r["method"], r["region"])] = (float(r["psnr"]), float(r["lpips"]))
    pat = {v: re.compile(rf"^gs_tok{re.escape(v)}{size}_.+_al(?:_tto100)?_local$") for v in variants}
    gvs = sorted({m[m.index("[") + 1:-1] for d in data.values() for (m, _) in d if m.startswith("hyb_token[")})
    gv_of = {v: next((g for g in gvs if pat[v].match(g)), None) for v in variants}
    pilot = set(base["dataset6"].get("pilot_shots") or [])
    out = ["# 第十七轮汇总（每个目标视点等权；直接 = 不用真值的贴回，oracle = 只贴真值失败区，诊断用）", "",
           "不差 = 该目标 fg 的 PSNR 不低于且 LPIPS 不高于纯光流。失败区按可见性拆开：两侧可见 = 两个输入视角都看得到"
           "（联合优化能约束），单侧可见 = 只有一个输入视角看得到（深度只能靠背景先验）。失败区几栏都是 oracle 贴回的结果。", ""]
    for title, keep in (("全部场景", lambda s: True), ("第七轮试点场景（检测 CNN 未用其训练）", lambda s: s in pilot)):
        ks = [k for k in sorted(data) if keep(k[0]) and (ref, "fg") in data[k]]
        if not ks:
            continue
        out += [f"## {title}（{len(ks)} 个目标视点）", "",
                "| 方法 | 直接 fg PSNR / LPIPS | 直接不差 | oracle fg PSNR / LPIPS | oracle 不差 | 失败区 | 失败区·两侧可见 | 失败区·单侧可见 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|"]

        def cell(m, r):
            v = [data[k][(m, r)] for k in ks if (m, r) in data[k]]
            return f"{np.mean([x[0] for x in v]):.2f} / {np.mean([x[1] for x in v]):.4f}" if v else "—"

        def nw(m):
            n = sum(data[k][(m, "fg")][0] >= data[k][(ref, "fg")][0] - 1e-6
                    and data[k][(m, "fg")][1] <= data[k][(ref, "fg")][1] + 1e-6 for k in ks if (m, "fg") in data[k])
            return f"{n}/{len(ks)}"
        out.append(f"| 纯光流 | {cell(ref, 'fg')} | — | — | — | {cell(ref, 'oracle_fail')} | {cell(ref, 'fail_vis2')} | "
                   f"{cell(ref, 'fail_vis1')} |")
        for v in variants:
            g = gv_of[v]
            if g is None:
                continue
            d, o = f"hyb_token[{g}]", f"hyb_tokchk_oracle[{g}]"
            out.append(f"| {v}{' (TTO 100)' if g.endswith('_tto100_local') else ''} | {cell(d, 'fg')} | {nw(d)} | "
                       f"{cell(o, 'fg')} | {nw(o)} | {cell(o, 'oracle_fail')} | {cell(o, 'fail_vis2')} | {cell(o, 'fail_vis1')} |")
        out.append("")
    return out


if __name__ == "__main__":
    main()
