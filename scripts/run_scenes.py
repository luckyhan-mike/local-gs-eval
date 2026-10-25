#!/usr/bin/env python3
"""Run the lean local-GS experiment on several shots of the six-camera dataset and summarise them.

Every shot in cfg["dataset6"]["shots"] (e.g. changjing1/shot_0012) gets its own work directory
work_root/<scene>_<shot> with a filled-in copy of the config, the interpolation time t of every target
from the calibrated camera centres (poses/<setup>.json, so no target image is used), and then the usual
steps: prepare -> detect_tokens -> run_anysplat -> run_flow (reference flow only) -> evaluate ->
validate_results. Afterwards work_root/summary_all.md lists flow / feed-forward / TTO per shot and over
all shots, with the TTO wall times.

Usage: python scripts/run_scenes.py --config configs/dataset6_lean.yaml [--shots changjing/shot_0007 ...]
       [--steps prepare tokens anysplat flow evaluate validate] [--summary-only]
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
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shots", nargs="+")
    ap.add_argument("--steps", nargs="+", default=list(STEPS))
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    data_root = Path(os.environ.get("LGS_DATA_ROOT", base["data_root"]))
    dcfg = base["dataset6"]
    root = Path(dcfg.get("work_root", "work/dataset6"))
    root = root if root.is_absolute() else REPO / root
    shots = args.shots or dcfg["shots"]
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
                    json.dump({"context": ctx, "targets": targets, "t_pose": t, "source": "calibrated centres"},
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
    variant = base["eval"].get("tto_main_variant", "rife")
    pat = re.compile(rf"^(hyb_token|hyb_tokchk_oracle)\[gs_tok{re.escape(variant)}\d+_.+_al(?:_tto(\d+))?_local\]$")
    per = defaultdict(dict)    # (shot, setup, stage, how) -> {(target, region): (psnr, lpips)}
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
            if r["method"] == ref:
                key = (shot, r["setup"], "flow", "—")
            else:
                m = pat.match(r["method"])
                if not m:
                    continue
                key = (shot, r["setup"], int(m.group(2) or 0), "直接" if m.group(1) == "hyb_token" else "oracle")
            per[key][(int(r["target"]), r["region"])] = (float(r["psnr"]), float(r["lpips"]))
        for meta in (work / "gs").glob(f"*/gs_tok{variant}*_al_tto*_local/meta.json"):
            m = json.loads(meta.read_text(encoding="utf-8"))
            setup = meta.parts[-3]
            if m.get("tto"):
                times[(shot, setup)] = {e["iter"]: e["time_s"] for e in m["tto"]["log"] if "time_s" in e}
                times[(shot, setup)]["box"] = (m["tto"]["log"][0].get("box") or {}).get("area_frac")
                times[(shot, setup)]["encode"] = m.get("time_encode_s")
        tf = work / "timing.json"
        if tf.exists():
            times[(shot, "steps")] = json.loads(tf.read_text(encoding="utf-8"))

    def avg(d, region):
        v = [x for (k, r), x in d.items() if r == region]
        return (np.mean([x[0] for x in v]), np.mean([x[1] for x in v]), len(v)) if v else None

    def cell(a, nd=4):
        return f"{a[0]:.2f} / {a[1]:.{nd}f}" if a else "—"

    lines = ["# 六路数据集：局部 GS 验证汇总", "",
             f"token 变体 {variant}、`_al` 对齐域；PSNR / LPIPS 为各目标视点平均；"
             "不差目标数 = fg 的 PSNR 不低于且 LPIPS 不高于纯光流的目标数；oracle 只贴 core ∩ 真值失败区（诊断用）。"
             "object = 真值中最大的高饱和色块膨胀 40 px（没有彩色物体的场景为空，记为 —）。", ""]
    setups = sorted({k[1] for k in per})
    stages = sorted({k[2] for k in per if k[2] != "flow"})
    for setup in setups:
        lines += [f"## {setup}", "", "| 场景 | 方法 | 贴回 | fg PSNR / LPIPS | object PSNR / LPIPS | rest PSNR | 不差目标数 | TTO 累计耗时 |",
                  "|---|---|---|---:|---:|---:|---:|---:|"]
        pooled = defaultdict(dict)
        for shot in shots:
            fl = per.get((shot, setup, "flow", "—"))
            if not fl:
                continue
            for stage, how in [("flow", "—")] + [(s, h) for s in stages for h in ("直接", "oracle")]:
                d = per.get((shot, setup, stage, how))
                if not d:
                    continue
                pooled[(stage, how)].update({(shot, k, r): v for (k, r), v in d.items()})
                ks = sorted({k for (k, r) in d if r == "fg"})
                ok = sum(d[(k, "fg")][0] >= fl[(k, "fg")][0] - 1e-6 and d[(k, "fg")][1] <= fl[(k, "fg")][1] + 1e-6
                         for k in ks if (k, "fg") in fl)
                name = "纯光流" if stage == "flow" else ("前馈 FF" if stage == 0 else f"TTO {stage} 步")
                tt = times.get((shot, setup), {}).get(stage) if stage not in ("flow", 0) else None
                rest = avg(d, "rest")
                lines.append(f"| {shot} | {name} | {how} | {cell(avg(d, 'fg'))} | {cell(avg(d, 'object'))} | "
                             f"{f'{rest[0]:.2f}' if rest else '—'} | {'—' if stage == 'flow' else f'{ok}/{len(ks)}'} | "
                             f"{f'{tt:.2f} s' if tt is not None else '—'} |")
        lines += ["", f"### {setup}：全部场景合计（每个目标视点等权）", "",
                  "| 方法 | 贴回 | fg PSNR / LPIPS | object PSNR / LPIPS | rest PSNR | 不差目标数 |", "|---|---|---:|---:|---:|---:|"]
        fl = pooled.get(("flow", "—"), {})
        for (stage, how), d in sorted(pooled.items(), key=lambda x: (x[0][0] != "flow", x[0][0] if x[0][0] != "flow" else -1, x[0][1])):
            dd = {((s, k), r): v for (s, k, r), v in d.items()}
            ks = sorted({(s, k) for (s, k, r) in d if r == "fg"})
            ok = sum(d[(s, k, "fg")][0] >= fl[(s, k, "fg")][0] - 1e-6 and d[(s, k, "fg")][1] <= fl[(s, k, "fg")][1] + 1e-6
                     for (s, k) in ks if (s, k, "fg") in fl)
            name = "纯光流" if stage == "flow" else ("前馈 FF" if stage == 0 else f"TTO {stage} 步")
            rest = avg(dd, "rest")
            lines.append(f"| {name} | {how} | {cell(avg(dd, 'fg'))} | {cell(avg(dd, 'object'))} | "
                         f"{f'{rest[0]:.2f}' if rest else '—'} | {'—' if stage == 'flow' else f'{ok}/{len(ks)}'} |")
        lines.append("")
    lines += ["## 耗时", "", "| 场景 | 设置 | 编码 | 监督框面积 | TTO 各检查点累计 | 各步骤墙钟时间 |", "|---|---|---:|---:|---|---|"]
    for (shot, setup), t in sorted(times.items()):
        if setup == "steps":
            continue
        cps = ", ".join(f"{it}: {v:.2f}s" for it, v in sorted((k, v) for k, v in t.items() if isinstance(k, int)))
        st = times.get((shot, "steps"), {})
        lines.append(f"| {shot} | {setup} | {t.get('encode') or float('nan'):.2f} s | "
                     f"{(t.get('box') or float('nan')) * 100:.0f}% | {cps} | "
                     + ", ".join(f"{k} {v:.0f}s" for k, v in st.items()) + " |")
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary_all.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"done -> {root / 'summary_all.md'}")


if __name__ == "__main__":
    main()
