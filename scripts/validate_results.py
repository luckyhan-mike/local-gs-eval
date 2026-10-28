"""Check experiment completeness, finite metrics, registration and image artifacts."""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lgs.common import imread_rgb, load_cfg, load_json, setup_cams  # noqa: E402


def main():
    cfg = load_cfg(sys.argv[1] if len(sys.argv) > 1 else "configs/mofang_9cam.yaml")
    work = Path(cfg["work_dir"])
    gcfg = cfg["gs"]
    metrics = list(csv.DictReader((work / "eval/metrics.csv").open(encoding="utf-8")))
    auc_rows = list(csv.DictReader((work / "eval/auc.csv").open(encoding="utf-8")))
    targets_count = sum(len(setup_cams(cfg, s)[1]) for s in cfg["setups"])
    assert len(auc_rows) == targets_count * 5, len(auc_rows)
    keys = {(r["setup"], r["target"], r["method"], r["region"]) for r in metrics}
    assert len(keys) == len(metrics), "Duplicate metric rows"
    for row in metrics:
        assert int(row["area_px"]) > 0, row
        for col in ("psnr", "ssim", "lpips", "area_frac_fg", "coverage"):
            assert math.isfinite(float(row[col])), row
    for row in auc_rows:
        assert 0 <= float(row["auc"]) <= 1, row
    prepare = load_json(work / "prepare/prepare.json")
    assert len(prepare) == cfg["num_cams"]
    assert all(r["warp_psnr_db"] > 30 for r in prepare.values())

    counts = {"gs_renders": 0, "flow_renders": 0, "montages": 0}
    warnings = []
    for setup in cfg["setups"]:
        _, targets = setup_cams(cfg, setup)
        suffix = {"vggt": "", "interp": "_int", "aligned_int": "_al", "aligned_vggt": "_alv", "calib": "_cal", "calibK": "_calK"}
        tok_root = work / "tokens" / setup
        tok_vars = [v for v in (cfg.get("tokens", {}).get("variants") or {}) if (tok_root / v).is_dir()]
        labels = [f"tok{v}" for v in tok_vars] if tok_vars else ["token"]
        fixed = gcfg.get("token_window", "fixed") == "fixed" and tok_vars
        lv = gcfg.get("local_variants")
        if lv and tok_vars:
            labels = [f"tok{v}" for v in tok_vars if v in lv]
        tails = [f"{size}_{rv['name']}{suffix[src]}" for size in gcfg["sizes"] for rv in gcfg["renders"]
                 for src in gcfg.get("pose_sources", ["vggt"])]
        variants = []
        for crop in gcfg["crops"]:
            if crop == "token" and tok_vars:
                # fixed: one encode, full render gs_tok*; per_variant: one encode (and full render) per variant;
                # gs.multi_window (round 14): composited local renders gs_tok<v>_mw*
                if gcfg.get("render_full", True):
                    variants += [f"gs_tok{t}" for t in tails] if fixed else [f"gs_{lab}{t}" for lab in labels for t in tails]
                labs = labels + ([f"{lab}_mw" for lab in labels] if (gcfg.get("multi_window") or {}).get("enabled") else [])
                variants += [f"gs_{lab}{t}_local" for lab in labs for t in tails]
                for it in (gcfg.get("tto") or {}).get("checkpoints") or []:  # TTO checkpoint renders
                    variants += [f"gs_{lab}{t}_tto{it}_local" for lab in labs for t in tails]
                pix = gcfg.get("pixgs") or {}
                if pix.get("enabled"):  # round 15: run_pixgs.py
                    variants += [f"gs_tok{v}_pix{gcfg['sizes'][0]}_pix_al_local" for v in pix.get("variants", [])
                                 if v in tok_vars]
                    gated = set((pix.get("gate") or {}).get("runs", []))
                    for r in pix.get("runs", []):  # round 16: <token>_<run>[_g]
                        for tok in r.get("tokens", pix.get("run_tokens", [])):   # round 18: per-run tokens, _s<n> snapshots
                            if tok in tok_vars:
                                sufs = [""] + (["_g"] if r["name"] in gated else []) + [f"_s{c}" for c in r.get("checkpoints", [])]
                                variants += [f"gs_tok{tok}_{r['name']}{s}{gcfg['sizes'][0]}_{r['method']}_al_local" for s in sufs]
            else:
                full = [f"gs_{lab}{t}" for lab in (labels if crop == "token" else [crop]) for t in tails]
                variants += full + ([v + "_local" for v in full] if tok_root.exists() else [])
        for variant in variants:
            folder = work / "gs" / setup / variant
            meta = load_json(folder / "meta.json")
            if max(meta["ctx_rot_err_deg"]) >= 1 or max(meta["ctx_center_err_rel_baseline"]) >= 0.05:
                warnings.append(f"pose alignment above threshold: {folder.name}")
            for k in targets:
                assert imread_rgb(folder / f"cam{k}.png").shape[2] == 3
                counts["gs_renders"] += 1
        flows = ([cfg["eval"]["ref_flow"]] if cfg.get("eval", {}).get("lean")
                 else [f"flow_s{scale}_{mode}" for scale in (1.0, 0.5) for mode in ("uniform", "pose", "oracle")])
        for flow in flows:
            folder = work / "flow" / setup / flow
            meta = load_json(folder / "meta.json")
            assert set(meta) == {str(k) for k in targets}
            for k in targets:
                with np.load(folder / f"cam{k}_signals.npz") as sig:
                    for name in ("r_warp", "r_col", "g_disp", "mask_ent", "mask_grad", "flow"):
                        assert np.isfinite(sig[name]).all(), (folder, k, name)
                counts["flow_renders"] += 1
        for k in targets:
            imread_rgb(work / "eval" / setup / f"cam{k}_montage.png")
            counts["montages"] += 1
    methods = sorted({r["method"] for r in metrics})
    result = dict(status="PASS", metric_rows=len(metrics), methods=len(methods), auc_rows=len(auc_rows),
                  warnings=warnings, **counts)
    (work / "eval/validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
