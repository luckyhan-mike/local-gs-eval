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
        crop_labels = [lab for crop in gcfg["crops"] for lab in (labels if crop == "token" else [crop])]
        variants = [f"gs_{lab}{size}_{rv['name']}{suffix[src]}" for lab in crop_labels for size in gcfg["sizes"]
                    for rv in gcfg["renders"] for src in gcfg.get("pose_sources", ["vggt"])]
        if (work / "tokens" / setup).exists():
            variants += [v + "_local" for v in variants]
        for variant in variants:
            folder = work / "gs" / setup / variant
            meta = load_json(folder / "meta.json")
            if max(meta["ctx_rot_err_deg"]) >= 1 or max(meta["ctx_center_err_rel_baseline"]) >= 0.05:
                warnings.append(f"pose alignment above threshold: {folder.name}")
            for k in targets:
                assert imread_rgb(folder / f"cam{k}.png").shape[2] == 3
                counts["gs_renders"] += 1
        for scale in (1.0, 0.5):
            for mode in ("uniform", "pose", "oracle"):
                folder = work / "flow" / setup / f"flow_s{scale}_{mode}"
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
