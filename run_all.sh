#!/usr/bin/env bash
# Full local-GS quality experiment. Usage: bash run_all.sh [config]   (run inside the AnySplat conda env)
set -euo pipefail
cd "$(dirname "$0")"
CFG=${1:-configs/mofang_9cam.yaml}
LOG_DIR=${LGS_WORK_DIR:-work}/logs
mkdir -p "$LOG_DIR"

python scripts/prepare.py --config "$CFG" 2>&1 | tee "$LOG_DIR/0_prepare.log"
python scripts/calibrate_rig.py --config "$CFG" 2>&1 | tee "$LOG_DIR/0b_calibrate.log"
python scripts/run_anysplat.py --config "$CFG" 2>&1 | tee "$LOG_DIR/1_anysplat.log"
python scripts/run_flow.py --config "$CFG" --scales 1.0 0.5 --tmodes uniform pose posecal oracle 2>&1 | tee "$LOG_DIR/2_flow.log"
python scripts/evaluate.py --config "$CFG" 2>&1 | tee "$LOG_DIR/3_evaluate.log"
python scripts/validate_results.py "$CFG" 2>&1 | tee "$LOG_DIR/4_validation.log"
echo "结果: ${LGS_WORK_DIR:-work}/eval/summary.md"
