#!/usr/bin/env bash
# Evaluate the v4 (masked predict-only loss) checkpoint on GPU1.
# Inference only: no training, loads the already-saved best_model.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
CFG="$ROOT/.local-configs/config_ewz_h1_small_v4_24_masked.yaml"
DEV="${DEVICE:-cuda:1}"

echo "[$(date '+%F %T')] v4 eval on $DEV"
echo "[$(date '+%F %T')] checkpoint: $ROOT/finetuned/ewz_h1_small_v4_24_masked/basemodel/best_model"

echo "[$(date '+%F %T')] stage 1/3: calibration on val"
"$PY" "$ROOT/scripts/tune_calibration.py" \
  --config "$CFG" --split val --pred-len 24 --stride 24 --samples 16 \
  --temperatures 0.9,1.0,1.2 --top-ps 0.85,0.9,0.95 \
  --device "$DEV" --out "$ROOT/results/calibration_v4_24.json"

T24=$("$PY" -c "import json; print(json.load(open('$ROOT/results/calibration_v4_24.json'))['temperature'])")
P24=$("$PY" -c "import json; print(json.load(open('$ROOT/results/calibration_v4_24.json'))['top_p'])")
echo "[$(date '+%F %T')] v4 24-bar best T=$T24 top_p=$P24"

echo "[$(date '+%F %T')] stage 2/3: calibrated test eval"
"$PY" "$ROOT/scripts/evaluate.py" \
  --config "$CFG" --split test --pred-len 24 --stride 24 --samples 64 \
  --temperature "$T24" --top-p "$P24" --calibration "$ROOT/results/calibration_v4_24.json" \
  --device "$DEV" --out "$ROOT/results/eval_v4_test_24_cal.json"

echo "[$(date '+%F %T')] stage 3/3: raw test eval"
"$PY" "$ROOT/scripts/evaluate.py" \
  --config "$CFG" --split test --pred-len 24 --stride 24 --samples 64 \
  --temperature "$T24" --top-p "$P24" \
  --device "$DEV" --out "$ROOT/results/eval_v4_test_24_raw.json"

echo "[$(date '+%F %T')] v4 evaluation complete"
