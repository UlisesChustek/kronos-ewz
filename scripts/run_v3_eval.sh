#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
CFG="$ROOT/.local-configs/config_ewz_h1_small_v3_7bar.yaml"
TRAIN_PID="${1:?usage: run_v3_eval.sh <training_pid>}"

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 30
done

echo "v3 training finished; starting calibration + eval"

"$PY" "$ROOT/scripts/tune_calibration.py" \
  --config "$CFG" --split val --pred-len 7 --stride 14 --samples 16 \
  --temperatures 0.9,1.0,1.2 --top-ps 0.85,0.9,0.95 \
  --device cuda:0 --out "$ROOT/results/calibration_v3_7.json"

T7=$("$PY" -c "import json; print(json.load(open('$ROOT/results/calibration_v3_7.json'))['temperature'])")
P7=$("$PY" -c "import json; print(json.load(open('$ROOT/results/calibration_v3_7.json'))['top_p'])")
echo "v3 7-bar best T=$T7 top_p=$P7"

"$PY" "$ROOT/scripts/evaluate.py" \
  --config "$CFG" --split test --pred-len 7 --stride 7 --samples 64 \
  --temperature "$T7" --top-p "$P7" --calibration "$ROOT/results/calibration_v3_7.json" \
  --device cuda:0 --out "$ROOT/results/eval_v3_test_7_cal.json"

"$PY" "$ROOT/scripts/evaluate.py" \
  --config "$CFG" --split test --pred-len 7 --stride 7 --samples 64 \
  --temperature "$T7" --top-p "$P7" \
  --device cuda:0 --out "$ROOT/results/eval_v3_test_7_raw.json"

echo "v3 evaluation complete"
