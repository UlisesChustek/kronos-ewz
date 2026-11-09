#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
CFG="$ROOT/.local-configs/config_ewz_h1_small_v2.yaml"
TRAIN_PID="${1:?usage: run_v2_eval.sh <training_pid>}"

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 30
done

echo "v2 training finished; starting calibration + eval"

"$PY" "$ROOT/scripts/tune_calibration.py" \
  --config "$CFG" --split val --pred-len 24 --stride 24 --samples 16 \
  --temperatures 0.9,1.0,1.2 --top-ps 0.85,0.9,0.95 \
  --device cuda:0 --out "$ROOT/results/calibration_v2_24.json"

T24=$("$PY" -c "import json; print(json.load(open('$ROOT/results/calibration_v2_24.json'))['temperature'])")
P24=$("$PY" -c "import json; print(json.load(open('$ROOT/results/calibration_v2_24.json'))['top_p'])")
echo "v2 24-bar best T=$T24 top_p=$P24"

"$PY" "$ROOT/scripts/evaluate.py" \
  --config "$CFG" --split test --pred-len 24 --stride 24 --samples 64 \
  --temperature "$T24" --top-p "$P24" --calibration "$ROOT/results/calibration_v2_24.json" \
  --device cuda:0 --out "$ROOT/results/eval_v2_test_24_cal.json"

"$PY" "$ROOT/scripts/evaluate.py" \
  --config "$CFG" --split test --pred-len 24 --stride 24 --samples 64 \
  --temperature "$T24" --top-p "$P24" \
  --device cuda:0 --out "$ROOT/results/eval_v2_test_24_raw.json"

"$PY" "$ROOT/scripts/tune_calibration.py" \
  --config "$CFG" --split val --pred-len 7 --stride 14 --samples 16 \
  --temperatures 0.9,1.0,1.2 --top-ps 0.85,0.9,0.95 \
  --device cuda:0 --out "$ROOT/results/calibration_v2_7.json"

T7=$("$PY" -c "import json; print(json.load(open('$ROOT/results/calibration_v2_7.json'))['temperature'])")
P7=$("$PY" -c "import json; print(json.load(open('$ROOT/results/calibration_v2_7.json'))['top_p'])")
echo "v2 7-bar best T=$T7 top_p=$P7"

"$PY" "$ROOT/scripts/evaluate.py" \
  --config "$CFG" --split test --pred-len 7 --stride 7 --samples 64 \
  --temperature "$T7" --top-p "$P7" --calibration "$ROOT/results/calibration_v2_7.json" \
  --device cuda:0 --out "$ROOT/results/eval_v2_test_7_cal.json"

"$PY" "$ROOT/scripts/evaluate.py" \
  --config "$CFG" --split test --pred-len 7 --stride 7 --samples 64 \
  --temperature "$T7" --top-p "$P7" \
  --device cuda:0 --out "$ROOT/results/eval_v2_test_7_raw.json"

echo "v2 evaluation complete"
