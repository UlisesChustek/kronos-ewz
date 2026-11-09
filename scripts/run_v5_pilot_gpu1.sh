#!/usr/bin/env bash
# Bounded v5 pilot for EWZ hourly 7-bar forecasting.
#
# This script is intentionally conservative:
#   * it never overwrites an existing run or result,
#   * it trains on GPU 1 only (config device_id: 1 / --device cuda:1),
#   * the predictor run is capped by the v5 config (8 epochs, patience 3),
#   * calibration is fit on validation, evaluation is reported on test,
#   * it makes no claim that the model beats the walk-forward baselines.
#
# Default mode is a dry run.  Review the printed plan, then execute with:
#
#   scripts/run_v5_pilot_gpu1.sh --run
#
# If a checkpoint already exists and only evaluation is wanted:
#
#   scripts/run_v5_pilot_gpu1.sh --evaluate-only --run
#
# Add --overwrite-existing only if you have deliberately decided to replace the
# existing v5 run/results.  Historical checkpoints and results are never touched.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
CONFIG="$ROOT/.local-configs/config_ewz_h1_small_v5_7bar_frozen.yaml"
DEVICE="${DEVICE:-cuda:1}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  echo "Run .venv/bin/python scripts/setup_local.py first." >&2
  exit 1
fi

TRAIN_DIR="$ROOT/finetuned/ewz_h1_small_v5_7bar_frozen"
# Keep reruns separate from the shipped historical results.
OUT_DIR="${V5_OUT_DIR:-$ROOT/results/local_v5_pilot}"
CAL_OUT="$OUT_DIR/calibration_v5_7bar.json"
RAW_OUT="$OUT_DIR/eval_v5_test_7bar_raw.json"
CAL_EVAL_OUT="$OUT_DIR/eval_v5_test_7bar_cal.json"
RAW_STATS="$OUT_DIR/stats_v5_test_7bar_raw.json"
CAL_STATS="$OUT_DIR/stats_v5_test_7bar_cal.json"
LOG="$ROOT/logs/run_v5_pilot_gpu1.log"

RUN=0
OVERWRITE=0
EVAL_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --run) RUN=1 ;;
    --evaluate-only) EVAL_ONLY=1 ;;
    --overwrite-existing) OVERWRITE=1 ;;
    -h|--help)
      sed -n '2,24p' "$0"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

MODE="train + evaluate"
[[ "$EVAL_ONLY" -eq 1 ]] && MODE="evaluate existing checkpoint only"

echo "v5 pilot plan"
echo "  mode     : $MODE"
echo "  config   : $CONFIG"
echo "  device   : $DEVICE"
echo "  train dir: $TRAIN_DIR"
echo "  outputs  : $CAL_OUT"
echo "             $RAW_OUT (+ .windows.csv)"
echo "             $CAL_EVAL_OUT (+ .windows.csv)"
echo "             $RAW_STATS"
echo "             $CAL_STATS"
echo "  log      : $LOG"

# Verify the config is the bounded pilot we expect before doing anything.
"$PY" - "$CONFIG" <<'PY'
import sys, yaml
path = sys.argv[1]
cfg = yaml.safe_load(open(path))
d, t, dev = cfg["data"], cfg["training"], cfg["device"]
checks = {
    "predict_window == 7": int(d["predict_window"]) == 7,
    "lookback_window == 128": int(d["lookback_window"]) == 128,
    "loss_mask_predict_only": bool(t.get("loss_mask_predict_only", False)),
    "teacher_forcing": bool(t.get("teacher_forcing", False)),
    "freeze_tokenizer": bool(t.get("freeze_tokenizer", False)),
    "basemodel_epochs <= 8": int(t["basemodel_epochs"]) <= 8,
    "early_stopping_patience > 0": int(t.get("early_stopping_patience", 0)) > 0,
    "train_tokenizer is false": not bool(cfg["experiment"].get("train_tokenizer", True)),
    "device_id == 1": int(dev.get("device_id", -1)) == 1,
}
bad = [name for name, ok in checks.items() if not ok]
for name, ok in checks.items():
    print(f"  [{'ok' if ok else 'FAIL'}] {name}")
if bad:
    raise SystemExit("config does not match the bounded v5 pilot: " + ", ".join(bad))
PY

if [[ "$EVAL_ONLY" -eq 1 ]]; then
  if [[ ! -d "$TRAIN_DIR/basemodel/best_model" ]]; then
    echo "evaluate-only requested but checkpoint is missing: $TRAIN_DIR/basemodel/best_model" >&2
    exit 1
  fi
fi

if [[ "$OVERWRITE" -eq 0 ]]; then
  existing=()
  if [[ "$EVAL_ONLY" -eq 0 ]]; then
    [[ -e "$TRAIN_DIR" ]] && existing+=("$TRAIN_DIR")
  fi
  for path in "$CAL_OUT" "$RAW_OUT" "$CAL_EVAL_OUT" "$RAW_STATS" "$CAL_STATS"; do
    [[ -e "$path" ]] && existing+=("$path")
  done
  if [[ "${#existing[@]}" -gt 0 ]]; then
    echo "Refusing to overwrite existing v5 artifacts:" >&2
    printf '  %s\n' "${existing[@]}" >&2
    echo "Use --overwrite-existing only after an explicit decision." >&2
    exit 1
  fi
fi

if [[ "$RUN" -ne 1 ]]; then
  echo
  echo "Dry run. Re-run with --run to execute."
  exit 0
fi

mkdir -p "$ROOT/logs" "$OUT_DIR"
if [[ "$EVAL_ONLY" -eq 0 ]]; then
  echo "[$(date '+%F %T')] stage 1/4: train predictor (bounded, GPU1)" | tee "$LOG"
  (
    cd "$ROOT/third_party/Kronos/finetune_csv"
    if [[ "$OVERWRITE" -eq 1 ]]; then
      "$PY" train_sequential.py --config "$CONFIG" --overwrite-existing
    else
      "$PY" train_sequential.py --config "$CONFIG"
    fi
  ) 2>&1 | tee -a "$LOG"
else
  echo "[$(date '+%F %T')] evaluate-only mode: skipping training" | tee "$LOG"
fi

CHECKPOINT="$TRAIN_DIR/basemodel/best_model"
[[ -d "$CHECKPOINT" ]] || { echo "missing checkpoint: $CHECKPOINT" >&2; exit 1; }

echo "[$(date '+%F %T')] stage 2/4: validation-only calibration" | tee -a "$LOG"
"$PY" "$ROOT/scripts/tune_calibration.py" \
  --config "$CONFIG" --split val --pred-len 7 --stride 7 --samples 16 \
  --temperatures 0.9,1.0,1.2 --top-ps 0.85,0.9,0.95 \
  --device "$DEVICE" --out "$CAL_OUT" 2>&1 | tee -a "$LOG"

T7="$("$PY" -c "import json; print(json.load(open('$CAL_OUT'))['temperature'])")"
P7="$("$PY" -c "import json; print(json.load(open('$CAL_OUT'))['top_p'])")"
echo "[$(date '+%F %T')] calibration chose T=$T7 top_p=$P7" | tee -a "$LOG"

echo "[$(date '+%F %T')] stage 3/4: raw test evaluation" | tee -a "$LOG"
"$PY" "$ROOT/scripts/evaluate.py" \
  --config "$CONFIG" --split test --pred-len 7 --stride 7 --samples 64 \
  --temperature "$T7" --top-p "$P7" \
  --device "$DEVICE" --out "$RAW_OUT" 2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] stage 4/4: calibrated test evaluation + stats" | tee -a "$LOG"
"$PY" "$ROOT/scripts/evaluate.py" \
  --config "$CONFIG" --split test --pred-len 7 --stride 7 --samples 64 \
  --temperature "$T7" --top-p "$P7" --calibration "$CAL_OUT" \
  --device "$DEVICE" --out "$CAL_EVAL_OUT" 2>&1 | tee -a "$LOG"

"$PY" "$ROOT/scripts/stats.py" \
  --windows "${RAW_OUT%.json}.windows.csv" --n-boot 5000 --out "$RAW_STATS" \
  2>&1 | tee -a "$LOG"
"$PY" "$ROOT/scripts/stats.py" \
  --windows "${CAL_EVAL_OUT%.json}.windows.csv" --n-boot 5000 --out "$CAL_STATS" \
  2>&1 | tee -a "$LOG"

echo "[$(date '+%F %T')] pilot complete" | tee -a "$LOG"
echo "No win claim. Compare these results with:" | tee -a "$LOG"
echo "  results/walk_forward_baselines_v5.json" | tee -a "$LOG"
echo "and read the retrospective/leakage caveats in PROJECT_STATUS.md." | tee -a "$LOG"
