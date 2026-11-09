#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${1:-$ROOT/logs/train_h1_small.log}"

PID="$(pgrep -f 'train_sequential.py' | head -1 || true)"
if [[ -n "${PID}" ]]; then
  echo "STATUS: running (pid ${PID}, elapsed $(ps -p "${PID}" -o etime= | tr -d ' '))"
else
  echo "STATUS: not running"
fi

if grep -q "Training completed successfully" "${LOG}" 2>/dev/null; then
  echo "STAGE:  done"
elif grep -q "Basemodel training completed" "${LOG}" 2>/dev/null; then
  echo "STAGE:  basemodel done, finishing"
elif grep -q "Starting fine-tuning training" "${LOG}" 2>/dev/null; then
  echo "STAGE:  predictor (basemodel)"
else
  echo "STAGE:  tokenizer"
fi

echo "LAST:   $(grep -E '\[Epoch' "${LOG}" | tail -1 || echo 'no epoch yet')"
echo "CHECKPOINTS:"
find "${ROOT}/finetuned" -maxdepth 6 -type d -name best_model 2>/dev/null | sed "s|${ROOT}/||" | sort || true
echo "GPU:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
