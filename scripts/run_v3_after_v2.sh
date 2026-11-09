#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_PID="${1:?usage: run_v3_after_v2.sh <v2_training_pid>}"

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 30
done

echo "v2 training finished; starting v3 (7-bar, base-only, v2 tokenizer)"
exec "$ROOT/scripts/run_finetune.sh" "$ROOT/.local-configs/config_ewz_h1_small_v3_7bar.yaml"
