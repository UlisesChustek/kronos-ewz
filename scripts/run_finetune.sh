#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${1:-$ROOT/.local-configs/config_ewz_h1_small.yaml}"
if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  echo "Run .venv/bin/python scripts/setup_local.py first, or pass a resolved config path." >&2
  exit 1
fi
CONFIG="$(realpath "$CONFIG")"

cd "$ROOT/third_party/Kronos/finetune_csv"
exec "$ROOT/.venv/bin/python" train_sequential.py --config "$CONFIG"
