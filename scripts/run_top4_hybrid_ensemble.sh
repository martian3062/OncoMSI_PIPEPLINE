#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:?pass bundle root, e.g. /home/pardeep/.../automation/tcga_slide_triads/run-xxxx}"
PYTHON_BIN="${2:-/home/pardeep/.venvs/pathology310-hybrid/bin/python}"
OUT_DIR="$RUN_ROOT/ensemble/Top4-Hybrid-Ensemble"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "$OUT_DIR"

"$PYTHON_BIN" "$PROJECT_ROOT/tools/top4_late_fusion_ensemble.py" \
  --run-root "$RUN_ROOT" \
  --output-dir "$OUT_DIR" \
  --threshold 0.5

echo "Ensemble outputs written to $OUT_DIR"
