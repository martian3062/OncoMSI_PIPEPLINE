#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc"
RUN_ROOT="$ROOT/automation/tcga_slide_triads/run-65512be1f9c4"
BUNDLE="$RUN_ROOT/bundle_config.json"
PYTHON="/home/pardeep/.venvs/pathology310-hybrid/bin/python"
LOG="$ROOT/django_rebuild_cleaned_msi/runtime/launch_logs/run-65512be1f9c4-recovery.log"
exec >> "$LOG" 2>&1

echo "[$(date '+%F %T')] recovery supervisor started"
while ps -p 140072 >/dev/null 2>&1 || ps -p 140073 >/dev/null 2>&1; do
  sleep 30
done

echo "[$(date '+%F %T')] first pair finished, launching second pair"
mkdir -p "$RUN_ROOT/approaches/Approach5-H-Optimus-0" "$RUN_ROOT/approaches/Approach6-Midnight-12k"
nohup "$PYTHON" "$ROOT/scripts/run_tcga_coad_automated_triad.py" --bundle-config "$BUNDLE" --stage train-approach --approach-label Approach5-H-Optimus-0 >> "$RUN_ROOT/approaches/Approach5-H-Optimus-0/runner.log" 2>&1 &
PID5=$!
nohup "$PYTHON" "$ROOT/scripts/run_tcga_coad_automated_triad.py" --bundle-config "$BUNDLE" --stage train-approach --approach-label Approach6-Midnight-12k >> "$RUN_ROOT/approaches/Approach6-Midnight-12k/runner.log" 2>&1 &
PID6=$!
echo "[$(date '+%F %T')] launched Approach5-H-Optimus-0 pid=$PID5"
echo "[$(date '+%F %T')] launched Approach6-Midnight-12k pid=$PID6"
wait "$PID5"
wait "$PID6"

echo "[$(date '+%F %T')] second pair finished, finalizing existing bundle"
"$PYTHON" "$ROOT/scripts/run_tcga_coad_automated_triad.py" --bundle-config "$BUNDLE" --stage finalize-existing
STATUS=$?
echo "[$(date '+%F %T')] finalize-existing exit=$STATUS"
exit "$STATUS"
