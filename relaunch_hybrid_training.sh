#!/usr/bin/env bash
set -euo pipefail

CFG=/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/django_rebuild_cleaned_msi/runtime/bundle_configs/run-7808c90045e9.json
ROOT=/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/automation/tcga_slide_triads/run-7808c90045e9
PY=/opt/miniforge3/envs/pathology310/bin/python
SCRIPT=/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/scripts/run_tcga_coad_automated_triad.py

labels=(
  Approach1-Virchow
  Approach2-RetCCL
  Approach3-CTransPath
  Approach4-CONCH
  Approach5-Virchow2
  Approach6-UNI2-H
  Approach7-H-Optimus-0
)

for label in "${labels[@]}"; do
  mkdir -p "$ROOT/approaches/$label"
  nohup env PYTHONNOUSERSITE=1 PYTHONPATH= VIRTUAL_ENV= "$PY" "$SCRIPT" --bundle-config "$CFG" --stage train-approach --approach-label "$label" > "$ROOT/approaches/$label/runner.log" 2>&1 < /dev/null &
  echo "$label $!"
done
