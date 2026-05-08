#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc"
DJANGO_ROOT="$PROJECT_ROOT/django_rebuild_cleaned_msi"
RUNTIME_ROOT="$DJANGO_ROOT/runtime"
SCRIPTS_ROOT="$PROJECT_ROOT/scripts"
TOOLS_ROOT="$PROJECT_ROOT/tools"
MODELS_ROOT="$PROJECT_ROOT/models/virchow"
AUTOMATION_ROOT="$PROJECT_ROOT/automation/tcga_slide_triads"
VENV_ROOT="/home/pardeep/.venvs/pathology310-hybrid"

mkdir -p \
  "$DJANGO_ROOT" \
  "$RUNTIME_ROOT/annotations" \
  "$RUNTIME_ROOT/bundle_configs" \
  "$RUNTIME_ROOT/launch_logs" \
  "$RUNTIME_ROOT/run_status" \
  "$RUNTIME_ROOT/tmp" \
  "$SCRIPTS_ROOT" \
  "$TOOLS_ROOT" \
  "$MODELS_ROOT" \
  "$AUTOMATION_ROOT" \
  "$(dirname "$VENV_ROOT")"

if [ ! -d "$VENV_ROOT" ]; then
  python3 -m venv "$VENV_ROOT"
fi

export SF_BACKEND="torch"
export SF_SLIDE_BACKEND="libvips"

"$VENV_ROOT/bin/python" -m pip install --upgrade pip
"$VENV_ROOT/bin/python" -m pip install Django==5.2.13 djangorestframework==3.17.1 plotly==6.7.0 numpy pandas scikit-learn fastai

echo "Bootstrap scaffold created."
echo "Project root: $PROJECT_ROOT"
echo "Runner python: $VENV_ROOT/bin/python"
