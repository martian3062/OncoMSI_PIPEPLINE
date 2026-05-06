#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/django_rebuild_cleaned_msi"
BACKUP="/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/backups"
ARCHIVE="/home/pardeep/pathology310_projects/single_slide_morphology/project_1_slideflow_msi_tcga_crc/hybrid-02-deploy.tar.gz"

mkdir -p "$BACKUP"
TS="$(date +%Y%m%d_%H%M%S)"

cd "$ROOT"
backup_items=(apps msi_platform manage.py README.md vm_patch)
existing_items=()
for item in "${backup_items[@]}"; do
  if [ -e "$item" ]; then
    existing_items+=("$item")
  fi
done
if [ "${#existing_items[@]}" -gt 0 ]; then
  tar -czf "$BACKUP/django_rebuild_cleaned_msi_$TS.tgz" "${existing_items[@]}"
fi

tar -xzf "$ARCHIVE" -C "$ROOT"

./.venv/bin/python manage.py migrate

pkill -f "manage.py runserver 0.0.0.0:8000 --noreload" || true
nohup ./.venv/bin/python manage.py runserver 0.0.0.0:8000 --noreload >/tmp/cleaned_msi_runserver.log 2>&1 &
sleep 5
pgrep -af "manage.py runserver 0.0.0.0:8000 --noreload"
