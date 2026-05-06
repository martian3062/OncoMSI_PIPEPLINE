#!/usr/bin/env bash
set -euo pipefail

TARGET_ROOT="${1:?usage: import_workflow_vm.sh <django_rebuild_root>}"
WORKFLOW_PATH="${2:-${TARGET_ROOT}/automation/n8n/msi_django_launch.json}"

export TMPDIR="${HOME}/tmp"
mkdir -p "${TMPDIR}" "${TARGET_ROOT}/runtime/n8n_data"

export NVM_DIR="${HOME}/.nvm"
. "${NVM_DIR}/nvm.sh"

export N8N_HOST=127.0.0.1
export N8N_PORT=5678
export N8N_PROTOCOL=http
export N8N_SECURE_COOKIE=false
export N8N_USER_FOLDER="${TARGET_ROOT}/runtime/n8n_data"
export N8N_ENCRYPTION_KEY=cleaned-msi-local-n8n-key

n8n import:workflow --input="${WORKFLOW_PATH}"
