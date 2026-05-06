#!/usr/bin/env bash
set -euo pipefail

TARGET_ROOT="${1:?usage: start_n8n_vm.sh <django_rebuild_root>}"
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

if [[ -f "${TARGET_ROOT}/runtime/n8n.pid" ]] && kill -0 "$(cat "${TARGET_ROOT}/runtime/n8n.pid")" 2>/dev/null; then
  kill "$(cat "${TARGET_ROOT}/runtime/n8n.pid")" || true
  sleep 2
fi

nohup bash -lc "export TMPDIR='${TMPDIR}'; export NVM_DIR='${NVM_DIR}'; . '${NVM_DIR}/nvm.sh'; export N8N_HOST='${N8N_HOST}'; export N8N_PORT='${N8N_PORT}'; export N8N_PROTOCOL='${N8N_PROTOCOL}'; export N8N_SECURE_COOKIE='${N8N_SECURE_COOKIE}'; export N8N_USER_FOLDER='${N8N_USER_FOLDER}'; export N8N_ENCRYPTION_KEY='${N8N_ENCRYPTION_KEY}'; n8n start" > "${TARGET_ROOT}/runtime/n8n.log" 2>&1 < /dev/null &
echo $! > "${TARGET_ROOT}/runtime/n8n.pid"
sleep 10
curl -fsS "http://${N8N_HOST}:${N8N_PORT}/healthz" || curl -I "http://${N8N_HOST}:${N8N_PORT}"
