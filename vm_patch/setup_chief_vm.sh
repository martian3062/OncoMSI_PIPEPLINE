#!/usr/bin/env bash
set -euo pipefail

CHIEF_REPO_PATH="${CHIEF_REPO_PATH:-/home/pardeep/models/CHIEF}"
CHIEF_CTRANSPATH_WEIGHTS="${CHIEF_CTRANSPATH_WEIGHTS:-${CHIEF_REPO_PATH}/model_weight/CHIEF_CTransPath.pth}"

mkdir -p "$(dirname "${CHIEF_REPO_PATH}")"
if [ ! -d "${CHIEF_REPO_PATH}/.git" ]; then
  git clone https://github.com/hms-dbmi/CHIEF.git "${CHIEF_REPO_PATH}"
else
  git -C "${CHIEF_REPO_PATH}" pull --ff-only
fi

mkdir -p "$(dirname "${CHIEF_CTRANSPATH_WEIGHTS}")"

cat <<MSG
CHIEF source is ready at: ${CHIEF_REPO_PATH}

Strict CHIEF runs also require the official CTransPath checkpoint:
  ${CHIEF_CTRANSPATH_WEIGHTS}

The CHIEF GitHub README says the pretrained weights must be requested from:
  https://drive.google.com/drive/folders/1uRv9A1HuTW5m_pJoyMzdN31bE1i-tDaV?usp=sharing

After downloading, place CHIEF_CTransPath.pth at the path above. The runner is
intentionally strict and will fail CHIEF if this file is missing instead of
falling back to CTransPath/Phikon/any proxy encoder.
MSG

if [ ! -f "${CHIEF_CTRANSPATH_WEIGHTS}" ]; then
  exit 2
fi
