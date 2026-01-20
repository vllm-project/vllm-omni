#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
UV_BIN="uv"
PYTHON_BOOTSTRAP="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-simple-test}"
TARGET_PY_VERSION="3.12"

if ! command -v "${PYTHON_BOOTSTRAP}" >/dev/null 2>&1; then
  PYTHON_BOOTSTRAP="python"
fi

cd "${REPO_ROOT}"

# Ensure pip-installed scripts (including uv) are on PATH
PYTHON_SCRIPTS="$(${PYTHON_BOOTSTRAP} -c 'import sysconfig; print(sysconfig.get_path("scripts"))')"
USER_BASE="$(${PYTHON_BOOTSTRAP} -m site --user-base 2>/dev/null || true)"
PATH="${PYTHON_SCRIPTS}:${PATH}"
if [[ -n "${USER_BASE}" ]]; then
  PATH="${USER_BASE}/bin:${PATH}"
fi

if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
  "${PYTHON_BOOTSTRAP}" -m pip install --upgrade pip
  "${PYTHON_BOOTSTRAP}" -m pip install uv
  hash -r
fi

TARGET_PY_PATH="$(${UV_BIN} python find "${TARGET_PY_VERSION}" 2>/dev/null | head -n1 || true)"
if [[ -z "${TARGET_PY_PATH}" ]]; then
  "${UV_BIN}" python install "${TARGET_PY_VERSION}"
  TARGET_PY_PATH="$(${UV_BIN} python find "${TARGET_PY_VERSION}" 2>/dev/null | head -n1 || true)"
fi

if [[ -z "${TARGET_PY_PATH}" ]]; then
  echo "Failed to locate python ${TARGET_PY_VERSION}" >&2
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  "${UV_BIN}" venv --python "${TARGET_PY_PATH}" "${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
[[ -x "${VENV_PYTHON}" ]] || { echo "Python not found in ${VENV_DIR}"; exit 1; }

"${UV_BIN}" pip install --python "${VENV_PYTHON}" vllm==0.12.0
"${UV_BIN}" pip install --python "${VENV_PYTHON}" -e ".[dev]"
"${VENV_PYTHON}" -m pytest -v -s tests/entrypoints/
"${VENV_PYTHON}" -m pytest -v -s tests/diffusion/cache/
"${VENV_PYTHON}" -m pytest -v -s tests/model_executor/models/qwen2_5_omni/test_audio_length.py
"${VENV_PYTHON}" -m pytest -v -s tests/worker/
