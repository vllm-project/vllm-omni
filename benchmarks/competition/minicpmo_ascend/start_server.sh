#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL="${MODEL:-openbmb/MiniCPM-o-4_5}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-${ROOT_DIR}/vllm_omni/deploy/minicpmo_4_5_ascend_910c_1card.yaml}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8099}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT_DIR}/artifacts/minicpmo_ascend/server}"
SERVER_BIN="${SERVER_BIN:-${ROOT_DIR}/.venv/bin/vllm-omni}"

mkdir -p "${ARTIFACT_DIR}"
if [[ ! -x "${SERVER_BIN}" ]]; then
    SERVER_BIN="${SERVER_BIN_FALLBACK:-vllm-omni}"
fi

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
if [[ -x "${PYTHON_BIN}" ]]; then
    VLLM_ASCEND_LIB_DIR="$("${PYTHON_BIN}" - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("vllm_ascend")
if spec is not None and spec.origin is not None:
    candidate = Path(spec.origin).resolve().parent / "lib"
    if (candidate / "libvllm_ascend_kernels.so").is_file():
        print(candidate)
PY
)"
    if [[ -n "${VLLM_ASCEND_LIB_DIR}" ]]; then
        export LD_LIBRARY_PATH="${VLLM_ASCEND_LIB_DIR}:${LD_LIBRARY_PATH:-}"
    fi
fi

command=(
    "${SERVER_BIN}" serve "${MODEL}"
    --omni
    --deploy-config "${DEPLOY_CONFIG}"
    --trust-remote-code
    --host "${HOST}"
    --port "${PORT}"
)
if [[ -n "${MODEL_REVISION:-}" ]]; then
    command+=(--revision "${MODEL_REVISION}")
fi
if [[ -n "${ALLOWED_LOCAL_MEDIA_PATH:-}" ]]; then
    command+=(--allowed-local-media-path "${ALLOWED_LOCAL_MEDIA_PATH}")
fi
if [[ -n "${SERVER_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_args=(${SERVER_EXTRA_ARGS})
    command+=("${extra_args[@]}")
fi

printf '%q ' "${command[@]}" | tee "${ARTIFACT_DIR}/server_command.txt"
printf '\n' | tee -a "${ARTIFACT_DIR}/server_command.txt"
printf 'LD_LIBRARY_PATH=%q\n' "${LD_LIBRARY_PATH:-}" | tee "${ARTIFACT_DIR}/server_environment.txt"
exec "${command[@]}" > >(tee "${ARTIFACT_DIR}/server.log") 2>&1
