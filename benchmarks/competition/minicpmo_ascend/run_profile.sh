#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
MODEL="${MODEL:-/workspace/minicpmo-ascend/models/MiniCPM-o-4_5}"
BASE_CONFIG="${BASE_CONFIG:-${ROOT_DIR}/vllm_omni/deploy/minicpmo_4_5_ascend_910c_1card.yaml}"
PROFILE_ID="${PROFILE_ID:-$(date -u +%Y%m%dT%H%M%SZ)-stage2-text-audio}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/artifacts/minicpmo_ascend/profiles/${PROFILE_ID}}"
TRACE_DIR="${TRACE_DIR:-${OUTPUT_DIR}/traces}"
PROFILE_STAGES="${PROFILE_STAGES:-2}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8099}"
BASE_URL="http://${HOST}:${PORT}/v1"
SERVER_DIR="${OUTPUT_DIR}/server"
WORKLOAD_DIR="${OUTPUT_DIR}/workload"
PROFILE_CONFIG="${OUTPUT_DIR}/profile_deploy.yaml"

IFS=',' read -r -a stage_args <<<"${PROFILE_STAGES}"
config_args=()
capture_stage_args=()
for stage in "${stage_args[@]}"; do
    config_args+=("${stage}")
    capture_stage_args+=("${stage}")
done

if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -print -quit)" ]]; then
    echo "refusing to reuse non-empty profile output: ${OUTPUT_DIR}" >&2
    exit 1
fi
mkdir -p "${OUTPUT_DIR}" "${SERVER_DIR}" "${WORKLOAD_DIR}" "${TRACE_DIR}"
if curl --silent --show-error --fail "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    echo "refusing to start: a service is already healthy at http://${HOST}:${PORT}" >&2
    exit 1
fi
"${PYTHON_BIN}" -m benchmarks.competition.minicpmo_ascend.profile_config \
    --base-config "${BASE_CONFIG}" \
    --output "${PROFILE_CONFIG}" \
    --trace-dir "${TRACE_DIR}" \
    --stages "${config_args[@]}"

server_pid=""
cleanup() {
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        kill -TERM -- "-${server_pid}" 2>/dev/null || kill -TERM "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if command -v setsid >/dev/null 2>&1; then
    setsid env \
        MODEL="${MODEL}" \
        DEPLOY_CONFIG="${PROFILE_CONFIG}" \
        HOST="${HOST}" \
        PORT="${PORT}" \
        ARTIFACT_DIR="${SERVER_DIR}" \
        bash "${ROOT_DIR}/benchmarks/competition/minicpmo_ascend/start_server.sh" &
else
    env \
        MODEL="${MODEL}" \
        DEPLOY_CONFIG="${PROFILE_CONFIG}" \
        HOST="${HOST}" \
        PORT="${PORT}" \
        ARTIFACT_DIR="${SERVER_DIR}" \
        bash "${ROOT_DIR}/benchmarks/competition/minicpmo_ascend/start_server.sh" &
fi
server_pid=$!
printf '%s\n' "${server_pid}" >"${SERVER_DIR}/server.pid"

deadline=$((SECONDS + ${SERVER_START_TIMEOUT:-1200}))
until curl --silent --show-error --fail "http://${HOST}:${PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        echo "profile server exited before becoming healthy" >&2
        exit 1
    fi
    if ((SECONDS >= deadline)); then
        echo "profile server did not become healthy before timeout" >&2
        exit 1
    fi
    sleep 5
done

capture_args=(
    --base-url "${BASE_URL}"
    --model "${MODEL}"
    --output-dir "${WORKLOAD_DIR}"
    --stages "${capture_stage_args[@]}"
    --input-modality "${PROFILE_INPUT_MODALITY:-text}"
    --output-mode "${PROFILE_OUTPUT_MODE:-text_audio}"
    --warmups "${PROFILE_WARMUPS:-2}"
    --requests "${PROFILE_REQUESTS:-1}"
    --thinker-max-tokens "${THINKER_MAX_TOKENS:-128}"
    --talker-max-tokens "${TALKER_MAX_TOKENS:-128}"
    --timeout "${PROFILE_TIMEOUT:-900}"
)
if [[ -n "${PROFILE_MEDIA:-}" ]]; then
    capture_args+=(--media "${PROFILE_MEDIA}")
fi
if [[ -n "${PROFILE_PROMPT:-}" ]]; then
    capture_args+=(--prompt "${PROFILE_PROMPT}")
fi

"${PYTHON_BIN}" -m benchmarks.competition.minicpmo_ascend.profile "${capture_args[@]}"
cleanup
server_pid=""
"${PYTHON_BIN}" -m benchmarks.competition.minicpmo_ascend.profile_analysis analyze \
    "${TRACE_DIR}" \
    --capture "${WORKLOAD_DIR}/profile_capture.json" \
    --output "${OUTPUT_DIR}/profile_analysis.json"
"${PYTHON_BIN}" -m benchmarks.competition.minicpmo_ascend.profile_analysis manifest \
    "${OUTPUT_DIR}" \
    --output "${OUTPUT_DIR}/artifact_manifest.sha256"

echo "Profile artifacts: ${OUTPUT_DIR}"
