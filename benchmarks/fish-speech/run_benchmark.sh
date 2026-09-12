#!/bin/bash
# Fish Speech S2 Pro Benchmark Runner
#
# Benchmarks already-running vllm-omni and/or sglang-omni servers.
# Produces JSON results and comparison plots.
#
# Usage:
#   # vllm-omni only (default):
#   bash run_benchmark.sh
#
#   # sglang-omni only:
#   bash run_benchmark.sh --sglang-only
#
#   # Compare both frameworks:
#   bash run_benchmark.sh --compare
#
# Environment variables:
#   NUM_PROMPTS   - Number of prompts per concurrency level (default: 50)
#   CONCURRENCY   - Space-separated concurrency levels (default: "1 4 10")
#   PORT          - vllm-omni server port (default: 8091)
#   SGLANG_PORT   - sglang-omni server port (default: 8000)
#   NUM_WARMUPS   - Warmup requests before each sweep (default: 3)
#   CHECK_TIMEOUT - Server probe timeout in seconds (default: 5)
#   REQUEST_TIMEOUT - Per-request benchmark timeout in seconds (default: 120)
#   SAMPLE_RATE   - Output PCM sample rate used for duration metrics (default: 44100)
#   CHANNELS      - Output PCM channel count used for duration metrics (default: 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Defaults
NUM_PROMPTS="${NUM_PROMPTS:-50}"
CONCURRENCY="${CONCURRENCY:-1 4 10}"
PORT="${PORT:-8091}"
SGLANG_PORT="${SGLANG_PORT:-8000}"
NUM_WARMUPS="${NUM_WARMUPS:-3}"
CHECK_TIMEOUT="${CHECK_TIMEOUT:-5}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-120}"
SAMPLE_RATE="${SAMPLE_RATE:-44100}"
CHANNELS="${CHANNELS:-1}"
RESULT_DIR="${SCRIPT_DIR}/results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# Parse args
RUN_VLLM=true
RUN_SGLANG=false
for arg in "$@"; do
    case "$arg" in
        --sglang-only) RUN_VLLM=false; RUN_SGLANG=true ;;
        --compare) RUN_SGLANG=true ;;
    esac
done

mkdir -p "${RESULT_DIR}"

echo "============================================================"
echo " Fish Speech S2 Pro Benchmark"
echo "============================================================"
echo " Prompts:      ${NUM_PROMPTS}"
echo " Concurrency:  ${CONCURRENCY}"
echo " Sample rate:  ${SAMPLE_RATE}"
echo " Channels:     ${CHANNELS}"
echo " Port (vllm):  ${PORT}"
if [ "${RUN_SGLANG}" = true ]; then
    echo " Port (sglang): ${SGLANG_PORT}"
fi
echo " Results:      ${RESULT_DIR}"
echo "============================================================"

probe_server() {
    local url="$1"

    curl --silent --show-error --fail \
        --connect-timeout "${CHECK_TIMEOUT}" \
        --max-time "${CHECK_TIMEOUT}" \
        "${url}" > /dev/null 2>&1
}

check_vllm_server() {
    local voices_url="http://127.0.0.1:${PORT}/v1/audio/voices"
    local health_url="http://127.0.0.1:${PORT}/health"

    if probe_server "${voices_url}"; then
        return 0
    fi

    if probe_server "${health_url}"; then
        echo "WARNING: vllm-omni speech probe failed at ${voices_url}"
        echo "         but /health is OK; continuing benchmark anyway."
        return 0
    fi

    echo "ERROR: vllm-omni is not reachable at ${voices_url}"
    echo "Start the server first, then rerun this benchmark."
    return 1
}

check_sglang_server() {
    local voices_url="http://127.0.0.1:${SGLANG_PORT}/v1/audio/voices"
    local health_url="http://127.0.0.1:${SGLANG_PORT}/health"

    if probe_server "${voices_url}"; then
        return 0
    fi

    if probe_server "${health_url}"; then
        echo "WARNING: sglang-omni speech probe failed at ${voices_url}"
        echo "         but /health is OK; continuing benchmark anyway."
        return 0
    fi

    echo "ERROR: sglang-omni is not reachable at ${voices_url}"
    echo "Start the server first, then rerun this benchmark."
    return 1
}

run_vllm_bench() {
    echo ""
    echo "============================================================"
    echo " Benchmarking vllm-omni"
    echo "============================================================"

    check_vllm_server

    local conc_args=""
    for c in ${CONCURRENCY}; do
        conc_args="${conc_args} ${c}"
    done

    cd "${PROJECT_ROOT}"
    # shellcheck disable=SC2086
    python "${SCRIPT_DIR}/vllm_omni/bench_fish_server.py" \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --num-prompts "${NUM_PROMPTS}" \
        --max-concurrency ${conc_args} \
        --num-warmups "${NUM_WARMUPS}" \
        --sample-rate "${SAMPLE_RATE}" \
        --channels "${CHANNELS}" \
        --request-timeout "${REQUEST_TIMEOUT}" \
        --config-name "vllm_omni" \
        --result-dir "${RESULT_DIR}" \
        --timestamp "${TIMESTAMP}"
}

run_sglang_bench() {
    echo ""
    echo "============================================================"
    echo " Benchmarking sglang-omni"
    echo "============================================================"

    check_sglang_server

    local conc_args=""
    for c in ${CONCURRENCY}; do
        conc_args="${conc_args} ${c}"
    done

    cd "${PROJECT_ROOT}"
    # shellcheck disable=SC2086
    python "${SCRIPT_DIR}/sglang_omni/bench_fish_server.py" \
        --host 127.0.0.1 \
        --port "${SGLANG_PORT}" \
        --num-prompts "${NUM_PROMPTS}" \
        --max-concurrency ${conc_args} \
        --num-warmups "${NUM_WARMUPS}" \
        --sample-rate "${SAMPLE_RATE}" \
        --channels "${CHANNELS}" \
        --request-timeout "${REQUEST_TIMEOUT}" \
        --config-name "sglang_omni" \
        --result-dir "${RESULT_DIR}" \
        --timestamp "${TIMESTAMP}"
}

if [ "${RUN_VLLM}" = true ]; then
    run_vllm_bench
fi

if [ "${RUN_SGLANG}" = true ]; then
    run_sglang_bench
fi

echo ""
echo "============================================================"
echo " Generating plots..."
echo "============================================================"

PLOT_SCRIPT="${PROJECT_ROOT}/benchmarks/fish-speech/plot_results.py"

RESULT_FILES=""
LABELS=""

if [ "${RUN_VLLM}" = true ]; then
    VLLM_FILE=$(ls -t "${RESULT_DIR}"/bench_vllm_omni_*.json 2>/dev/null | head -1)
    if [ -n "${VLLM_FILE}" ]; then
        RESULT_FILES="${VLLM_FILE}"
        LABELS="vllm-omni"
    fi
fi

if [ "${RUN_SGLANG}" = true ]; then
    SGLANG_FILE=$(ls -t "${RESULT_DIR}"/bench_sglang_omni_*.json 2>/dev/null | head -1)
    if [ -n "${SGLANG_FILE}" ]; then
        if [ -n "${RESULT_FILES}" ]; then
            RESULT_FILES="${RESULT_FILES} ${SGLANG_FILE}"
            LABELS="${LABELS} sglang-omni"
        else
            RESULT_FILES="${SGLANG_FILE}"
            LABELS="sglang-omni"
        fi
    fi
fi

if [ -n "${RESULT_FILES}" ]; then
    # shellcheck disable=SC2086
    python "${PLOT_SCRIPT}" \
        --results ${RESULT_FILES} \
        --labels ${LABELS} \
        --title "Fish Speech S2 Pro" \
        --output "${RESULT_DIR}/fish_speech_benchmark_${TIMESTAMP}.png"
fi

echo ""
echo "============================================================"
echo " Benchmark complete!"
echo " Results: ${RESULT_DIR}"
echo "============================================================"
