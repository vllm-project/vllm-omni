#!/bin/bash
# Fish Speech S2 Pro Benchmark Runner
#
# Benchmarks vllm-omni serving, optionally comparing against sglang-omni.
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
#   # Custom settings:
#   GPU_DEVICE=1 NUM_PROMPTS=20 CONCURRENCY="1 4" bash run_benchmark.sh
#
#   # Custom stage config:
#   STAGE_CONFIG=/path/to/custom.yaml bash run_benchmark.sh
#
# Environment variables:
#   GPU_DEVICE       - GPU index to use (default: 0)
#   NUM_PROMPTS      - Number of prompts per concurrency level (default: 50)
#   CONCURRENCY      - Space-separated concurrency levels (default: "1 4 10")
#   MODEL            - Model name (default: fishaudio/s2-pro)
#   PORT             - vllm-omni server port (default: 8091)
#   SGLANG_PORT      - sglang-omni server port (default: 8000)
#   GPU_MEM_AR       - gpu_memory_utilization for Slow AR stage (default: 0.6)
#   GPU_MEM_DAC      - gpu_memory_utilization for DAC decoder stage (default: 0.1)
#   STAGE_CONFIG     - Path to stage config YAML (default: upstream fish_speech_s2_pro.yaml)
#   SGLANG_OMNI_DIR  - Path to sglang-omni checkout (default: ~/Dev/sglang-omni)
#   SGLANG_CONFIG    - sglang-omni config path (default: examples/configs/s2pro_tts.yaml)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Defaults
GPU_DEVICE="${GPU_DEVICE:-0}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"
CONCURRENCY="${CONCURRENCY:-1 4 10}"
MODEL="${MODEL:-fishaudio/s2-pro}"
PORT="${PORT:-8091}"
SGLANG_PORT="${SGLANG_PORT:-8000}"
GPU_MEM_AR="${GPU_MEM_AR:-0.6}"
GPU_MEM_DAC="${GPU_MEM_DAC:-0.1}"
NUM_WARMUPS="${NUM_WARMUPS:-3}"
# Default: use the upstream stage config from the model executor
DEFAULT_STAGE_CONFIG="${PROJECT_ROOT}/vllm_omni/model_executor/stage_configs/fish_speech_s2_pro.yaml"
STAGE_CONFIG="${STAGE_CONFIG:-${DEFAULT_STAGE_CONFIG}}"
SGLANG_OMNI_DIR="${SGLANG_OMNI_DIR:-${HOME}/Dev/sglang-omni}"
SGLANG_CONFIG="${SGLANG_CONFIG:-examples/configs/s2pro_tts.yaml}"
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
echo " GPU:          ${GPU_DEVICE}"
echo " Model:        ${MODEL}"
echo " Prompts:      ${NUM_PROMPTS}"
echo " Concurrency:  ${CONCURRENCY}"
echo " Port (vllm):  ${PORT}"
if [ "${RUN_SGLANG}" = true ]; then
echo " Port (sglang): ${SGLANG_PORT}"
fi
echo " Stage config: ${STAGE_CONFIG}"
echo " Results:      ${RESULT_DIR}"
echo "============================================================"

# -------------------------------------------------------------------
# Server lifecycle helpers
# -------------------------------------------------------------------

prepare_config() {
    local config_template="$1"
    local config_name="$2"
    local output_path="${RESULT_DIR}/${config_name}_stage_config.yaml"

    sed \
        -e "s/devices: \"0\"/devices: \"${GPU_DEVICE}\"/g" \
        -e "s/gpu_memory_utilization: 0.6/gpu_memory_utilization: ${GPU_MEM_AR}/g" \
        -e "s/gpu_memory_utilization: 0.1/gpu_memory_utilization: ${GPU_MEM_DAC}/g" \
        "${config_template}" > "${output_path}"

    echo "${output_path}"
}

start_vllm_server() {
    local stage_config="$1"
    local config_name="$2"
    local log_file="${RESULT_DIR}/server_${config_name}_${TIMESTAMP}.log"

    echo ""
    echo "Starting vllm-omni server with config: ${config_name}"
    echo "  Stage config: ${stage_config}"
    echo "  Log file: ${log_file}"

    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" \
    python -m vllm_omni.entrypoints.cli.main serve "${MODEL}" \
        --omni \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --stage-configs-path "${stage_config}" \
        --stage-init-timeout 120 \
        --trust-remote-code \
        --enforce-eager \
        --disable-log-stats \
        > "${log_file}" 2>&1 &

    SERVER_PID=$!
    echo "  Server PID: ${SERVER_PID}"

    echo "  Waiting for server to be ready..."
    local max_wait=300
    local waited=0
    while [ ${waited} -lt ${max_wait} ]; do
        if curl -sf "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
            echo "  Server is ready! (waited ${waited}s)"
            return 0
        fi
        if ! kill -0 ${SERVER_PID} 2>/dev/null; then
            echo "  ERROR: Server process died. Check log: ${log_file}"
            tail -20 "${log_file}"
            return 1
        fi
        sleep 2
        waited=$((waited + 2))
    done

    echo "  ERROR: Server did not start within ${max_wait}s. Check log: ${log_file}"
    kill ${SERVER_PID} 2>/dev/null || true
    return 1
}

start_sglang_server() {
    local log_file="${RESULT_DIR}/server_sglang_omni_${TIMESTAMP}.log"

    if [ ! -d "${SGLANG_OMNI_DIR}" ]; then
        echo "  ERROR: sglang-omni directory not found at ${SGLANG_OMNI_DIR}"
        echo "  Set SGLANG_OMNI_DIR or clone: git clone https://github.com/sgl-project/sglang-omni.git"
        return 1
    fi

    echo ""
    echo "Starting sglang-omni server"
    echo "  Directory: ${SGLANG_OMNI_DIR}"
    echo "  Config: ${SGLANG_CONFIG}"
    echo "  Log file: ${log_file}"

    (
        cd "${SGLANG_OMNI_DIR}"
        if [ -f ".venv/bin/activate" ]; then
            # shellcheck disable=SC1091
            source .venv/bin/activate
        fi
        CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" \
        sgl-omni serve \
            --model-path "${MODEL}" \
            --config "${SGLANG_CONFIG}" \
            --port "${SGLANG_PORT}" \
            > "${log_file}" 2>&1
    ) &

    SGLANG_PID=$!
    echo "  Server PID: ${SGLANG_PID}"

    echo "  Waiting for sglang-omni server to be ready..."
    local max_wait=300
    local waited=0
    while [ ${waited} -lt ${max_wait} ]; do
        if curl -sf "http://127.0.0.1:${SGLANG_PORT}/health" > /dev/null 2>&1; then
            echo "  Server is ready! (waited ${waited}s)"
            return 0
        fi
        if ! kill -0 ${SGLANG_PID} 2>/dev/null; then
            echo "  ERROR: sglang-omni server process died. Check log: ${log_file}"
            tail -20 "${log_file}"
            return 1
        fi
        sleep 2
        waited=$((waited + 2))
    done

    echo "  ERROR: sglang-omni server did not start within ${max_wait}s."
    kill ${SGLANG_PID} 2>/dev/null || true
    return 1
}

stop_server() {
    local pid_var="$1"
    local port="$2"
    local pid="${!pid_var:-}"

    if [ -n "${pid}" ]; then
        echo "  Stopping server (PID: ${pid})..."
        kill "${pid}" 2>/dev/null || true
        wait "${pid}" 2>/dev/null || true
        local pids
        pids=$(lsof -ti:"${port}" 2>/dev/null || true)
        if [ -n "${pids}" ]; then
            echo "  Cleaning up remaining processes on port ${port}..."
            echo "${pids}" | xargs kill -9 2>/dev/null || true
        fi
        echo "  Server stopped."
        eval "${pid_var}="
    fi
}

# Cleanup on exit
cleanup() {
    stop_server SERVER_PID "${PORT}"
    stop_server SGLANG_PID "${SGLANG_PORT}"
}
trap cleanup EXIT

# -------------------------------------------------------------------
# Benchmark execution
# -------------------------------------------------------------------

run_vllm_bench() {
    local config_name="$1"
    local config_template="$2"

    echo ""
    echo "============================================================"
    echo " Benchmarking vllm-omni: ${config_name}"
    echo "============================================================"

    local stage_config
    stage_config=$(prepare_config "${config_template}" "${config_name}")

    start_vllm_server "${stage_config}" "${config_name}"

    # Convert concurrency string to args
    local conc_args=""
    for c in ${CONCURRENCY}; do
        conc_args="${conc_args} ${c}"
    done

    cd "${PROJECT_ROOT}"
    # shellcheck disable=SC2086
    python "${SCRIPT_DIR}/vllm_omni/bench_tts_serve.py" \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --num-prompts "${NUM_PROMPTS}" \
        --max-concurrency ${conc_args} \
        --num-warmups "${NUM_WARMUPS}" \
        --config-name "${config_name}" \
        --result-dir "${RESULT_DIR}"

    stop_server SERVER_PID "${PORT}"
    sleep 5
}

run_sglang_bench() {
    echo ""
    echo "============================================================"
    echo " Benchmarking sglang-omni"
    echo "============================================================"

    start_sglang_server

    local conc_args=""
    for c in ${CONCURRENCY}; do
        conc_args="${conc_args} ${c}"
    done

    cd "${PROJECT_ROOT}"
    # shellcheck disable=SC2086
    python "${SCRIPT_DIR}/sglang_omni/bench_tts_serve.py" \
        --host 127.0.0.1 \
        --port "${SGLANG_PORT}" \
        --num-prompts "${NUM_PROMPTS}" \
        --max-concurrency ${conc_args} \
        --num-warmups "${NUM_WARMUPS}" \
        --config-name "sglang_omni" \
        --result-dir "${RESULT_DIR}"

    stop_server SGLANG_PID "${SGLANG_PORT}"
    sleep 5
}

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if [ "${RUN_VLLM}" = true ]; then
    run_vllm_bench "vllm_omni" "${STAGE_CONFIG}"
fi

if [ "${RUN_SGLANG}" = true ]; then
    run_sglang_bench
fi

# -------------------------------------------------------------------
# Plot results
# -------------------------------------------------------------------

echo ""
echo "============================================================"
echo " Generating plots..."
echo "============================================================"

PLOT_SCRIPT="${PROJECT_ROOT}/benchmarks/qwen3-tts/plot_results.py"

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
