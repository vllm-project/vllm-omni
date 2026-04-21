#!/bin/bash
# Qwen3-TTS Benchmark Runner
#
# Compares vllm-omni streaming serving vs HuggingFace transformers offline inference.
# Produces JSON results and comparison plots.
#
# Usage:
#   # Full comparison (vllm-omni + HF):
#   bash run_benchmark.sh
#
#   # Only vllm-omni async_chunk config:
#   bash run_benchmark.sh --async-only
#
#   # Only HuggingFace baseline:
#   bash run_benchmark.sh --hf-only
#
#   # vllm-omni only (skip HF):
#   bash run_benchmark.sh --skip-hf
#
#   # Custom settings:
#   GPU_DEVICE=1 NUM_PROMPTS=20 CONCURRENCY="1 4" bash run_benchmark.sh
#
#   # Use 1.7B model:
#   MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice bash run_benchmark.sh --async-only
#
#   # Use Voice Clone model
#   MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base TASK_TYPE=Base bash run_benchmark.sh --async-only
#
#   # Use batch_size=4:
#   BATCH_SIZE=4 bash run_benchmark.sh --async-only
#
# Environment variables:
#   GPU_DEVICE       - GPU index to use (default: 0)
#   NUM_PROMPTS      - Number of prompts per concurrency level (default: 50)
#   CONCURRENCY      - Space-separated concurrency levels (default: "1 4 10")
#   MODEL            - Model name (default: Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
#   PORT             - Server port (default: 8000)
#   BATCH_SIZE       - Per-stage ``max_num_seqs`` for both talker and code2wav (default: 1)
#   GPU_MEM_TALKER   - gpu_memory_utilization for talker stage (default: 0.3 at bs=1, else 0.2)
#   GPU_MEM_CODE2WAV - gpu_memory_utilization for code2wav stage (default: 0.3 at bs=1, else 0.2)
#   TASK_TYPE        - Task type: CustomVoice, VoiceDesign, Base (default: CustomVoice)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Defaults
GPU_DEVICE="${GPU_DEVICE:-0}"
NUM_PROMPTS="${NUM_PROMPTS:-50}"
CONCURRENCY="${CONCURRENCY:-1 4 10}"
MODEL="${MODEL:-Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice}"
PORT="${PORT:-8000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DEFAULT_MEM=$([ "${BATCH_SIZE}" = "1" ] && echo "0.3" || echo "0.2")
GPU_MEM_TALKER="${GPU_MEM_TALKER:-${DEFAULT_MEM}}"
GPU_MEM_CODE2WAV="${GPU_MEM_CODE2WAV:-${DEFAULT_MEM}}"
NUM_WARMUPS="${NUM_WARMUPS:-3}"
DEPLOY_CONFIG="vllm_omni/deploy/qwen3_tts.yaml"
RESULT_DIR="${SCRIPT_DIR}/results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TASK_TYPE="${TASK_TYPE:-CustomVoice}"

# Build --stage-overrides JSON from BATCH_SIZE + GPU_MEM_*.
STAGE_OVERRIDES=$(
    BATCH_SIZE="${BATCH_SIZE}" \
    GPU_MEM_TALKER="${GPU_MEM_TALKER}" \
    GPU_MEM_CODE2WAV="${GPU_MEM_CODE2WAV}" \
    python - <<'PYEOF'
import json, os
bs = int(os.environ["BATCH_SIZE"])
mem_t = float(os.environ["GPU_MEM_TALKER"])
mem_c = float(os.environ["GPU_MEM_CODE2WAV"])
# Prefill budget grows with batch size on both stages.
talker_batched = 512 if bs <= 4 else 4096
code2wav_batched = 8192 if bs <= 4 else 32768
print(json.dumps({
    "0": {"max_num_seqs": bs, "gpu_memory_utilization": mem_t, "max_num_batched_tokens": talker_batched},
    "1": {"max_num_seqs": bs, "gpu_memory_utilization": mem_c, "max_num_batched_tokens": code2wav_batched},
}))
PYEOF
)

# Parse args
RUN_ASYNC=true
RUN_HF=true
for arg in "$@"; do
    case "$arg" in
        --async-only) RUN_HF=false ;;
        --hf-only) RUN_ASYNC=false ;;
        --skip-hf) RUN_HF=false ;;
    esac
done

mkdir -p "${RESULT_DIR}"

echo "============================================================"
echo " Qwen3-TTS Benchmark"
echo "============================================================"
echo " GPU:            ${GPU_DEVICE}"
echo " Model:          ${MODEL}"
echo " Prompts:        ${NUM_PROMPTS}"
echo " Concurrency:    ${CONCURRENCY}"
echo " Port:           ${PORT}"
echo " Deploy config:  ${DEPLOY_CONFIG}"
echo " Batch size:     ${BATCH_SIZE}"
echo " GPU mem T/C:    ${GPU_MEM_TALKER} / ${GPU_MEM_CODE2WAV}"
echo " Results:        ${RESULT_DIR}"
echo " Task type:      ${TASK_TYPE}"
echo "============================================================"

# Start server and wait for it to be ready
start_server() {
    local config_name="$1"
    local log_file="${RESULT_DIR}/server_${config_name}_${TIMESTAMP}.log"

    echo ""
    echo "Starting server with config: ${config_name}"
    echo "  Deploy config: ${DEPLOY_CONFIG}"
    echo "  Stage overrides: ${STAGE_OVERRIDES}"
    echo "  Log file: ${log_file}"

    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" \
    python -m vllm_omni.entrypoints.cli.main serve "${MODEL}" \
        --omni \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --deploy-config "${DEPLOY_CONFIG}" \
        --stage-overrides "${STAGE_OVERRIDES}" \
        --stage-init-timeout 120 \
        --trust-remote-code \
        --disable-log-stats \
        > "${log_file}" 2>&1 &

    SERVER_PID=$!
    echo "  Server PID: ${SERVER_PID}"

    # Wait for server to be ready
    echo "  Waiting for server to be ready..."
    local max_wait=300
    local waited=0
    while [ ${waited} -lt ${max_wait} ]; do
        if curl -sf "http://127.0.0.1:${PORT}/v1/models" > /dev/null 2>&1; then
            echo "  Server is ready! (waited ${waited}s)"
            return 0
        fi
        # Check if process is still alive
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

# Stop the server
stop_server() {
    if [ -n "${SERVER_PID:-}" ]; then
        echo "  Stopping server (PID: ${SERVER_PID})..."
        kill ${SERVER_PID} 2>/dev/null || true
        wait ${SERVER_PID} 2>/dev/null || true
        # Kill any remaining child processes on the port
        local pids
        pids=$(lsof -ti:${PORT} 2>/dev/null || true)
        if [ -n "${pids}" ]; then
            echo "  Cleaning up remaining processes on port ${PORT}..."
            echo "${pids}" | xargs kill -9 2>/dev/null || true
        fi
        echo "  Server stopped."
        SERVER_PID=""
    fi
}

# Cleanup on exit
trap 'stop_server' EXIT

# Run benchmark for a given config
run_bench() {
    local config_name="$1"

    echo ""
    echo "============================================================"
    echo " Benchmarking: ${config_name}"
    echo "============================================================"

    start_server "${config_name}"

    # Convert concurrency string to args
    local conc_args=""
    for c in ${CONCURRENCY}; do
        conc_args="${conc_args} ${c}"
    done

    cd "${PROJECT_ROOT}"
    python "${SCRIPT_DIR}/vllm_omni/bench_tts_serve.py" \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --num-prompts "${NUM_PROMPTS}" \
        --max-concurrency ${conc_args} \
        --num-warmups "${NUM_WARMUPS}" \
        --config-name "${config_name}" \
        --result-dir "${RESULT_DIR}" \
        --task-type "${TASK_TYPE}"

    stop_server

    # Allow GPU memory to settle
    sleep 5
}

# Run vllm-omni benchmark
if [ "${RUN_ASYNC}" = true ]; then
    run_bench "async_chunk"
fi

# Run HuggingFace baseline benchmark
if [ "${RUN_HF}" = true ]; then
    echo ""
    echo "============================================================"
    echo " Benchmarking: HuggingFace transformers (offline)"
    echo "============================================================"

    cd "${PROJECT_ROOT}"
    python "${SCRIPT_DIR}/transformers/bench_tts_hf.py" \
        --model "${MODEL}" \
        --num-prompts "${NUM_PROMPTS}" \
        --num-warmups "${NUM_WARMUPS}" \
        --gpu-device "${GPU_DEVICE}" \
        --config-name "hf_transformers" \
        --result-dir "${RESULT_DIR}" \
        --task-type "${TASK_TYPE}"

    # Allow GPU memory to settle
    sleep 5
fi

# Plot results
echo ""
echo "============================================================"
echo " Generating plots..."
echo "============================================================"

RESULT_FILES=""
LABELS=""

if [ "${RUN_ASYNC}" = true ]; then
    ASYNC_FILE=$(ls -t "${RESULT_DIR}"/bench_async_chunk_*.json 2>/dev/null | head -1)
    if [ -n "${ASYNC_FILE}" ]; then
        RESULT_FILES="${ASYNC_FILE}"
        LABELS="async_chunk"
    fi
fi

if [ "${RUN_HF}" = true ]; then
    HF_FILE=$(ls -t "${RESULT_DIR}"/bench_hf_transformers_*.json 2>/dev/null | head -1)
    if [ -n "${HF_FILE}" ]; then
        if [ -n "${RESULT_FILES}" ]; then
            RESULT_FILES="${RESULT_FILES} ${HF_FILE}"
            LABELS="${LABELS} hf_transformers"
        else
            RESULT_FILES="${HF_FILE}"
            LABELS="hf_transformers"
        fi
    fi
fi

if [ -n "${RESULT_FILES}" ]; then
    python "${SCRIPT_DIR}/plot_results.py" \
        --results ${RESULT_FILES} \
        --labels ${LABELS} \
        --output "${RESULT_DIR}/qwen3_tts_benchmark_${TIMESTAMP}.png"
fi

echo ""
echo "============================================================"
echo " Benchmark complete!"
echo " Results: ${RESULT_DIR}"
echo "============================================================"
