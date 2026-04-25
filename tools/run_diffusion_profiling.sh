#!/usr/bin/env bash
# Diffusion profiling helper for vLLM-Omni.
#
# Usage:
#   bash tools/run_diffusion_profiling.sh <model> [all|stage_durations|torch_profile]
#
# Example:
#   bash tools/run_diffusion_profiling.sh tencent/HunyuanImage-3.0-Instruct all

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model> [all|stage_durations|torch_profile]"
    exit 1
fi

MODEL="$1"
PHASE="${2:-all}"
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-$(pwd)}"
MODEL_SHORT="$(echo "$MODEL" | sed 's|.*/||; s|[^a-zA-Z0-9_-]|_|g')"
RESULT_DIR="${RESULT_DIR:-/tmp/${MODEL_SHORT}_profiling_results}"
TORCH_PROFILE_DIR="${TORCH_PROFILE_DIR:-/tmp/${MODEL_SHORT}_torch_traces}"
PORT="${PORT:-8100}"
HOST="${HOST:-127.0.0.1}"
NUM_PROMPTS="${NUM_PROMPTS:-3}"
NUM_STEPS="${NUM_STEPS:-50}"
WIDTH="${WIDTH:-1024}"
HEIGHT="${HEIGHT:-1024}"

SERVER_PID=""
declare -a SERVER_ARGS
CONFIG_NAMES=("tp4_fp8" "tp2_fp8_sp2" "tp2_fp8_cfgp2")

mkdir -p "$RESULT_DIR" "$TORCH_PROFILE_DIR"

preflight() {
    echo "=== Preflight checks ==="
    echo "--- GPU status ---"
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
    echo "--- HF cache ---"
    echo "HF_HOME=${HF_HOME:-not set}"
    ls "${HF_HOME:-/root/.cache/huggingface}/hub/" 2>/dev/null | grep "models--" | head -5 || echo "(no models cached)"
    echo "--- Cache vars ---"
    env | grep -iE "cache|hf_home|offline" || true
    unset TRANSFORMERS_CACHE 2>/dev/null || true
    echo "=== Preflight done ==="
}

build_server_args() {
    local name="$1"
    local profiler_json="${2:-}"

    case "$name" in
        tp4_fp8)
            SERVER_ARGS=(
                --tensor-parallel-size 4
                --quantization fp8
                --distributed-executor-backend mp
                --enforce-eager
                --enable-diffusion-pipeline-profiler
            )
            ;;
        tp2_fp8_sp2)
            SERVER_ARGS=(
                --tensor-parallel-size 2
                --usp 2
                --quantization fp8
                --distributed-executor-backend mp
                --enforce-eager
                --enable-diffusion-pipeline-profiler
            )
            ;;
        tp2_fp8_cfgp2)
            SERVER_ARGS=(
                --tensor-parallel-size 2
                --cfg-parallel-size 2
                --quantization fp8
                --distributed-executor-backend mp
                --enforce-eager
                --enable-diffusion-pipeline-profiler
            )
            ;;
        *)
            echo "Unknown config: $name"
            exit 1
            ;;
    esac

    if [ -n "$profiler_json" ]; then
        SERVER_ARGS+=(--profiler-config "$profiler_json")
    fi
}

start_server() {
    local name="$1"
    local profiler_json="${2:-}"
    local log_file="$RESULT_DIR/server_${name}.log"

    build_server_args "$name" "$profiler_json"
    echo ">>> Starting server for $name"
    python -m vllm_omni.entrypoints.cli.main serve "$MODEL" \
        --omni \
        --host "$HOST" \
        --port "$PORT" \
        "${SERVER_ARGS[@]}" \
        2>&1 | tee "$log_file" &
    SERVER_PID=$!

    echo ">>> Waiting for server on $HOST:$PORT ..."
    local waited=0
    while ! curl -s "http://$HOST:$PORT/health" > /dev/null 2>&1; do
        sleep 5
        waited=$((waited + 5))
        if [ "$waited" -ge 600 ]; then
            echo "ERROR: Server did not start within 600s"
            stop_server
            return 1
        fi
    done
    echo ">>> Server ready (waited ${waited}s)"
}

stop_server() {
    if [ -n "${SERVER_PID:-}" ]; then
        echo ">>> Stopping server (PID=$SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
        sleep 5
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader || true
    fi
}

run_benchmark() {
    local name="$1"
    local output_file="$RESULT_DIR/bench_${name}.json"

    echo ">>> Running benchmark: $name"
    python "$VLLM_OMNI_DIR/benchmarks/diffusion/diffusion_benchmark_serving.py" \
        --backend vllm-omni \
        --host "$HOST" \
        --port "$PORT" \
        --dataset random \
        --task t2i \
        --width "$WIDTH" \
        --height "$HEIGHT" \
        --num-inference-steps "$NUM_STEPS" \
        --num-prompts "$NUM_PROMPTS" \
        --max-concurrency 1 \
        --output-file "$output_file"
    echo ">>> Results saved to: $output_file"
}

run_torch_profile_request() {
    local name="$1"

    echo ">>> Sending profiling request for: $name"
    curl -s -X POST "http://$HOST:$PORT/start_profile" || true
    sleep 2

    curl -s -X POST "http://$HOST:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"messages\": [{\"role\": \"user\", \"content\": \"A cinematic photo of a glass observatory on Mars at sunrise\"}],
            \"extra_body\": {
                \"height\": $HEIGHT,
                \"width\": $WIDTH,
                \"num_inference_steps\": $NUM_STEPS,
                \"guidance_scale\": 5.0,
                \"seed\": 42
            }
        }" -o "$RESULT_DIR/profile_response_${name}.json"

    sleep 2
    curl -s -X POST "http://$HOST:$PORT/stop_profile" || true
    sleep 5
    echo ">>> Torch traces should be in: $TORCH_PROFILE_DIR/$name/"
}

print_stage_duration_summary() {
    echo ""
    echo "=== Stage duration results ==="
    for name in "${CONFIG_NAMES[@]}"; do
        echo ""
        echo "--- $name ---"
        python3 -c "
import json
path = '$RESULT_DIR/bench_${name}.json'
with open(path) as f:
    data = json.load(f)
print(f'  Throughput: {data.get(\"throughput_qps\", 0):.4f} qps')
print(f'  Latency mean: {data.get(\"latency_mean\", 0):.3f}s')
print(f'  Peak memory: {data.get(\"peak_memory_mb_mean\", 0):.0f} MB')
stage_durations = data.get('stage_durations_mean', {})
if stage_durations:
    print('  Stage durations (mean):')
    for key, value in sorted(stage_durations.items(), key=lambda item: -item[1]):
        print(f'    {key}: {value:.4f}s')
else:
    print('  (no stage_durations in output)')
" 2>/dev/null || echo "  (parse error, check $RESULT_DIR/bench_${name}.json)"
    done
}

run_stage_durations() {
    echo ""
    echo "============================================================"
    echo "  Phase 1: Per-layer stage durations"
    echo "============================================================"

    for name in "${CONFIG_NAMES[@]}"; do
        echo ""
        echo "--- Config: $name ---"
        start_server "$name"
        run_benchmark "$name"
        stop_server
    done

    print_stage_duration_summary
}

run_torch_profiling() {
    echo ""
    echo "============================================================"
    echo "  Phase 2: Torch profiler"
    echo "============================================================"
    echo "  Only running tp4_fp8 config as the representative FP8 path"
    echo ""

    local name="tp4_fp8"
    local trace_dir="$TORCH_PROFILE_DIR/$name"
    mkdir -p "$trace_dir"

    local profiler_json
    profiler_json=$(printf '{"profiler":"torch","torch_profiler_dir":"%s","delay_iterations":1,"warmup_iterations":0,"active_iterations":1,"max_iterations":2}' "$trace_dir")

    start_server "$name" "$profiler_json"
    run_torch_profile_request "$name"
    stop_server

    echo ""
    echo "=== Torch profiler traces ==="
    echo "Traces saved to: $trace_dir/"
    ls -la "$trace_dir/" 2>/dev/null || echo "(empty)"
    echo ""
    echo "To analyze top-N kernels:"
    echo "  python3 tools/analyze_torch_trace.py $trace_dir/"
}

trap stop_server EXIT

preflight

case "$PHASE" in
    all)
        run_stage_durations
        run_torch_profiling
        ;;
    stage_durations)
        run_stage_durations
        ;;
    torch_profile)
        run_torch_profiling
        ;;
    *)
        echo "Usage: $0 <model> [all|stage_durations|torch_profile]"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  All done. Results in: $RESULT_DIR"
echo "============================================================"
