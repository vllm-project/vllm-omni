#!/usr/bin/env bash
# =============================================================================
# Diffusion Model Profiling Script
# Collects per-layer breakdown + torch.profiler kernel trace for any diffusion
# model served by vllm-omni.
#
# Usage:
#   bash run_diffusion_profiling.sh <model> [phase]
#   phase: all | stage_durations | torch_profile  (default: all)
#
# Example:
#   bash run_diffusion_profiling.sh tencent/HunyuanImage-3.0-Instruct all
#   bash run_diffusion_profiling.sh Qwen/Qwen-Image-2512 stage_durations
# =============================================================================
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model> [all|stage_durations|torch_profile]"
    exit 1
fi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL="$1"
PHASE="${2:-all}"
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-/path/to/vllm-omni}"
MODEL_SHORT="$(echo "$MODEL" | sed 's|.*/||; s|[^a-zA-Z0-9_-]|_|g')"
RESULT_DIR="/tmp/${MODEL_SHORT}_profiling_results"
TORCH_PROFILE_DIR="/tmp/${MODEL_SHORT}_torch_traces"
PORT=8100
HOST="127.0.0.1"
NUM_PROMPTS=3
NUM_STEPS=50
WIDTH=1024
HEIGHT=1024

mkdir -p "$RESULT_DIR" "$TORCH_PROFILE_DIR"

# ---------------------------------------------------------------------------
# 环境检查（规则 18：启动前三连）
# ---------------------------------------------------------------------------
preflight() {
    echo "=== Preflight checks ==="
    echo "--- GPU status ---"
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
    echo "--- HF cache ---"
    echo "HF_HOME=${HF_HOME:-not set}"
    ls "${HF_HOME:-/root/.cache/huggingface}/hub/" 2>/dev/null | grep "models--" | head -5 || echo "(no models cached)"
    echo "--- Cache vars ---"
    env | grep -iE "cache|hf_home|offline" || true
    echo "--- unset TRANSFORMERS_CACHE ---"
    unset TRANSFORMERS_CACHE 2>/dev/null || true
    echo "=== Preflight done ==="
}

# ---------------------------------------------------------------------------
# Stage config YAML 生成
# ---------------------------------------------------------------------------
write_stage_config() {
    local name="$1" tp="$2" quant="$3"
    shift 3
    local extra_parallel="$*"
    local yaml_file="$RESULT_DIR/stage_config_${name}.yaml"

    cat > "$yaml_file" <<YAML
stage_args:
  - stage_id: 0
    stage_type: diffusion
    engine_args:
      enforce_eager: true
      distributed_executor_backend: mp
      quantization: ${quant}
      parallel_config:
        tensor_parallel_size: ${tp}
${extra_parallel}
    final_output: true
    final_output_type: image
YAML
    echo "$yaml_file"
}

write_stage_config_with_torch_profiler() {
    local name="$1" tp="$2" quant="$3"
    shift 3
    local extra_parallel="$*"
    local yaml_file="$RESULT_DIR/stage_config_${name}_torch.yaml"

    cat > "$yaml_file" <<YAML
stage_args:
  - stage_id: 0
    stage_type: diffusion
    engine_args:
      enforce_eager: true
      distributed_executor_backend: mp
      quantization: ${quant}
      parallel_config:
        tensor_parallel_size: ${tp}
${extra_parallel}
      profiler_config:
        profiler: torch
        torch_profiler_dir: ${TORCH_PROFILE_DIR}/${name}
        delay_iterations: 1
        warmup_iterations: 0
        active_iterations: 1
        max_iterations: 2
    final_output: true
    final_output_type: image
YAML
    echo "$yaml_file"
}

# ---------------------------------------------------------------------------
# Server 启停
# ---------------------------------------------------------------------------
start_server() {
    local config_yaml="$1"
    echo ">>> Starting server with config: $config_yaml"
    python -m vllm_omni.entrypoints.cli.main serve "$MODEL" \
        --stage-configs-path "$config_yaml" \
        --enable-diffusion-pipeline-profiler \
        --port "$PORT" \
        2>&1 | tee "$RESULT_DIR/server_$(basename "$config_yaml" .yaml).log" &
    SERVER_PID=$!

    echo ">>> Waiting for server on $HOST:$PORT ..."
    local waited=0
    while ! curl -s "http://$HOST:$PORT/health" > /dev/null 2>&1; do
        sleep 5
        waited=$((waited + 5))
        if [ $waited -ge 600 ]; then
            echo "ERROR: Server did not start within 600s"
            kill $SERVER_PID 2>/dev/null || true
            return 1
        fi
    done
    echo ">>> Server ready (waited ${waited}s)"
}

stop_server() {
    echo ">>> Stopping server (PID=$SERVER_PID)"
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 5
    # 确认 GPU 显存释放
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
}

# ---------------------------------------------------------------------------
# Benchmark 运行
# ---------------------------------------------------------------------------
run_benchmark() {
    local name="$1" output_file="$RESULT_DIR/bench_${name}.json"
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

# ---------------------------------------------------------------------------
# Torch profiler: 发一个请求触发 trace
# ---------------------------------------------------------------------------
run_torch_profile_request() {
    local name="$1"
    echo ">>> Sending profiling request for: $name"

    # start_profile
    curl -s -X POST "http://$HOST:$PORT/start_profile" || true
    sleep 2

    # 发一个请求
    curl -s -X POST "http://$HOST:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL\",
            \"messages\": [{\"role\": \"user\", \"content\": \"A photo of a cat sitting on a windowsill\"}],
            \"width\": $WIDTH,
            \"height\": $HEIGHT,
            \"num_inference_steps\": $NUM_STEPS
        }" -o "$RESULT_DIR/profile_response_${name}.json"

    sleep 2
    # stop_profile
    curl -s -X POST "http://$HOST:$PORT/stop_profile" || true
    sleep 5
    echo ">>> Torch traces should be in: $TORCH_PROFILE_DIR/$name/"
}

# ---------------------------------------------------------------------------
# 三种配置定义
# ---------------------------------------------------------------------------
declare -A CONFIGS
CONFIGS[tp4_fp8]="4 fp8 "
CONFIGS[tp2_fp8_sp2]="2 fp8         sequence_parallel_size: 2
        ulysses_degree: 2"
CONFIGS[tp2_fp8_cfgp2]="2 fp8         cfg_parallel_size: 2"

CONFIG_NAMES=("tp4_fp8" "tp2_fp8_sp2" "tp2_fp8_cfgp2")

# ---------------------------------------------------------------------------
# Phase 1: Stage durations (lightweight pipeline profiler)
# ---------------------------------------------------------------------------
run_stage_durations() {
    echo ""
    echo "============================================================"
    echo "  Phase 1: Per-layer stage_durations (pipeline profiler)"
    echo "============================================================"

    for name in "${CONFIG_NAMES[@]}"; do
        echo ""
        echo "--- Config: $name ---"
        local tp quant extra
        read -r tp quant <<< "$(echo "${CONFIGS[$name]}" | head -1 | awk '{print $1, $2}')"

        local extra_lines=""
        if [ "$name" = "tp2_fp8_sp2" ]; then
            extra_lines="        sequence_parallel_size: 2
        ulysses_degree: 2"
        elif [ "$name" = "tp2_fp8_cfgp2" ]; then
            extra_lines="        cfg_parallel_size: 2"
        fi

        local yaml_file
        yaml_file=$(write_stage_config "$name" "$tp" "$quant" "$extra_lines")

        start_server "$yaml_file"
        run_benchmark "$name"
        stop_server
    done

    echo ""
    echo "=== Stage duration results ==="
    for name in "${CONFIG_NAMES[@]}"; do
        echo ""
        echo "--- $name ---"
        python3 -c "
import json, sys
with open('$RESULT_DIR/bench_${name}.json') as f:
    data = json.load(f)
print(f'  Throughput: {data.get(\"throughput_qps\", \"N/A\"):.4f} qps')
print(f'  Latency mean: {data.get(\"latency_mean\", \"N/A\"):.3f}s')
print(f'  Peak memory: {data.get(\"peak_memory_mb_mean\", \"N/A\"):.0f} MB')
sd = data.get('stage_durations_mean', {})
if sd:
    print('  Stage durations (mean):')
    for k, v in sorted(sd.items(), key=lambda x: -x[1]):
        print(f'    {k}: {v:.4f}s')
else:
    print('  (no stage_durations in output)')
" 2>/dev/null || echo "  (parse error, check $RESULT_DIR/bench_${name}.json)"
    done
}

# ---------------------------------------------------------------------------
# Phase 2: Torch profiler (kernel-level)
# ---------------------------------------------------------------------------
run_torch_profiling() {
    echo ""
    echo "============================================================"
    echo "  Phase 2: Torch profiler (kernel traces for fp8 GEMM)"
    echo "============================================================"
    echo "  Only running tp4_fp8 config (representative for fp8 path)"
    echo ""

    local name="tp4_fp8"
    local tp=4 quant="fp8"
    mkdir -p "$TORCH_PROFILE_DIR/$name"

    local yaml_file
    yaml_file=$(write_stage_config_with_torch_profiler "$name" "$tp" "$quant" "")

    start_server "$yaml_file"
    run_torch_profile_request "$name"
    stop_server

    echo ""
    echo "=== Torch profiler traces ==="
    echo "Traces saved to: $TORCH_PROFILE_DIR/$name/"
    ls -la "$TORCH_PROFILE_DIR/$name/" 2>/dev/null || echo "(empty)"
    echo ""
    echo "To analyze top-N kernels:"
    echo "  python3 tools/analyze_torch_trace.py $TORCH_PROFILE_DIR/$name/"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
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
        echo "Usage: $0 [all|stage_durations|torch_profile]"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "  All done. Results in: $RESULT_DIR"
echo "============================================================"
