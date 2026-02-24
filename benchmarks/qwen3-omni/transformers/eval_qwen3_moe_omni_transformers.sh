#!/bin/bash
# Qwen3-Omni Transformers Benchmark Evaluation Script
# This script must be run from the vllm-omni root directory

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Navigate to vllm-omni root directory (4 levels up from script location)
VLLM_OMNI_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$VLLM_OMNI_ROOT" || { echo "Error: Failed to navigate to vllm-omni directory"; exit 1; }

echo "Working directory: $(pwd)"
# Verify we're in the correct directory and run benchmark
if [[ ! -f "benchmarks/qwen3-omni/transformers/qwen3_omni_moe_transformers.py" ]]; then
    echo "Error: Not in vllm-omni root directory. Please run from vllm-omni folder."
else
    cd benchmarks/qwen3-omni/transformers

    python qwen3_omni_moe_transformers.py --prompts-file ../../build_dataset/top100.txt --num-prompts 100

    echo "Logs and outputs are saved to $(pwd)/benchmark_results:"
    echo "  - perf_stats.json    Aggregated/per-prompt TPS and latency (thinker/talker/code2wav/overall)"
    echo "  - results.json       Per-prompt outputs and audio paths"
    echo "  - audio/             Generated wav files, there should be 100 wav file generated"
    echo "Key checks: overall_tps and *_tps_avg should be non-zero and stable; investigate 0/NaN or unusually low TPS/long-tail latency."
fi
