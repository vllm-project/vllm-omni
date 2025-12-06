#!/bin/bash
# Qwen3-Omni Transformers Benchmark Evaluation Script
# This script must be run from the vllm-omni root directory

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Navigate to vllm-omni root directory (4 levels up from script location)
VLLM_OMNI_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$VLLM_OMNI_ROOT" || { echo "Error: Failed to navigate to vllm-omni directory"; exit 1; }

echo "Working directory: $(pwd)"
# Verify we're in the correct directory and run benchmark
if [[ ! -f "examples/benchmark/qwen3-omni/transformers/qwen3_omni_moe_transformers.py" ]]; then
    echo "Error: Not in vllm-omni root directory. Please run from vllm-omni folder."
else
    cd examples/benchmark/qwen3-omni/transformers

    python qwen3_omni_moe_transformers.py --prompts_file ../../build_dataset/top100.txt --num_prompts 100
fi