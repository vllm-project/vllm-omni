#!/bin/bash
# Qwen3-Omni Benchmark Evaluation Script
# This script must be run from the vllm-omni root directory

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Navigate to vllm-omni root directory (4 levels up from script location)
VLLM_OMNI_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$VLLM_OMNI_ROOT" || { echo "Error: Failed to navigate to vllm-omni directory"; exit 1; }

echo "Working directory: $(pwd)"

# Verify we're in the correct directory and run benchmark
if [[ ! -d "examples/offline_inference/qwen3_omni" ]]; then
    echo "Error: Not in vllm-omni root directory. Please run from vllm-omni folder."
else
    cd examples/offline_inference/qwen3_omni

    python end2end.py --output-wav output_audio \
                      --query-type text \
                      --txt-prompts ../../benchmark/build_dataset/top100.txt \
                      --enable-stats
    echo "You can check the saved logs in $(pwd), named as:\
    omni_llm_pipeline_text\
    omni_llm_pipeline_text.orchestrator.stats.jsonl\
    omni_llm_pipeline_text.overall.stats.jsonl\
    omni_llm_pipeline_text.stage0.log\
    omni_llm_pipeline_text.stage1.log\
    omni_llm_pipeline_text.stage2.log\
    "
fi
