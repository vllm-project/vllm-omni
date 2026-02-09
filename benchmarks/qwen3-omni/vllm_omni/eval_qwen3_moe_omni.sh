#!/bin/bash
# Qwen3-Omni Benchmark Evaluation Script
# This script must be run from the vllm-omni root directory

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Navigate to vllm-omni root directory (4 levels up from script location)
VLLM_OMNI_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$VLLM_OMNI_ROOT" || { echo "Error: Failed to navigate to vllm-omni directory"; exit 1; }

echo "Working directory: $(pwd)"

# Verify we're in the correct directory and run benchmark
if [[ ! -d "benchmarks/qwen3-omni/vllm_omni" ]]; then
    echo "Error: Not in vllm-omni root directory. Please run from vllm-omni folder."
else
    log_dir=benchmarks/qwen3-omni/vllm_omni/logs
    outputs_dir=benchmarks/qwen3-omni/vllm_omni/outputs
    end2end_script_path=examples/offline_inference/qwen3_omni/end2end.py
    build_dataset_path=benchmarks/build_dataset/top100.txt

    python $end2end_script_path --output-wav $outputs_dir \
                      --query-type text \
                      --txt-prompts $build_dataset_path \
                      --log-stats \
                      --log-dir $log_dir
    echo "Logs and outputs are saved in ${log_dir} and ${outputs_dir} respectively:"
    echo "  - omni_llm_pipeline_text                       run dir/base name"
    echo "  - omni_llm_pipeline_text.orchestrator.stats.jsonl  orchestrator-stage latency stats"
    echo "  - omni_llm_pipeline_text.overall.stats.jsonl       overall latency/TPS stats"
    echo "  - omni_llm_pipeline_text.stage0.log                per-stage detailed logs"
    echo "  - omni_llm_pipeline_text.stage1.log"
    echo "  - omni_llm_pipeline_text.stage2.log"
    echo "Key checks: overall.stats.jsonl for end-to-end latency/TPS; orchestrator.stats.jsonl for stable per-stage latency; stage*.log for errors or long tails."
    echo "  - outputs/             Generated txt and wav files, there should be 100 text and wav files generated respectively"
fi
