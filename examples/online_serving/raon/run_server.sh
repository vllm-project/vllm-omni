#!/bin/bash
# Launch vLLM-Omni server for Raon-Speech
#
# Usage:
#   ./run_server.sh                           # Default: KRAFTON/Raon-Speech-9B
#   ./run_server.sh KRAFTON/Raon-Speech-9B    # Explicit model path

set -e

MODEL="${1:-KRAFTON/Raon-Speech-9B}"

echo "Starting Raon-Speech server with model: $MODEL"

vllm-omni serve "$MODEL" \
    --stage-configs-path vllm_omni/model_executor/stage_configs/raon.yaml \
    --host 0.0.0.0 \
    --port 8091 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --omni
