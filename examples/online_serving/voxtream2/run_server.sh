#!/bin/bash
# Launch vLLM-Omni server for Voxtream2 TTS.
#
# Usage:
#   ./examples/online_serving/voxtream2/run_server.sh
#   CUDA_VISIBLE_DEVICES=3 VOXTREAM2_ROOT=voxtream ./examples/online_serving/voxtream2/run_server.sh

set -e

MODEL="${MODEL:-herimor/voxtream2}"
PORT="${PORT:-8091}"
VOXTREAM2_ROOT="${VOXTREAM2_ROOT:-voxtream}"
STAGE_CONFIG="${STAGE_CONFIG:-vllm_omni/model_executor/stage_configs/voxtream2_1stage.yaml}"

export VOXTREAM2_ROOT
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

echo "Starting Voxtream2 server with model: $MODEL"
echo "Voxtream root: $VOXTREAM2_ROOT"

vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --stage-configs-path "$STAGE_CONFIG" \
    --trust-remote-code \
    --enforce-eager \
    --omni
