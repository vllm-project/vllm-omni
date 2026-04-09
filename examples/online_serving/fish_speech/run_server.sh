#!/bin/bash
# Launch vLLM-Omni server for Fish Speech S2 Pro
#
# Usage:
#   ./run_server.sh
#   CUDA_VISIBLE_DEVICES=0 ./run_server.sh

set -e

MODEL="${MODEL:-fishaudio/s2-pro}"
PORT="${PORT:-8091}"

echo "Starting Fish Speech S2 Pro server with model: $MODEL"

FLASHINFER_DISABLE_VERSION_CHECK=1 \
vllm-omni serve "$MODEL" \
    --stage-configs-path vllm_omni/model_executor/stage_configs/fish_speech_s2_pro.yaml \
    --host 0.0.0.0 \
    --port "$PORT" \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --enforce-eager \
    --omni
