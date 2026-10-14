#!/bin/bash
# Launch vLLM-Omni server for higgs-audio v3.
#
# Supports plain text TTS and voice cloning via /v1/audio/speech.
#
# Usage:
#   ./run_server.sh                 # default port 8095, GPU 0
#   PORT=8096 GPUS=0,1 ./run_server.sh
#   MODEL=/path/to/local/checkpoint ./run_server.sh

set -e

MODEL="${MODEL:-bosonai/higgs-audio-v3-tts-4b}"
PORT="${PORT:-8095}"
GPUS="${GPUS:-0}"
GPU_UTIL="${GPU_UTIL:-0.6}"

echo "Starting higgs-audio v3 server"
echo "  MODEL=$MODEL"
echo "  PORT=$PORT"
echo "  CUDA_VISIBLE_DEVICES=$GPUS"

CUDA_VISIBLE_DEVICES="$GPUS" \
VLLM_USE_DEEP_GEMM=0 \
VLLM_MOE_USE_DEEP_GEMM=0 \
vllm-omni serve "$MODEL" \
    --deploy-config vllm_omni/deploy/higgs_multimodal_qwen3.yaml \
    --host 0.0.0.0 \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --trust-remote-code \
    --omni
