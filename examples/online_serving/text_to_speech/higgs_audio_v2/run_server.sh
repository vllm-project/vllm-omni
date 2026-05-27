#!/bin/bash
# Launch vLLM-Omni server for higgs-audio v2.
#
# v1 scope: plain text -> 24 kHz speech only. Voice cloning, multi-speaker,
# ChatML rich content, and language overrides are rejected by the validator
# with explicit 4xx (see vllm_omni/entrypoints/openai/serving_speech.py).
#
# Usage:
#   ./run_server.sh                 # default port 8094, GPUs 6 and 7
#   PORT=8095 GPUS=6,7 ./run_server.sh
#   MODEL=bosonai/higgs-audio-v2-generation-3B-base ./run_server.sh

set -e

MODEL="${MODEL:-bosonai/higgs-audio-v2-generation-3B-base}"
PORT="${PORT:-8094}"
GPUS="${GPUS:-6,7}"
GPU_UTIL="${GPU_UTIL:-0.4}"

echo "Starting higgs-audio v2 server"
echo "  MODEL=$MODEL"
echo "  PORT=$PORT"
echo "  CUDA_VISIBLE_DEVICES=$GPUS"

# DeepGEMM FP8 kernels are optional and trip warmup on builds without
# the deep_gemm backend; disable them so the example works out of the box.
# Users with deep_gemm installed can re-enable via the same env vars.
CUDA_VISIBLE_DEVICES="$GPUS" \
VLLM_USE_DEEP_GEMM=0 \
VLLM_MOE_USE_DEEP_GEMM=0 \
vllm-omni serve "$MODEL" \
    --deploy-config vllm_omni/deploy/higgs_audio_v2.yaml \
    --host 0.0.0.0 \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --trust-remote-code \
    --omni
