#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

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

stage_overrides=$(python3 -c '
import json
import sys

gpu_util = float(sys.argv[1])
print(json.dumps({"0": {"gpu_memory_utilization": gpu_util}}))
' "$GPU_UTIL")

CUDA_VISIBLE_DEVICES="$GPUS" \
VLLM_USE_DEEP_GEMM=0 \
VLLM_MOE_USE_DEEP_GEMM=0 \
vllm-omni serve "$MODEL" \
    --deploy-config vllm_omni/deploy/higgs_multimodal_qwen3.yaml \
    --host 0.0.0.0 \
    --port "$PORT" \
    --stage-overrides "$stage_overrides" \
    --trust-remote-code \
    --omni
