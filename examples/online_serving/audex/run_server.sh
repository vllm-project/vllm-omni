#!/bin/bash
# Launch the vLLM-Omni server for Audex (Nemotron-Labs-Audex-2B) TTS.
#
# English-only plain TTS via /v1/audio/speech (no voice cloning; single
# built-in voice). Pass the HF repo ROOT as MODEL — per-stage subfolders
# resolve automatically.
#
# Usage:
#   ./run_server.sh                 # default port 8097, GPU 0
#   PORT=8098 GPUS=1 ./run_server.sh
#   MODEL=/path/to/local/snapshot ./run_server.sh

set -e

MODEL="${MODEL:-nvidia/Nemotron-Labs-Audex-2B}"
PORT="${PORT:-8097}"
GPUS="${GPUS:-0}"

echo "Starting Audex TTS server"
echo "  MODEL=$MODEL"
echo "  PORT=$PORT"
echo "  CUDA_VISIBLE_DEVICES=$GPUS"

CUDA_VISIBLE_DEVICES="$GPUS" \
vllm-omni serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --omni
