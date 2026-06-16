#!/bin/bash
# Launch a vLLM-Omni server for CSM-1B (Sesame) text-to-speech.
#
# CSM-1B is a 2-stage TTS model: Stage 0 is a Llama-style backbone AR that
# samples codebook-0 per 80 ms frame and runs a 31-step depth decoder inline to
# produce the full 32-code frame; Stage 1 is the Mimi vocoder (code2wav). The
# stage topology + per-stage GPU memory split is defined in the deploy config.
#
# Usage:
#   ./run_server.sh
#   PORT=8091 ./run_server.sh
#   HF_HUB_OFFLINE=1 ./run_server.sh        # use only the local checkpoint cache

set -e

MODEL="${MODEL:-sesame/csm-1b}"
PORT="${PORT:-8091}"

echo "Starting CSM-1B server with model: $MODEL"

vllm-omni serve "$MODEL" \
    --deploy-config vllm_omni/deploy/csm.yaml \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --omni
