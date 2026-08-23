#!/bin/bash
# Serve Qwen3-ASR (the same audio-LLM AURA uses as its ASR stage) and run the
# webui<->ASR bridge. Then start the webui with:
#   ASR_URL=ws://127.0.0.1:8093/v1/asr bash scripts/start_server.sh
set -euo pipefail

ASR_MODEL="${ASR_MODEL:-Qwen/Qwen3-ASR-1.7B}"
ASR_PORT="${ASR_PORT:-8094}"
BRIDGE_PORT="${BRIDGE_PORT:-8093}"
GPU="${GPU:-2}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Qwen3-ASR is audio-in / text-out; served as a normal multimodal chat model.
CUDA_VISIBLE_DEVICES="${GPU}" vllm serve "${ASR_MODEL}" \
    --host 0.0.0.0 --port "${ASR_PORT}" \
    --gpu-memory-utilization 0.3 --trust-remote-code &
ASR_PID=$!
trap 'kill ${ASR_PID} 2>/dev/null || true' EXIT

until curl -s "http://127.0.0.1:${ASR_PORT}/health" >/dev/null 2>&1; do sleep 2; done
python "${SCRIPT_DIR}/../bridges/asr_bridge.py" \
    --port "${BRIDGE_PORT}" \
    --backend-url "http://127.0.0.1:${ASR_PORT}" \
    --model "${ASR_MODEL}"
