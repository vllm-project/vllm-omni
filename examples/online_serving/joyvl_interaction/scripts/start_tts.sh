#!/bin/bash
# Serve Qwen3-TTS (vLLM-Omni) and run the webui<->TTS bridge.
# Then start the webui with:  TTS_URL=ws://127.0.0.1:8092/v1/tts bash scripts/start_server.sh
set -euo pipefail

TTS_MODEL="${TTS_MODEL:-Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice}"
TTS_PORT="${TTS_PORT:-8091}"
BRIDGE_PORT="${BRIDGE_PORT:-8092}"
GPU="${GPU:-1}"
VOICE="${VOICE:-vivian}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

CUDA_VISIBLE_DEVICES="${GPU}" vllm-omni serve "${TTS_MODEL}" \
    --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
    --host 0.0.0.0 --port "${TTS_PORT}" \
    --gpu-memory-utilization 0.9 --trust-remote-code --omni &
TTS_PID=$!
trap 'kill ${TTS_PID} 2>/dev/null || true' EXIT

# wait for the TTS server, then run the bridge in the foreground
until curl -s "http://127.0.0.1:${TTS_PORT}/health" >/dev/null 2>&1; do sleep 2; done
python "${SCRIPT_DIR}/../bridges/tts_bridge.py" \
    --port "${BRIDGE_PORT}" \
    --backend-url "ws://127.0.0.1:${TTS_PORT}/v1/audio/speech/stream" \
    --voice "${VOICE}"
