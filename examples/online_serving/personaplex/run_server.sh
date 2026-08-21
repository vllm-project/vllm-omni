#!/bin/bash
# PersonaPlex full-duplex online serving.
#
# Serves the OFFICIAL PersonaPlex web client at / (auto-downloaded from
# nvidia/personaplex-7b-v1) and the Moshi WebSocket protocol at /api/chat,
# backed by the native vLLM-Omni engine. Requires HF access to the gated repo.

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8124}"

echo "Starting PersonaPlex duplex server on ${HOST}:${PORT}..."
echo "Open http://localhost:${PORT}/ for the official web client (run it near the server)."

python -m vllm_omni.experimental.fullduplex.personaplex.serving.server \
    --host "$HOST" --port "$PORT"
