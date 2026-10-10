#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Launch the Audio8 TTS Preview server + Gradio demo together.
#
# Usage:
#   ./run_gradio_demo.sh
#   CUDA_VISIBLE_DEVICES=0 PORT=8092 GRADIO_PORT=7861 ./run_gradio_demo.sh

set -e

MODEL="${MODEL:-Audio8/Audio8-TTS-Preview-0.6b}"
PORT="${PORT:-8092}"
GRADIO_PORT="${GRADIO_PORT:-7861}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Starting Audio8 TTS Preview server (port $PORT)..."
vllm serve "$MODEL" \
    --omni \
    --host 0.0.0.0 \
    --port "$PORT" &
SERVER_PID=$!

cleanup() {
    echo "Stopping server (PID $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
}
trap cleanup EXIT

echo "Waiting for server to start..."
for _ in $(seq 1 120); do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "Server ready."
        break
    fi
    sleep 2
done

echo "Starting Gradio demo (port $GRADIO_PORT)..."
python "$SCRIPT_DIR/gradio_demo.py" \
    --api-base "http://localhost:$PORT" \
    --port "$GRADIO_PORT"
