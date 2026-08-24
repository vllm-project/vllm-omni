#!/bin/bash
# Launch GLM-TTS server + Gradio demo together.
#
# Usage:
#   ./run_gradio_demo.sh
#   CUDA_VISIBLE_DEVICES=0 PORT=8091 GRADIO_PORT=7860 ./run_gradio_demo.sh

set -e

MODEL="${MODEL:-zai-org/GLM-TTS}"
PORT="${PORT:-8091}"
GRADIO_PORT="${GRADIO_PORT:-7860}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

echo "Starting GLM-TTS server (port $PORT)..."
FLASHINFER_DISABLE_VERSION_CHECK=1 \
vllm-omni serve "$MODEL" \
    --deploy-config "$REPO_ROOT/vllm_omni/deploy/glm_tts.yaml" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --enforce-eager \
    --omni &
SERVER_PID=$!

cleanup() {
    echo "Stopping server (PID $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
}
trap cleanup EXIT

# Wait for server to be ready.
echo "Waiting for server to start..."
for i in $(seq 1 120); do
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
