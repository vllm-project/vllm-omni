#!/bin/bash
# Launch vLLM-Omni server + Gradio demo for Raon-Speech
set -e

MODEL="${1:-KRAFTON/Raon-Speech-9B}"
PORT="${2:-8091}"
GRADIO_PORT="${3:-7860}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "Starting Raon-Speech server on port $PORT..."
vllm-omni serve "$MODEL" \
    --stage-configs-path "$REPO_ROOT/vllm_omni/model_executor/stage_configs/raon.yaml" \
    --host 0.0.0.0 --port "$PORT" \
    --gpu-memory-utilization 0.9 --trust-remote-code --omni &
SERVER_PID=$!

cleanup() {
    echo "Stopping server (PID $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
}
trap cleanup EXIT

echo "Waiting for server..."
SERVER_READY=false
for i in $(seq 1 60); do
    curl -s "http://localhost:$PORT/health" >/dev/null 2>&1 && { SERVER_READY=true; break; }
    sleep 10
done
if [ "$SERVER_READY" = false ]; then
    echo "WARNING: Server did not become healthy after 10 minutes."
fi
echo "Server ready."

echo "Starting Gradio demo on port $GRADIO_PORT..."
python "$SCRIPT_DIR/gradio_demo.py" \
    --api-base "http://localhost:$PORT" --port "$GRADIO_PORT"
