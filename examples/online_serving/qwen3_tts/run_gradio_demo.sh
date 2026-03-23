#!/bin/bash
# Launch both vLLM server and Gradio demo for Qwen3-TTS
#
# Usage:
#   ./run_gradio_demo.sh                                    # Default: CustomVoice
#   ./run_gradio_demo.sh --task-type VoiceDesign            # VoiceDesign model
#   ./run_gradio_demo.sh --task-type Base --gradio-port 7861
#
# Options:
#   --task-type TYPE        Task type: CustomVoice, VoiceDesign, Base (default: CustomVoice)
#   --server-port PORT      Port for vLLM server (default: 8000)
#   --gradio-port PORT      Port for Gradio demo (default: 7860)
#   --server-host HOST      Host for vLLM server (default: 0.0.0.0)
#   --gradio-ip IP          IP for Gradio demo (default: 127.0.0.1)
#   --share                 Share Gradio demo publicly

set -e

# Default values
TASK_TYPE="CustomVoice"
SERVER_PORT=8000
GRADIO_PORT=7860
SERVER_HOST="0.0.0.0"
GRADIO_IP="127.0.0.1"
GRADIO_SHARE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task-type)
            TASK_TYPE="$2"
            shift 2
            ;;
        --server-port)
            SERVER_PORT="$2"
            shift 2
            ;;
        --gradio-port)
            GRADIO_PORT="$2"
            shift 2
            ;;
        --server-host)
            SERVER_HOST="$2"
            shift 2
            ;;
        --gradio-ip)
            GRADIO_IP="$2"
            shift 2
            ;;
        --share)
            GRADIO_SHARE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --task-type TYPE        Task type: CustomVoice, VoiceDesign, Base (default: CustomVoice)"
            echo "  --server-port PORT      Port for vLLM server (default: 8000)"
            echo "  --gradio-port PORT      Port for Gradio demo (default: 7860)"
            echo "  --server-host HOST      Host for vLLM server (default: 0.0.0.0)"
            echo "  --gradio-ip IP          IP for Gradio demo (default: 127.0.0.1)"
            echo "  --share                 Share Gradio demo publicly"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Map task type to model
case "$TASK_TYPE" in
    CustomVoice)
        MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        ;;
    VoiceDesign)
        MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        ;;
    Base)
        MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        ;;
    *)
        echo "Unknown task type: $TASK_TYPE"
        echo "Supported: CustomVoice, VoiceDesign, Base"
        exit 1
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_BASE="http://localhost:${SERVER_PORT}"

echo "=========================================="
echo "Qwen3-TTS Gradio Demo"
echo "=========================================="
echo "Task Type : $TASK_TYPE"
echo "Model     : $MODEL"
echo "Server    : http://${SERVER_HOST}:${SERVER_PORT}"
echo "Gradio    : http://${GRADIO_IP}:${GRADIO_PORT}"
echo "=========================================="

# Cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down..."
    if [ -n "$SERVER_PID" ]; then
        echo "Stopping vLLM server (PID: $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    if [ -n "$GRADIO_PID" ]; then
        echo "Stopping Gradio demo (PID: $GRADIO_PID)..."
        kill "$GRADIO_PID" 2>/dev/null || true
        wait "$GRADIO_PID" 2>/dev/null || true
    fi
    echo "Cleanup complete"
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start vLLM server
echo ""
echo "Starting vLLM server..."
LOG_FILE="/tmp/vllm_tts_server_${SERVER_PORT}.log"

vllm-omni serve "$MODEL" \
    --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts.yaml \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --enforce-eager \
    --omni 2>&1 | tee "$LOG_FILE" &
SERVER_PID=$!

# Wait for server startup
echo ""
echo "Waiting for vLLM server to be ready..."
STARTUP_FLAG="/tmp/vllm_tts_startup_flag_${SERVER_PORT}.tmp"
rm -f "$STARTUP_FLAG"

(
    tail -f "$LOG_FILE" 2>/dev/null | grep -m 1 "Application startup complete" > /dev/null && touch "$STARTUP_FLAG"
) &
TAIL_PID=$!

MAX_WAIT=300
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if [ -f "$STARTUP_FLAG" ]; then
        kill "$TAIL_PID" 2>/dev/null || true
        wait "$TAIL_PID" 2>/dev/null || true
        echo ""
        echo "vLLM server is ready!"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$TAIL_PID" 2>/dev/null || true
        echo ""
        echo "Error: vLLM server failed to start"
        exit 1
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

rm -f "$STARTUP_FLAG"

if [ $ELAPSED -ge $MAX_WAIT ]; then
    kill "$TAIL_PID" 2>/dev/null || true
    echo "Error: Server startup timed out after ${MAX_WAIT}s"
    kill "$SERVER_PID" 2>/dev/null || true
    exit 1
fi

# Start Gradio demo
echo ""
echo "Starting Gradio demo..."
cd "$SCRIPT_DIR"
GRADIO_CMD=("python" "gradio_demo.py" "--api-base" "$API_BASE" "--ip" "$GRADIO_IP" "--port" "$GRADIO_PORT")
if [ "$GRADIO_SHARE" = true ]; then
    GRADIO_CMD+=("--share")
fi

"${GRADIO_CMD[@]}" &
GRADIO_PID=$!

echo ""
echo "=========================================="
echo "Both services are running!"
echo "=========================================="
echo "vLLM Server : http://${SERVER_HOST}:${SERVER_PORT}"
echo "Gradio Demo : http://${GRADIO_IP}:${GRADIO_PORT}"
echo ""
echo "Press Ctrl+C to stop both services"
echo "=========================================="
echo ""

wait $SERVER_PID $GRADIO_PID || true
cleanup
