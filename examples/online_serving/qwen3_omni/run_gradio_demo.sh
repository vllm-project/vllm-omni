#!/bin/bash
# Convenience script to launch both vLLM server and Gradio demo for Qwen3-Omni
#
# Usage:
#   ./run_gradio_demo.sh [OPTIONS]
#
# Example:
#   ./run_gradio_demo.sh --model Qwen/Qwen3-Omni-30B-A3B-Instruct --server-port 8091 --gradio-port 7861

set -e

# Default values
MODEL="Qwen/Qwen3-Omni-30B-A3B-Instruct"
SERVER_PORT=8091
GRADIO_PORT=7861
STAGE_CONFIGS_PATH=""
SERVER_HOST="0.0.0.0"
GRADIO_IP="127.0.0.1"
GRADIO_SHARE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
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
        --stage-configs-path)
            STAGE_CONFIGS_PATH="$2"
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
            echo "  --model MODEL                 Model name/path (default: Qwen/Qwen3-Omni-30B-A3B-Instruct)"
            echo "  --server-port PORT            Port for vLLM server (default: 8091)"
            echo "  --gradio-port PORT            Port for Gradio demo (default: 7861)"
            echo "  --stage-configs-path PATH     Path to custom stage configs YAML file (optional)"
            echo "  --server-host HOST            Host for vLLM server (default: 0.0.0.0)"
            echo "  --gradio-ip IP                IP for Gradio demo (default: 127.0.0.1)"
            echo "  --share                       Share Gradio demo publicly"
            echo "  --help                        Show this help message"
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

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_BASE="http://localhost:${SERVER_PORT}/v1"
HEALTH_URL="http://localhost:${SERVER_PORT}/health"

echo "=========================================="
echo "Starting vLLM-Omni Gradio Demo"
echo "=========================================="
echo "Model: $MODEL"
echo "Server: http://${SERVER_HOST}:${SERVER_PORT}"
echo "Gradio: http://${GRADIO_IP}:${GRADIO_PORT}"
echo "=========================================="

# Build vLLM server command
SERVER_CMD=("vllm" "serve" "$MODEL" "--omni" "--port" "$SERVER_PORT" "--host" "$SERVER_HOST")
if [ -n "$STAGE_CONFIGS_PATH" ]; then
    SERVER_CMD+=("--stage-configs-path" "$STAGE_CONFIGS_PATH")
fi

# Function to cleanup on exit
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

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start vLLM server with output shown in real-time and saved to log
echo ""
echo "Starting vLLM server..."
LOG_FILE="/tmp/vllm_server_${SERVER_PORT}.log"
"${SERVER_CMD[@]}" 2>&1 | tee "$LOG_FILE" &
SERVER_PID=$!

# Start a background process to monitor the log for startup completion
STARTUP_COMPLETE=false
TAIL_PID=""

# Function to cleanup tail process
cleanup_tail() {
    if [ -n "$TAIL_PID" ]; then
        kill "$TAIL_PID" 2>/dev/null || true
        wait "$TAIL_PID" 2>/dev/null || true
    fi
}

# Wait for server to be ready by checking log output
echo ""
echo "Waiting for vLLM server to be ready (checking for 'Application startup complete' message)..."
echo ""

# Monitor log file for startup completion message
MAX_WAIT=300  # 5 minutes timeout as fallback
ELAPSED=0

# Use a temporary file to track startup completion
STARTUP_FLAG="/tmp/vllm_startup_flag_${SERVER_PORT}.tmp"
rm -f "$STARTUP_FLAG"

# Start monitoring in background
(
    tail -f "$LOG_FILE" 2>/dev/null | grep -m 1 "Application startup complete" > /dev/null && touch "$STARTUP_FLAG"
) &
TAIL_PID=$!

while [ $ELAPSED -lt $MAX_WAIT ]; do
    # Check if startup flag file exists (startup complete)
    if [ -f "$STARTUP_FLAG" ]; then
        cleanup_tail
        echo ""
        echo "✓ vLLM server is ready!"
        STARTUP_COMPLETE=true
        break
    fi

    # Check if server process is still running
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        cleanup_tail
        echo ""
        echo "Error: vLLM server failed to start (process terminated)"
        wait "$SERVER_PID" 2>/dev/null || true
        exit 1
    fi

    sleep 1
    ELAPSED=$((ELAPSED + 1))
done

cleanup_tail
rm -f "$STARTUP_FLAG"

if [ "$STARTUP_COMPLETE" != "true" ]; then
    echo ""
    echo "Error: vLLM server did not complete startup within ${MAX_WAIT} seconds"
    kill "$SERVER_PID" 2>/dev/null || true
    exit 1
fi

# Start Gradio demo
echo ""
echo "Starting Gradio demo..."
cd "$SCRIPT_DIR"
GRADIO_CMD=("python" "gradio_demo.py" "--model" "$MODEL" "--api-base" "$API_BASE" "--ip" "$GRADIO_IP" "--port" "$GRADIO_PORT")
if [ "$GRADIO_SHARE" = true ]; then
    GRADIO_CMD+=("--share")
fi

"${GRADIO_CMD[@]}" > /tmp/gradio_demo.log 2>&1 &
GRADIO_PID=$!

echo ""
echo "=========================================="
echo "Both services are running!"
echo "=========================================="
echo "vLLM Server: http://${SERVER_HOST}:${SERVER_PORT}"
echo "Gradio Demo: http://${GRADIO_IP}:${GRADIO_PORT}"
echo ""
echo "Press Ctrl+C to stop both services"
echo "=========================================="
echo ""

# Wait for either process to exit
wait $SERVER_PID $GRADIO_PID || true

cleanup
