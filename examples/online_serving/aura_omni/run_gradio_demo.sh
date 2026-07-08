#!/usr/bin/env bash
set -euo pipefail

MODEL="aurateam/AURA"
SERVER_MODEL="aurateam/AURA"
DEPLOY_CONFIG="/data/yrr/vllm-omni/vllm_omni/deploy/aura_omni.yaml"
SERVER_PORT=8091
GRADIO_PORT=7862
SERVER_HOST="0.0.0.0"
GRADIO_IP="127.0.0.1"
GRADIO_SHARE=false


while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --server-model) SERVER_MODEL="$2"; shift 2 ;;
    --deploy-config) DEPLOY_CONFIG="$2"; shift 2 ;;
    --server-port) SERVER_PORT="$2"; shift 2 ;;
    --gradio-port) GRADIO_PORT="$2"; shift 2 ;;
    --server-host) SERVER_HOST="$2"; shift 2 ;;
    --gradio-ip) GRADIO_IP="$2"; shift 2 ;;
    --share) GRADIO_SHARE=true; shift ;;
    --help)
      echo "Usage: $0 [--model SERVED_MODEL_NAME] [--server-model MODEL_PATH] [--deploy-config YAML] [--server-port PORT] [--gradio-port PORT] [--share]"
      exit 0
      ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_BASE="http://localhost:${SERVER_PORT}/v1"
LOG_FILE="/tmp/aura_omni_vllm_${SERVER_PORT}.log"

cleanup() {
  echo "Shutting down..."
  [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null || true
  [[ -n "${GRADIO_PID:-}" ]] && kill "$GRADIO_PID" 2>/dev/null || true
}
trap cleanup SIGINT SIGTERM EXIT

vllm serve "$SERVER_MODEL" \
  --omni \
  --host "$SERVER_HOST" \
  --port "$SERVER_PORT" \
  --deploy-config "$DEPLOY_CONFIG" \
  --served-model-name "$MODEL" \
  --trust-remote-code 2>&1 | tee "$LOG_FILE" &
SERVER_PID=$!

echo "Waiting for server startup..."
for _ in $(seq 1 600); do
  if grep -q "Application startup complete" "$LOG_FILE" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vLLM server exited before startup completed"
    wait "$SERVER_PID" || true
    exit 1
  fi
  sleep 1
done

# cd "$SCRIPT_DIR"
GRADIO_CMD=(python gradio_demo.py --model "$MODEL" --api-base "$API_BASE" --ip "$GRADIO_IP" --port "$GRADIO_PORT")
if [[ "$GRADIO_SHARE" == "true" ]]; then
  GRADIO_CMD+=(--share)
fi
"${GRADIO_CMD[@]}" &
GRADIO_PID=$!

echo "vLLM server: http://${SERVER_HOST}:${SERVER_PORT}"
echo "Gradio demo: http://${GRADIO_IP}:${GRADIO_PORT}"
if [[ -n "${SERVER_PID:-}" ]]; then
  wait "$SERVER_PID" "$GRADIO_PID"
else
  wait "$GRADIO_PID"
fi
