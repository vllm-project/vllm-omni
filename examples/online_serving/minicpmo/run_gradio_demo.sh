#!/bin/bash
# Launch the MiniCPM-o gradio demo against one or both backends.
#
# Prereq:
#   1. Start a vllm-omni OpenAI server for MiniCPM-o 4.5 on :8099 (see
#      start_minicpmo45_server.sh) and / or one for 2.6 on :8091.
#   2. This script picks them up by default; override via env vars below.
set -e

HERE="$(cd "$(dirname "$0")" && pwd)"

source /cache/caitianchi/install/miniconda3/etc/profile.d/conda.sh
conda activate vllm

: "${MINICPMO45_API_BASE:=http://localhost:8099/v1}"
: "${MINICPMO45_MODEL:=/cache/caitianchi/model/MiniCPM-o-4_5_full}"
: "${MINICPMO26_API_BASE:=}"
: "${MINICPMO26_MODEL:=}"
: "${GRADIO_HOST:=0.0.0.0}"
: "${GRADIO_PORT:=7862}"
# HTTPS (browsers require a secure context for microphone access).
# Set GRADIO_SSL_CERTFILE / GRADIO_SSL_KEYFILE to enable TLS.
: "${GRADIO_SSL_CERTFILE:=/cache/caitianchi/certs/gradio.crt}"
: "${GRADIO_SSL_KEYFILE:=/cache/caitianchi/certs/gradio.key}"

# If the 2.6 server from /cache/guiqingxin is up on :8091, pick it up automatically.
if [ -z "$MINICPMO26_API_BASE" ] && curl -sf --max-time 2 http://localhost:8091/health >/dev/null 2>&1; then
  MINICPMO26_API_BASE="http://localhost:8091/v1"
  MINICPMO26_MODEL="./MiniCPM-o-2_6"
  echo "Auto-detected MiniCPM-o-2.6 server at :8091"
fi

export MINICPMO45_API_BASE MINICPMO45_MODEL MINICPMO26_API_BASE MINICPMO26_MODEL

SSL_ARGS=()
if [ -n "$GRADIO_SSL_CERTFILE" ] && [ -n "$GRADIO_SSL_KEYFILE" ] \
   && [ -f "$GRADIO_SSL_CERTFILE" ] && [ -f "$GRADIO_SSL_KEYFILE" ]; then
  SSL_ARGS=(--ssl-certfile "$GRADIO_SSL_CERTFILE" --ssl-keyfile "$GRADIO_SSL_KEYFILE")
  echo "HTTPS enabled: cert=$GRADIO_SSL_CERTFILE key=$GRADIO_SSL_KEYFILE"
else
  echo "HTTPS disabled (cert/key not found). Microphone won't work on remote browsers."
fi

exec python "$HERE/gradio_demo.py" \
    --minicpmo45-api-base "$MINICPMO45_API_BASE" \
    --minicpmo45-model "$MINICPMO45_MODEL" \
    --minicpmo26-api-base "$MINICPMO26_API_BASE" \
    --minicpmo26-model "$MINICPMO26_MODEL" \
    --host "$GRADIO_HOST" \
    --port "$GRADIO_PORT" \
    "${SSL_ARGS[@]}"
