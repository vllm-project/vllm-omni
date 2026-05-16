#!/bin/bash
# Launch the MiniCPM-o 4.5 gradio demo.
#
# Prereq:
#   Start a vllm-omni OpenAI server for MiniCPM-o 4.5 on :8099 (see the
#   8x4090 stage config under vllm_omni/model_executor/stage_configs).
set -e

HERE="$(cd "$(dirname "$0")" && pwd)"

: "${MINICPMO45_API_BASE:=http://localhost:8099/v1}"
: "${MINICPMO45_MODEL:=openbmb/MiniCPM-o-4_5}"
: "${GRADIO_HOST:=0.0.0.0}"
: "${GRADIO_PORT:=7862}"
# HTTPS (browsers require a secure context for microphone access).
# Set GRADIO_SSL_CERTFILE / GRADIO_SSL_KEYFILE to enable TLS.
: "${GRADIO_SSL_CERTFILE:=}"
: "${GRADIO_SSL_KEYFILE:=}"

export MINICPMO45_API_BASE MINICPMO45_MODEL

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
    --host "$GRADIO_HOST" \
    --port "$GRADIO_PORT" \
    "${SSL_ARGS[@]}"
