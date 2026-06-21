#!/bin/bash
# Launch the MiniCPM-o gradio demo against one or both backends.
#
# Prereq:
#   Start vllm-omni OpenAI servers:
#     - MiniCPM-o 4.5 on :8099 (stage config: minicpmo45_8x4090.yaml etc.)
#     - MiniCPM-o 2.6 on :8091 (stage config: minicpmo_2_6_8x4090.yaml etc.)
#   The script auto-detects any reachable backend.
set -e

HERE="$(cd "$(dirname "$0")" && pwd)"

: "${MINICPMO45_API_BASE:=}"
: "${MINICPMO45_MODEL:=openbmb/MiniCPM-o-4_5}"
: "${MINICPMO26_API_BASE:=}"
: "${MINICPMO26_MODEL:=}"
: "${GRADIO_HOST:=0.0.0.0}"
: "${GRADIO_PORT:=7862}"
# HTTPS (browsers require a secure context for microphone access).
: "${GRADIO_SSL_CERTFILE:=}"
: "${GRADIO_SSL_KEYFILE:=}"

# Auto-detect 4.5 server if it's running on :8099 and no explicit config given.
if [ -z "$MINICPMO45_API_BASE" ] && curl -sf --max-time 2 http://localhost:8099/health >/dev/null 2>&1; then
  MINICPMO45_API_BASE="http://localhost:8099/v1"
  echo "Auto-detected MiniCPM-o-4.5 server at :8099"
fi

# Auto-detect 2.6 server if it's running on :8091 and no explicit config given.
if [ -z "$MINICPMO26_API_BASE" ] && curl -sf --max-time 2 http://localhost:8091/health >/dev/null 2>&1; then
  MINICPMO26_API_BASE="http://localhost:8091/v1"
  MINICPMO26_MODEL="${MINICPMO26_MODEL:-./MiniCPM-o-2_6}"
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
