#!/bin/bash
# Launch the interaction orchestrator in front of a running model server.
set -euo pipefail

PORT="${PORT:-8070}"
MAIN_BACKEND_URL="${MAIN_BACKEND_URL:-http://127.0.0.1:8061/v1}"
MAIN_MODEL="${MAIN_MODEL:-JoyAI-VL-Interaction-Preview}"
PERSONA="${PERSONA:-default}"

# By default memory reuses the main model as its own summarizer. Set
# SUMMARIZER_BACKEND_URL/SUMMARIZER_MODEL to a dedicated Qwen3-VL-4B server,
# or pass --no-memory for the lightest setup.
EXTRA=()
[[ -n "${SUMMARIZER_BACKEND_URL:-}" ]] && EXTRA+=(--summarizer-backend-url "${SUMMARIZER_BACKEND_URL}")
[[ -n "${SUMMARIZER_MODEL:-}" ]] && EXTRA+=(--summarizer-model "${SUMMARIZER_MODEL}")
[[ "${NO_MEMORY:-0}" == "1" ]] && EXTRA+=(--no-memory)

python -m vllm_omni.experimental.fullduplex.joyvl.serving.server \
    --port "${PORT}" \
    --main-backend-url "${MAIN_BACKEND_URL}" \
    --main-model "${MAIN_MODEL}" \
    --persona "${PERSONA}" \
    "${EXTRA[@]}"
