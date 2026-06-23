#!/bin/bash
# One-shot JoyVL demo: model -> interaction orchestrator -> JD webui.
#
# The model is the only fixed component; every external module is pluggable and
# configured here via env (unset = disabled), matching JoyVL's design:
#
#   TTS_URL                       voice out  (point at our tts_bridge, or any OpenAI-Realtime-style TTS WS)
#   ASR_URL                       voice in   (point at our asr_bridge, or any compatible ASR WS)
#   DELEGATION_BACKEND_URL        orchestrator background brain for </delegation> (OpenAI-compatible; DELEGATION_MODEL/DELEGATION_API_KEY pick model+key)
#                                 DELEGATION_KIND=chat (default) text/VL brain answers; =image a text-to-image model generates a picture
#                                 chat+Claude: DELEGATION_BACKEND_URL=https://api.anthropic.com/v1/ DELEGATION_MODEL=claude-... DELEGATION_API_KEY=sk-ant-...
#                                 image+Qwen-Image: DELEGATION_KIND=image DELEGATION_BACKEND_URL=http://127.0.0.1:8091/v1
#   BACKGROUND_CODEX_API_URL      webui-side background agent
#   ENABLE_MEMORY (default 1)     3-tier memory; SUMMARIZER_BACKEND_URL/MODEL pick the summarizer (default: reuse main)
#
# Usage:  bash start_all.sh
#         TTS_URL=ws://host/v1/tts ASR_URL=ws://host/v1/asr GPU=0 bash start_all.sh
set -o pipefail

CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
VENV="${VENV:-$CODE/.venv}"
WEBUI_DIR="${WEBUI_DIR:-$CODE/joyvl-interaction/joyvl-interaction-webui}"
MODEL="${MODEL:-jdopensource/JoyAI-VL-Interaction-Preview}"
SERVED_NAME="${SERVED_NAME:-JoyAI-VL-Interaction-Preview}"
GPU="${GPU:-0}"
MODEL_PORT="${MODEL_PORT:-8061}"
ORCH_PORT="${ORCH_PORT:-8070}"
WEBUI_PORT="${WEBUI_PORT:-8999}"
WEBUI_USERNAME="${WEBUI_USERNAME:-admin}"
WEBUI_PASSWORD="${WEBUI_PASSWORD:-changeme}"   # override via env for any exposed deployment
ENABLE_MEMORY="${ENABLE_MEMORY:-1}"
RESPONSE_DEDUP="${RESPONSE_DEDUP:-0.85}"   # demo drops near-duplicate narration; set 1.0 for reference (exact-only)
TTS_URL="${TTS_URL:-}"
ASR_URL="${ASR_URL:-}"
BACKGROUND_CODEX_API_URL="${BACKGROUND_CODEX_API_URL:-}"

LOG=/tmp/joyvl_demo
mkdir -p "$LOG"

[ -d "$WEBUI_DIR" ] || { echo "! WEBUI_DIR not found: $WEBUI_DIR (set WEBUI_DIR=...)"; exit 1; }

kill_port() {
  local p
  p=$(ss -tlnp 2>/dev/null | grep ":$1 " | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
  [ -n "${p:-}" ] && kill "$p" 2>/dev/null
  return 0
}
wait_http() {
  for _ in $(seq 1 150); do
    curl -sk -m2 "$1" -o /dev/null 2>/dev/null && return 0
    sleep 2
  done
  echo "  ! timeout waiting for $1"
  return 1
}

echo "Resetting ports ${MODEL_PORT} ${ORCH_PORT} ${WEBUI_PORT}…"
for p in "$MODEL_PORT" "$ORCH_PORT" "$WEBUI_PORT"; do kill_port "$p"; done
sleep 3

[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"

# detach into a new session so servers outlive this launcher
spawn() { local log="$1"; shift; setsid "$@" > "$log" 2>&1 < /dev/null & }

echo "[1/3] model on GPU ${GPU}…"
spawn "$LOG/model.log" env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU" \
  vllm serve "$MODEL" --served-model-name "$SERVED_NAME" --port "$MODEL_PORT" \
  --gpu-memory-utilization 0.85 --max-model-len 131072 --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":256,"video":1}'
wait_http "http://127.0.0.1:${MODEL_PORT}/health" && echo "  up :${MODEL_PORT}"

echo "[2/3] orchestrator (memory=${ENABLE_MEMORY})…"
ORCH_ARGS=(--port "$ORCH_PORT" --main-backend-url "http://127.0.0.1:${MODEL_PORT}/v1"
           --main-model "$SERVED_NAME"
           --response-dedup-threshold "$RESPONSE_DEDUP")
# delegation: point at a stronger background brain to enable it; otherwise keep it off
if [ -n "${DELEGATION_BACKEND_URL:-}" ]; then
  ORCH_ARGS+=(--delegation-backend-url "$DELEGATION_BACKEND_URL")
  ORCH_ARGS+=(--delegation-kind "${DELEGATION_KIND:-chat}")
  [ -n "${DELEGATION_MODEL:-}" ] && ORCH_ARGS+=(--delegation-model "$DELEGATION_MODEL")
  [ -n "${DELEGATION_API_KEY:-}" ] && ORCH_ARGS+=(--delegation-api-key "$DELEGATION_API_KEY")
else
  ORCH_ARGS+=(--no-delegation)
fi
[ "$ENABLE_MEMORY" = "0" ] && ORCH_ARGS+=(--no-memory)
[ -n "${SUMMARIZER_BACKEND_URL:-}" ] && ORCH_ARGS+=(--summarizer-backend-url "$SUMMARIZER_BACKEND_URL")
[ -n "${SUMMARIZER_MODEL:-}" ] && ORCH_ARGS+=(--summarizer-model "$SUMMARIZER_MODEL")
spawn "$LOG/orch.log" env PYTHONPATH="$CODE" python -m vllm_omni.experimental.fullduplex.joyvl.serving.server "${ORCH_ARGS[@]}"
wait_http "http://127.0.0.1:${ORCH_PORT}/health" && echo "  up :${ORCH_PORT}"

echo "[3/3] JD webui…"
spawn "$LOG/webui.log" env WEBUI_USERNAME="$WEBUI_USERNAME" WEBUI_PASSWORD="$WEBUI_PASSWORD" \
  TTS_URL="$TTS_URL" ASR_URL="$ASR_URL" BACKGROUND_CODEX_API_URL="$BACKGROUND_CODEX_API_URL" \
  bash -c "cd '$WEBUI_DIR' && exec bash scripts/start_server.sh --api-base 'http://127.0.0.1:${ORCH_PORT}/v1'"
webui_up=0
for _ in $(seq 1 60); do grep -q "Server startup complete" "$LOG/webui.log" 2>/dev/null && { webui_up=1; break; }; sleep 1; done
[ "$webui_up" = 1 ] || { echo "! webui did not start — see $LOG/webui.log"; exit 1; }

IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo
echo "=== JoyVL demo ready ==="
echo "  webui  : https://${IP}:${WEBUI_PORT}   (user ${WEBUI_USERNAME} / pass ${WEBUI_PASSWORD})"
echo "  memory : $([ "$ENABLE_MEMORY" = 0 ] && echo off || echo on)"
echo "  TTS    : ${TTS_URL:-<unset>}"
echo "  ASR    : ${ASR_URL:-<unset>}"
echo "  agent  : ${BACKGROUND_CODEX_API_URL:-<unset>}"
echo "  logs   : ${LOG}/{model,orch,webui}.log"
