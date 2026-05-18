#!/bin/bash
# Wrapper: launch fresh r12 server, hit it with ONE prompt, save WAV, kill server.
# Workaround for the per-request _audio_state leak that breaks the 2nd+ requests
# within the same server lifetime. R13 will fix the proper way.
set -u
PROMPT="$1"
OUT_PATH="$2"
MAX_TOKENS="${3:-200}"
ROOT="/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/tts/vllm-omni"
cd "$ROOT"
LOG=".humanize/rlcr/2026-05-15_04-34-42/logs/gen_$(date +%H%M%S).log"
touch "$LOG"

# Kill any leftover server
pids=$(ps -ef | /bin/grep -E "vllm-omni|StageEngine" | /bin/grep -v grep | awk '{print $2}')
[ -n "$pids" ] && kill -9 $pids 2>/dev/null
sleep 3

# Launch server in background, detached
setsid env PATH="$ROOT/.venv/bin:$PATH" \
  CUDA_VISIBLE_DEVICES=6,7 \
  VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 \
  HF_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/.cache/huggingface \
  vllm-omni serve "bosonai/higgs-audio-v2-generation-3B-base" \
    --deploy-config vllm_omni/deploy/higgs_audio_v2.yaml \
    --attention-backend FLEX_ATTENTION \
    --host 0.0.0.0 --port 8094 --gpu-memory-utilization 0.4 \
    --trust-remote-code --omni < /dev/null > "$LOG" 2>&1 &
SVPID=$!
disown $SVPID 2>/dev/null

# Wait for startup
until /bin/grep -q "Application startup complete" "$LOG"; do sleep 4; done

# One request
OUT_DIR=$(dirname "$OUT_PATH")
OUT_NAME=$(basename "$OUT_PATH" .wav)
TMP_DIR=$(mktemp -d)
.venv/bin/python examples/online_serving/text_to_speech/higgs_audio_v2/batch_speech_client.py \
    --base-url http://localhost:8094 \
    --model bosonai/higgs-audio-v2-generation-3B-base \
    --output-dir "$TMP_DIR" \
    --prompts "$PROMPT" \
    --max-new-tokens "$MAX_TOKENS" \
    --timeout-s 180 > /dev/null 2>&1

# Find the produced WAV and copy to OUT_PATH
SRC=$(/bin/ls "$TMP_DIR"/*.wav | head -1)
[ -n "$SRC" ] && cp "$SRC" "$OUT_PATH" && /bin/ls -la "$OUT_PATH"

# Cleanup
rm -rf "$TMP_DIR"
pids=$(ps -ef | /bin/grep -E "vllm-omni|StageEngine" | /bin/grep -v grep | awk '{print $2}')
[ -n "$pids" ] && kill -9 $pids 2>/dev/null
sleep 1
