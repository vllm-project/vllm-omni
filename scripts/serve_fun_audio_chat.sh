#!/usr/bin/env bash
# Start vllm-omni server for Fun-Audio-Chat-8B S2S via `uv run`.
#
# The env is defined by pyproject.toml + uv.lock. No need to pre-activate a
# .venv; `uv run --extra fun-audio-chat-ref` resolves the right interpreter
# and deps from the lockfile.
#
# Usage:
#   ./scripts/serve_fun_audio_chat.sh
#
# Overridable env:
#   PORT                 (default 8091)
#   HOST                 (default 0.0.0.0)
#   FUN_AUDIO_CKPT       (default /home/jovyan/ye/vllm-omni/pretrained_models/Fun-Audio-Chat-8B)
#   FUN_AUDIO_COSYVOICE_PATH  (default src/funaudiochat/third_party/CosyVoice)
#   FUN_AUDIO_VOCODER_PATH    (default pretrained_models/Fun-CosyVoice3-0.5B-2512)
#   STAGE_CONFIG         (default vllm_omni/deploy/funaudiochat.yaml)
#   EXTRA_ARGS           (extra args passed verbatim to `vllm-omni serve`)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# ─── config ───────────────────────────────────────────────────────────────────
: "${PORT:=8091}"
: "${HOST:=0.0.0.0}"
: "${FUN_AUDIO_CKPT:=/home/jovyan/ye/vllm-omni/pretrained_models/Fun-Audio-Chat-8B}"
: "${FUN_AUDIO_COSYVOICE_PATH:=$REPO/src/funaudiochat/third_party/CosyVoice}"
: "${FUN_AUDIO_VOCODER_PATH:=/home/jovyan/ye/vllm-omni/pretrained_models/Fun-CosyVoice3-0.5B-2512}"
: "${STAGE_CONFIG:=vllm_omni/deploy/funaudiochat.yaml}"
: "${EXTRA_ARGS:=}"

export FUN_AUDIO_COSYVOICE_PATH FUN_AUDIO_VOCODER_PATH
export FUN_AUDIO_REF_PATH="$REPO/src/funaudiochat"
# Reference CosyVoice + Matcha-TTS are source-only (no PyPI install); extend
# PYTHONPATH so Stage-1 token2wav can import them at runtime.
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$FUN_AUDIO_COSYVOICE_PATH:$FUN_AUDIO_COSYVOICE_PATH/third_party/Matcha-TTS:$FUN_AUDIO_REF_PATH"

# ─── sanity ───────────────────────────────────────────────────────────────────
[[ -d "$FUN_AUDIO_CKPT" ]] || { echo "missing model dir: $FUN_AUDIO_CKPT" >&2; exit 1; }
[[ -d "$FUN_AUDIO_VOCODER_PATH" ]] || { echo "missing vocoder dir: $FUN_AUDIO_VOCODER_PATH" >&2; exit 1; }
[[ -d "$FUN_AUDIO_COSYVOICE_PATH" ]] || { echo "missing CosyVoice submodule: $FUN_AUDIO_COSYVOICE_PATH" >&2; exit 1; }
[[ -f "$STAGE_CONFIG" ]] || { echo "missing stage config: $STAGE_CONFIG" >&2; exit 1; }

cat <<INFO
================================================================================
vllm-omni Fun-Audio-Chat-8B S2S server (uv-managed env)
  model        : $FUN_AUDIO_CKPT
  stage config : $STAGE_CONFIG
  vocoder      : $FUN_AUDIO_VOCODER_PATH
  cosyvoice    : $FUN_AUDIO_COSYVOICE_PATH
  listening on : http://$HOST:$PORT
================================================================================
INFO

# ─── serve ────────────────────────────────────────────────────────────────────
exec uv run \
    --extra fun-audio-chat-ref \
    --index-strategy unsafe-best-match \
    --frozen \
    -- \
    vllm-omni serve "$FUN_AUDIO_CKPT" \
        --stage-configs-path "$STAGE_CONFIG" \
        --host "$HOST" \
        --port "$PORT" \
        --trust-remote-code \
        --enforce-eager \
        --omni \
        --media-io-kwargs '{"audio": {"target_sr": 16000}}' \
        $EXTRA_ARGS
