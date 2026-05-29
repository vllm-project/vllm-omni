#!/usr/bin/env bash
# Unified curl example for the three Kimi-Audio task modes against
# vLLM-Omni's /v1/chat/completions.
#
# Usage:
#   TASK=audio2text  bash run_curl.sh           # ASR / audio QA
#   TASK=audio2audio bash run_curl.sh           # spoken response
#   TASK=text2audio  bash run_curl.sh           # TTS-style
#
# Server launch: vllm_omni/deploy/kimi_audio.yaml for all three; toggle the
# YAML's `async_chunk` flag for sub-second TTFB streaming.
set -euo pipefail

TASK="${TASK:-audio2text}"
PORT="${PORT:-8091}"
HOST="${HOST:-localhost}"
MARY_HAD_LAMB_AUDIO_URL="https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/mary_had_lamb.ogg"
# Default sample audio for the audio2text task. Originally taken from the
# MiniMax TTS-Multilingual test set (sample 10), mirrored to Google Drive
# for a stable link. Direct-download form of
# https://drive.google.com/file/d/1RHz6uUSbAR_N3Li1Bjh8dPknykw4IVio/view?usp=sharing.
AUDIO2TEXT_DEFAULT_URL="https://drive.google.com/uc?export=download&id=1RHz6uUSbAR_N3Li1Bjh8dPknykw4IVio"
OUT_FILE="${OUT_FILE:-response.wav}"

case "$TASK" in
  audio2text)
    question="${QUESTION:-Please transcribe the audio.}"
    audio_url="${AUDIO_URL:-$AUDIO2TEXT_DEFAULT_URL}"
    thinker_sampling_params='{
      "temperature": 0.0, "top_p": 1.0, "top_k": -1,
      "max_tokens": 512, "seed": 42, "repetition_penalty": 1.0
    }'
    request_body=$(cat <<EOF
{
  "model": "moonshotai/Kimi-Audio-7B-Instruct",
  "sampling_params_list": [$thinker_sampling_params],
  "messages": [
    {"role": "user", "content": [
      {"type": "audio_url", "audio_url": {"url": "$audio_url"}},
      {"type": "text", "text": "$question"}
    ]}
  ]
}
EOF
)
    curl -sS --retry 3 --retry-delay 3 --retry-connrefused \
        -X POST "http://${HOST}:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$request_body" | jq '.choices[0].message.content'
    ;;

  audio2audio|text2audio)
    thinker_sampling_params='{
      "temperature": 0.6, "top_p": 0.95, "top_k": 50,
      "max_tokens": 1024, "seed": 42, "repetition_penalty": 1.0
    }'
    code2wav_sampling_params='{
      "temperature": 0.0, "top_p": 1.0, "top_k": -1,
      "max_tokens": 8192, "seed": 42, "detokenize": false
    }'

    if [ "$TASK" = "audio2audio" ]; then
      question="${QUESTION:-Answer in audio. Briefly summarize what was said.}"
      user_content=$(cat <<EOF
[
  {"type": "audio_url", "audio_url": {"url": "$MARY_HAD_LAMB_AUDIO_URL"}},
  {"type": "text", "text": "$question"}
]
EOF
)
    else
      question="${QUESTION:-Please say the following in audio: \"Hello, my name is Kimi.\"}"
      user_content=$(cat <<EOF
[
  {"type": "text", "text": "$question"}
]
EOF
)
    fi

    request_body=$(cat <<EOF
{
  "model": "moonshotai/Kimi-Audio-7B-Instruct",
  "sampling_params_list": [$thinker_sampling_params, $code2wav_sampling_params],
  "modalities": ["text", "audio"],
  "messages": [
    {"role": "user", "content": $user_content}
  ]
}
EOF
)
    curl -sS --retry 3 --retry-delay 3 --retry-connrefused \
        -X POST "http://${HOST}:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$request_body" \
      | jq -r '.choices[0].message.audio.data' \
      | base64 -d > "$OUT_FILE"
    echo "Wrote $OUT_FILE"
    ;;

  *)
    echo "Unknown TASK: $TASK. Expected one of: audio2text, audio2audio, text2audio." >&2
    exit 1
    ;;
esac
