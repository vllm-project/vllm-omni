#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

set -euo pipefail

# Step-Audio2 curl client for online serving
# Usage: bash run_curl.sh [audio_to_text|text_to_audio|audio_to_audio]

QUERY_TYPE="${1:-audio_to_text}"
API_BASE="${API_BASE:-http://localhost:8092}"
MODALITIES=null

# Validate query type
if [[ ! "$QUERY_TYPE" =~ ^(audio_to_text|text_to_audio|audio_to_audio)$ ]]; then
    echo "Error: Invalid query type '$QUERY_TYPE'"
    echo "Usage: $0 [audio_to_text|text_to_audio|audio_to_audio]"
    echo "  audio_to_text: Speech recognition (ASR)"
    echo "  text_to_audio: Text-to-speech (TTS)"
    echo "  audio_to_audio: Voice conversion"
    exit 1
fi

# Default test audio URL
MARY_HAD_LAMB_URL="https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/mary_had_lamb.ogg"

# Sampling parameters for Thinker stage
thinker_sampling_params='{
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": -1,
  "max_tokens": 1024,
  "seed": 42,
  "detokenize": true,
  "repetition_penalty": 1.05
}'

# Sampling parameters for Token2Wav stage
token2wav_sampling_params='{
  "temperature": 0.0,
  "top_p": 1.0,
  "top_k": -1,
  "max_tokens": 1,
  "seed": 42,
  "detokenize": false
}'

# Build request based on query type
case "$QUERY_TYPE" in
  audio_to_text)
    MODALITIES='["text"]'
    system_content='[{"type": "text", "text": "请记录下你所听到的语音内容。"}]'
    user_content='[
      {"type": "audio_url", "audio_url": {"url": "'"$MARY_HAD_LAMB_URL"'"}}
    ]'
    # Use the model EOT EOS; no additional legacy stop token.
    thinker_sampling_params='{
      "temperature": 0.7,
      "top_p": 0.9,
      "top_k": -1,
      "max_tokens": 1024,
      "seed": 42,
      "detokenize": true,
      "repetition_penalty": 1.05
    }'
    ;;
  text_to_audio)
    system_content='[{"type": "text", "text": "以自然的语速读出下面的文字。"}]'
    user_content='[
      {"type": "text", "text": "Hello, this is a test of Step Audio 2 text to speech synthesis."}
    ]'
    thinker_sampling_params='{
      "temperature": 0.7,
      "top_p": 0.9,
      "top_k": -1,
      "max_tokens": 1024,
      "seed": 42,
      "detokenize": true,
      "repetition_penalty": 1.1
    }'
    ;;
  audio_to_audio)
    system_content='[{"type": "text", "text": "You are an audio processing assistant. Listen and repeat the audio content."}]'
    user_content='[
      {"type": "audio_url", "audio_url": {"url": "'"$MARY_HAD_LAMB_URL"'"}},
      {"type": "text", "text": "Please listen to this audio and repeat its content."}
    ]'
    thinker_sampling_params='{
      "temperature": 0.7,
      "top_p": 0.9,
      "top_k": -1,
      "max_tokens": 1024,
      "seed": 42,
      "detokenize": true,
      "repetition_penalty": 1.1
    }'
    ;;
esac

# Continue the assistant's speech prefix, matching the Python client.
assistant_message=''
generation_options=''
if [[ "$QUERY_TYPE" != "audio_to_text" ]]; then
  assistant_message=', {"role": "assistant", "content": "<tts_start>"}'
  generation_options='"continue_final_message": true, "add_generation_prompt": false,'
fi

sampling_params_list='[
  '"$thinker_sampling_params"',
  '"$token2wav_sampling_params"'
]'

echo "Query type: $QUERY_TYPE"
echo "API base: $API_BASE"
echo "Sending request..."
echo ""

output=$(curl -sS -X POST "${API_BASE}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d @- <<EOF
{
  "model": "stepfun-ai/Step-Audio-2-mini",
  "modalities": $MODALITIES,
  "sampling_params_list": $sampling_params_list,
  $generation_options
  "messages": [
    {
      "role": "system",
      "content": $system_content
    },
    {
      "role": "user",
      "content": $user_content
    }$assistant_message
  ]
}
EOF
)

# Extract and display text content
text_content=$(echo "$output" | jq -r '.choices[0].message.content // empty')
if [[ -n "$text_content" ]]; then
    echo "Text output: $text_content"
fi

# Check for audio content
audio_data=$(echo "$output" | jq -r '.choices[0].message.audio.data // empty')
if [[ -n "$audio_data" ]]; then
    echo "Audio output received (base64 encoded)"
    echo "To save audio, use the Python client or decode the base64 data"
fi

# Check for errors
error=$(echo "$output" | jq -r '.error // empty')
if [[ -n "$error" ]]; then
    echo "Error: $error"
fi

echo ""
echo "Done!"
