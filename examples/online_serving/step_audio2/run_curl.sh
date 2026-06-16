#!/usr/bin/env bash
set -euo pipefail

# Step-Audio2 curl client for online serving
# Usage: bash run_curl.sh [audio_to_text|text_to_audio|audio_to_audio]

QUERY_TYPE="${1:-audio_to_text}"
API_BASE="${API_BASE:-http://localhost:8092}"

# Validate query type
if [[ ! "$QUERY_TYPE" =~ ^(audio_to_text|text_to_audio|audio_to_audio)$ ]]; then
    echo "Error: Invalid query type '$QUERY_TYPE'"
    echo "Usage: $0 [audio_to_text|text_to_audio|audio_to_audio]"
    echo "  audio_to_text: Speech recognition (ASR)"
    echo "  text_to_audio: Text-to-speech (TTS)"
    echo "  audio_to_audio: Voice conversion"
    exit 1
fi

SEED=42

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
    system_content='[{"type": "text", "text": "You are a speech recognition assistant. Transcribe the audio accurately."}]'
    user_content='[
      {"type": "audio_url", "audio_url": {"url": "'"$MARY_HAD_LAMB_URL"'"}},
      {"type": "text", "text": "Please transcribe this audio."}
    ]'
    # Add stop token for ASR
    thinker_sampling_params='{
      "temperature": 0.7,
      "top_p": 0.9,
      "top_k": -1,
      "max_tokens": 1024,
      "seed": 42,
      "detokenize": true,
      "repetition_penalty": 1.05,
      "stop_token_ids": [151645]
    }'
    ;;
  text_to_audio)
    system_content='[{"type": "text", "text": "You are a text-to-speech assistant. Read the text aloud exactly as provided."}]'
    user_content='[
      {"type": "text", "text": "Hello, this is a test of Step Audio 2 text to speech synthesis.<tts_start>"}
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
      {"type": "text", "text": "Please listen to this audio and repeat its content.<tts_start>"}
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
  "model": "stepfun-ai/Step-Audio2-mini",
  "sampling_params_list": $sampling_params_list,
  "messages": [
    {
      "role": "system",
      "content": $system_content
    },
    {
      "role": "user",
      "content": $user_content
    }
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
