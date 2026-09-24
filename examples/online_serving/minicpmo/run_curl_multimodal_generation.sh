#!/usr/bin/env bash
set -euo pipefail

# Default query type
QUERY_TYPE="${1:-text}"

# Default modalities argument (JSON). Use null for server default (text+audio).
MODALITIES="${2:-null}"

# Validate query type
if [[ ! "$QUERY_TYPE" =~ ^(text|use_audio|use_image|use_video)$ ]]; then
    echo "Error: Invalid query type '$QUERY_TYPE'"
    echo "Usage: $0 [text|use_audio|use_image|use_video] [modalities]"
    echo "  text: Text query"
    echo "  use_audio: Audio + text query"
    echo "  use_image: Image + text query"
    echo "  use_video: Video + text query"
    echo "  modalities: JSON list or null (default: null → text+audio)"
    echo "    examples: '[\"text\"]'  '[\"text\",\"audio\"]'"
    exit 1
fi

HOST="${HOST:-localhost}"
PORT="${PORT:-8099}"
MODEL="${MODEL:-openbmb/MiniCPM-o-4_5}"

thinker_sampling_params='{
  "temperature": 0.0,
  "top_p": 1.0,
  "top_k": -1,
  "max_tokens": 2048,
  "seed": 42,
  "detokenize": true,
  "repetition_penalty": 1.1
}'

talker_sampling_params='{
  "temperature": 0.0,
  "top_p": 1.0,
  "top_k": -1,
  "max_tokens": 1,
  "seed": 42,
  "detokenize": false
}'
# Above is optional; defaults live in vllm_omni/deploy/minicpmo_4_5.yaml.

MARY_HAD_LAMB_AUDIO_URL="https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/mary_had_lamb.ogg"
CHERRY_BLOSSOM_IMAGE_URL="https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg"
SAMPLE_VIDEO_URL="https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4"

case "$QUERY_TYPE" in
  text)
    user_content='[
      {
        "type": "text",
        "text": "Say hello, then introduce vLLM-Omni in one sentence."
      }
    ]'
    ;;
  use_audio)
    user_content='[
        {
          "type": "audio_url",
          "audio_url": {
            "url": "'"$MARY_HAD_LAMB_AUDIO_URL"'"
          }
        },
        {
          "type": "text",
          "text": "What is the content of this audio?"
        }
      ]'
    ;;
  use_image)
    user_content='[
        {
          "type": "image_url",
          "image_url": {
            "url": "'"$CHERRY_BLOSSOM_IMAGE_URL"'"
          }
        },
        {
          "type": "text",
          "text": "What is the content of this image?"
        }
      ]'
    ;;
  use_video)
    user_content='[
        {
          "type": "video_url",
          "video_url": {
            "url": "'"$SAMPLE_VIDEO_URL"'"
          }
        },
        {
          "type": "text",
          "text": "Why is this video funny?"
        }
      ]'
    ;;
esac

sampling_params_list='[
  '"$thinker_sampling_params"',
  '"$talker_sampling_params"'
]'

# TTS speech path needs use_tts_template at the request root (curl does not
# flatten nested extra_body). Skip it when the caller asked for text-only.
USE_TTS_TEMPLATE=true
if [[ "$MODALITIES" == '["text"]' || "$MODALITIES" == "['text']" ]]; then
  USE_TTS_TEMPLATE=false
fi

echo "Running query type: $QUERY_TYPE (host=$HOST port=$PORT model=$MODEL)"
echo "modalities=$MODALITIES  use_tts_template=$USE_TTS_TEMPLATE"
echo ""

request_body=$(cat <<EOF
{
  "model": "$MODEL",
  "sampling_params_list": $sampling_params_list,
  "modalities": $MODALITIES,
  "chat_template_kwargs": {"use_tts_template": $USE_TTS_TEMPLATE},
  "messages": [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "You are MiniCPM-o, a helpful multimodal assistant that can understand images, audio and video, and respond in text and speech."
        }
      ]
    },
    {
      "role": "user",
      "content": $user_content
    }
  ]
}
EOF
)

output=$(curl -sS --retry 3 --retry-delay 3 --retry-connrefused \
    -X POST "http://${HOST}:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$request_body")

# Text content of the first choice only (audio is base64 WAV and too large to print).
echo "Output of request: $(echo "$output" | jq -r '.choices[0].message.content // .error // .')"
# Indicate whether any choice carried audio.
has_audio=$(echo "$output" | jq '[.choices[]? | select(.message.audio != null)] | length')
echo "Choices with audio: ${has_audio}"
