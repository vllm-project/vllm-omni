#!/usr/bin/env bash
set -euo pipefail

# Server port
PORT="${PORT:-8091}"
# Default query type
QUERY_TYPE="${1:-text}"

# Validate query type
if [[ ! "$QUERY_TYPE" =~ ^(text|use_audio|use_image|use_video|use_mixed_modalities)$ ]]; then
    echo "Error: Invalid query type '$QUERY_TYPE'"
    echo "Usage: $0 [text|use_audio|use_image|use_video|use_mixed_modalities]"
    echo "  text: Text-only query"
    echo "  use_audio: Audio + Text query"
    echo "  use_image: Image + Text query"
    echo "  use_video: Video + Text query"
    echo "  use_mixed_modalities: Audio + Image + Video + Text query"
    exit 1
fi

# Define URLs for assets
MARY_HAD_LAMB_AUDIO_URL="https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/mary_had_lamb.ogg"
CHERRY_BLOSSOM_IMAGE_URL="https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg"
SAMPLE_VIDEO_URL="https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4"

# Build user content based on query type
case "$QUERY_TYPE" in
  text)
    user_content='[
      {
        "type": "text",
        "text": "请详细介绍鹦鹉的生活习性。"
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
          "text": "Describe this image in detail."
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
          "text": "Please recognize the language of this speech and transcribe it. Format: oral."
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
          "text": "Describe what is happening in this video."
        }
      ]'
    ;;
  use_mixed_modalities)
    user_content='[
        {
          "type": "image_url",
          "image_url": {
            "url": "'"$CHERRY_BLOSSOM_IMAGE_URL"'"
          }
        },
        {
          "type": "audio_url",
          "audio_url": {
            "url": "'"$MARY_HAD_LAMB_AUDIO_URL"'"
          }
        },
        {
          "type": "text",
          "text": "Describe the image, and recognize the language of this speech and transcribe it. Format: oral"
        }
      ]'
    ;;
esac

echo "Running query type: $QUERY_TYPE"
echo ""

request_body=$(cat <<EOF
{
  "model": "Jonathan1909/Ming-flash-omni-2.0",
  "modalities": ["text"],
  "messages": [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "你是一个友好的AI助手。\n\ndetailed thinking off"
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
    -X POST http://localhost:${PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "$request_body")

echo "Output of request: $(echo "$output" | jq '.choices[0].message.content')"
