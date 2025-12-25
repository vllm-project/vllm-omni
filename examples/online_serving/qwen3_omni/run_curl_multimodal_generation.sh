#!/usr/bin/env bash
set -euo pipefail

# Default query type
QUERY_TYPE="${1:-use_video}"

# Default modalities argument
MODALITIES="${2:-null}"

# Validate query type
if [[ ! "$QUERY_TYPE" =~ ^(text|use_audio|use_image|use_video)$ ]]; then
    echo "Error: Invalid query type '$QUERY_TYPE'"
    echo "Usage: $0 [text|use_audio|use_image|use_video] [modalities]"
    echo "  text: Text query"
    echo "  use_audio: Audio + Text query"
    echo "  use_image: Image + Text query"
    echo "  use_video: Video + Text query"
    echo "  modalities: Modalities parameter (default: null)"
    exit 1
fi

SEED=42

thinker_sampling_params='{
  "temperature": 0.4,
  "top_p": 0.9,
  "top_k": 1,
  "max_tokens": 16384,
  "seed": 42,
  "repetition_penalty": 1.05,
  "stop_token_ids": [151645]
}'

talker_sampling_params='{
  "temperature": 0.9,
  "top_k": 50,
  "max_tokens": 4096,
  "seed": 42,
  "detokenize": false,
  "repetition_penalty": 1.05,
  "stop_token_ids": [2150]
}'

code2wav_sampling_params='{
  "temperature": 0.0,
  "top_p": 1.0,
  "top_k": -1,
  "max_tokens": 65536,
  "seed": 42,
  "detokenize": true,
  "repetition_penalty": 1.1
}'
# Above is optional, it has a default setting in stage_configs of the corresponding model.

# Define URLs for assets
MARY_HAD_LAMB_AUDIO_URL="https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/mary_had_lamb.ogg"
CHERRY_BLOSSOM_IMAGE_URL="https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg"
SAMPLE_VIDEO_URL="https://huggingface.co/datasets/raushan-testing-hf/videos-test/resolve/main/sample_demo_1.mp4"

# Build user content and extra fields based on query type
case "$QUERY_TYPE" in
  text)
    user_content='[
      {
        "type": "text",
        "text": "Explain the system architecture for a scalable audio generation pipeline. Answer in 15 words."
      }
    ]'
    sampling_params_list='[
      '"$thinker_sampling_params"',
      '"$talker_sampling_params"',
      '"$code2wav_sampling_params"'
    ]'
    mm_processor_kwargs="{}"
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
    sampling_params_list='[
      '"$thinker_sampling_params"',
      '"$talker_sampling_params"',
      '"$code2wav_sampling_params"'
    ]'
    mm_processor_kwargs="{}"
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
    sampling_params_list='[
      '"$thinker_sampling_params"',
      '"$talker_sampling_params"',
      '"$code2wav_sampling_params"'
    ]'
    mm_processor_kwargs="{}"
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
    sampling_params_list='[
      '"$thinker_sampling_params"',
      '"$talker_sampling_params"',
      '"$code2wav_sampling_params"'
    ]'
    mm_processor_kwargs="{}"
    ;;
esac

echo "Running query type: $QUERY_TYPE"
echo ""


output=$(curl -sS -X POST http://localhost:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d @- <<EOF
{
  "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
  "sampling_params_list": $sampling_params_list,
  "mm_processor_kwargs": $mm_processor_kwargs,
  "modalities": $MODALITIES,
  "messages": [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
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

# Here it only shows the text content of the first choice. Audio content has many binaries, so it's not displayed here.
echo "Output of request: $(echo "$output" | jq '.choices[0].message.content')"
