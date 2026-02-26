#!/bin/bash
# Wan2.2 image-to-video curl example (OpenAI-style multipart)

INPUT_IMAGE="${INPUT_IMAGE:-../../offline_inference/image_to_video/qwen-bear.png}"
OUTPUT_PATH="${OUTPUT_PATH:-wan22_i2v_output.mp4}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"

if [ ! -f "$INPUT_IMAGE" ]; then
    echo "Input image not found: $INPUT_IMAGE"
    exit 1
fi

NEGATIVE_PROMPT_FLAG=""
if [ -n "$NEGATIVE_PROMPT" ]; then
    NEGATIVE_PROMPT_FLAG="-F negative_prompt=${NEGATIVE_PROMPT}"
fi

curl -X POST http://localhost:8099/v1/videos \
  -H "Accept: application/json" \
  -F "prompt=A bear playing with yarn, smooth motion" \
  $NEGATIVE_PROMPT_FLAG \
  -F "input_reference=@${INPUT_IMAGE}" \
  -F "seconds=2" \
  -F "size=832x480" \
  -F "fps=16" \
  -F "num_inference_steps=40" \
  -F "guidance_scale=1.0" \
  -F "guidance_scale_2=1.0" \
  -F "boundary_ratio=0.875" \
  -F "flow_shift=12.0" \
  -F "seed=42" | jq -r '.data[0].b64_json' | base64 -d > "${OUTPUT_PATH}"

echo "Saved video to ${OUTPUT_PATH}"
