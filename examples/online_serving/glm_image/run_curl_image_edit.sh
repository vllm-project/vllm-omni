#!/bin/bash
# GLM-Image image-edit (image-to-image) curl example

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <input_image> \"<edit_prompt>\" [output_file]" >&2
  exit 1
fi

INPUT_IMG=$1
PROMPT=$2
SERVER="${SERVER:-http://localhost:8091}"
CURRENT_TIME=$(date +%Y%m%d%H%M%S)
OUTPUT="${3:-glm_image_i2i_${CURRENT_TIME}.png}"

if [[ ! -f "$INPUT_IMG" ]]; then
  echo "Input image not found: $INPUT_IMG" >&2
  exit 1
fi

# base64 encode (macOS uses -i, Linux uses -w0)
if [[ "$(uname)" == "Darwin" ]]; then
  IMG_B64=$(base64 < "$INPUT_IMG" | tr -d '\n')
else
  IMG_B64=$(base64 -w0 "$INPUT_IMG")
fi

REQUEST_JSON=$(
  jq -n --arg prompt "$PROMPT" --arg img "$IMG_B64" '{
    messages: [{
      role: "user",
      content: [
        {"type": "text", "text": $prompt},
        {"type": "image_url", "image_url": {"url": ("data:image/png;base64," + $img)}}
      ]
    }],
    extra_body: {
      height: 1024,
      width: 1024,
      num_inference_steps: 50,
      guidance_scale: 1.5,
      seed: 42
    }
  }'
)

echo "Generating edited image..."
echo "Server: $SERVER"
echo "Prompt: $PROMPT"
echo "Input : $INPUT_IMG"
echo "Output: $OUTPUT"

curl -s "$SERVER/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$REQUEST_JSON" \
  | jq -r '.choices[0].message.content[0].image_url.url' \
  | cut -d',' -f2- \
  | base64 -d > "$OUTPUT"

echo "Image saved to: $OUTPUT"
