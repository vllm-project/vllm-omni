#!/bin/bash
# Qwen-Image image-edit (image-to-image) curl example

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <input_image> \"<edit_prompt>\" [output_file]" >&2
  exit 1
fi

INPUT_IMG=$1
PROMPT=$2
SERVER="${SERVER:-http://localhost:8092}"
CURRENT_TIME=$(date +%Y%m%d%H%M%S)
OUTPUT="${3:-image_edit_${CURRENT_TIME}.png}"

if [[ ! -f "$INPUT_IMG" ]]; then
  echo "Input image not found: $INPUT_IMG" >&2
  exit 1
fi

REQUEST_JSON_FILE=$(mktemp)
trap 'rm -f "$REQUEST_JSON_FILE"' EXIT

# Pipe base64 into jq via stdin to avoid ARG_MAX limit on large images
base64 -w0 "$INPUT_IMG" \
  | jq -Rs --arg prompt "$PROMPT" '{
    messages: [{
      role: "user",
      content: [
        {"type": "text", "text": $prompt},
        {"type": "image_url", "image_url": {"url": ("data:image/png;base64," + .)}}
      ]
    }],
    extra_body: {
      num_inference_steps: 50,
      guidance_scale: 1,
      seed: 42
    }
  }' > "$REQUEST_JSON_FILE"

echo "Generating edited image..."
echo "Server: $SERVER"
echo "Prompt: $PROMPT"
echo "Input : $INPUT_IMG"
echo "Output: $OUTPUT"

curl -s "$SERVER/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @"$REQUEST_JSON_FILE" \
  | jq -r '.choices[0].message.content[0].image_url.url' \
  | cut -d',' -f2 \
  | base64 -d > "$OUTPUT"

if [[ -f "$OUTPUT" ]]; then
  echo "Image saved to: $OUTPUT"
  echo "Size: $(du -h "$OUTPUT" | cut -f1)"
else
  echo "Failed to generate image"
  exit 1
fi
