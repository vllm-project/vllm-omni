#!/bin/bash
# GLM-Image text-to-image curl example

set -euo pipefail

PROMPT="${1:-A beautiful sunset over the ocean with sailing boats}"
SERVER="${SERVER:-http://localhost:8091}"
OUTPUT="${OUTPUT:-glm_image_t2i_output.png}"

echo "Generating image..."
echo "Server: $SERVER"
echo "Prompt: $PROMPT"
echo "Output: $OUTPUT"

curl -s "$SERVER/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [
      {\"role\": \"user\", \"content\": \"$PROMPT\"}
    ],
    \"extra_body\": {
      \"height\": 1024,
      \"width\": 1024,
      \"num_inference_steps\": 50,
      \"guidance_scale\": 1.5,
      \"seed\": 42
    }
  }" | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2- | base64 -d > "$OUTPUT"

echo "Image saved to: $OUTPUT"
