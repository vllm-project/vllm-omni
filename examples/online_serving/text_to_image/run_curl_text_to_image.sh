#!/bin/bash
# Qwen-Image text-to-image curl example

SERVER="${SERVER:-http://localhost:8091}"
PROMPT="${PROMPT:-a good boy in the ocean}"
CURRENT_TIME=$(date +%Y%m%d%H%M%S)
OUTPUT="${OUTPUT:-qwen_image_output_${CURRENT_TIME}.png}"

echo "Generating image..."
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
      \"num_inference_steps\": 100,
      \"true_cfg_scale\": 4.0,
      \"seed\": 42,
      \"num_outputs_per_prompt\": 1
    }
  }" | jq -r '.choices[0].message.content[0].image_url.url' | sed 's/^data:image[^,]*,\s*//' | base64 -d > "$OUTPUT"

if [ -f "$OUTPUT" ]; then
    echo "Image saved to: $OUTPUT"
    echo "Size: $(du -h "$OUTPUT" | cut -f1)"
else
    echo "Failed to generate image"
    exit 1
fi
