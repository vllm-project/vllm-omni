#!/bin/bash
# Qwen-Image-Edit image editing curl example

SERVER="${SERVER:-http://localhost:8092}"
INPUT_IMAGE="${1:-input.png}"
PROMPT="${2:-Convert this image to watercolor style}"
OUTPUT="${3:-output_$(date +%Y%m%d_%H%M%S).png}"

if [ ! -f "$INPUT_IMAGE" ]; then
    echo "Error: Input image not found: $INPUT_IMAGE"
    echo "Usage: $0 <input_image> [prompt] [output]"
    exit 1
fi

echo "Editing image..."
echo "Input: $INPUT_IMAGE"
echo "Prompt: $PROMPT"
echo "Output: $OUTPUT"

# Read image and convert to base64
IMG_B64=$(base64 -w0 "$INPUT_IMAGE")

curl -s "$SERVER/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"text\", \"text\": \"$PROMPT\"},
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,$IMG_B64\"}}
      ]
    }]
  }" | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2 | base64 -d > "$OUTPUT"

if [ -f "$OUTPUT" ]; then
    echo "Image saved to: $OUTPUT"
    echo "Size: $(du -h "$OUTPUT" | cut -f1)"
else
    echo "Failed to edit image"
    exit 1
fi
