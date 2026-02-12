#!/bin/bash
# Online diffusion LoRA inference via OpenAI-compatible chat API.

SERVER="${SERVER:-http://localhost:8091}"
PROMPT="${PROMPT:-A piece of cheesecake}"

LORA_PATH="${LORA_PATH:-}"
LORA_NAME="${LORA_NAME:-lora}"
LORA_SCALE="${LORA_SCALE:-1.0}"
LORA_INT_ID="${LORA_INT_ID:-}"

HEIGHT="${HEIGHT:-1024}"
WIDTH="${WIDTH:-1024}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
SEED="${SEED:-42}"

CURRENT_TIME=$(date +%Y%m%d%H%M%S)
OUTPUT="${OUTPUT:-lora_online_output_${CURRENT_TIME}.png}"

if [ -z "$LORA_PATH" ]; then
  echo "ERROR: LORA_PATH is required (must be a server-local path)."
  exit 1
fi

echo "Generating image with LoRA..."
echo "Server: $SERVER"
echo "Prompt: $PROMPT"
echo "LoRA: name=$LORA_NAME id=${LORA_INT_ID:-auto} scale=$LORA_SCALE path=$LORA_PATH"
echo "Output: $OUTPUT"

LORA_INT_ID_FIELD=""
if [ -n "$LORA_INT_ID" ]; then
  LORA_INT_ID_FIELD=", \"int_id\": $LORA_INT_ID"
fi

curl -s "$SERVER/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [
      {\"role\": \"user\", \"content\": \"$PROMPT\"}
    ],
    \"extra_body\": {
      \"height\": $HEIGHT,
      \"width\": $WIDTH,
      \"num_inference_steps\": $NUM_INFERENCE_STEPS,
      \"seed\": $SEED,
      \"lora\": {
        \"name\": \"$LORA_NAME\",
        \"local_path\": \"$LORA_PATH\",
        \"scale\": $LORA_SCALE$LORA_INT_ID_FIELD
      }
    }
  }" | jq -r '.choices[0].message.content[0].image_url.url' | sed 's/^data:image[^,]*,\s*//' | base64 -d > "$OUTPUT"

if [ -f "$OUTPUT" ]; then
  echo "Image saved to: $OUTPUT"
  echo "Size: $(du -h "$OUTPUT" | cut -f1)"
else
  echo "Failed to generate image"
  exit 1
fi
