#!/bin/bash
# Wan2.2 text-to-video curl example

SERVER="${SERVER:-http://localhost:8093}"
PROMPT="${PROMPT:-A cinematic shot of a flying kite over the ocean.}"
OUTPUT="${OUTPUT:-wan22_output.mp4}"

HEIGHT="${HEIGHT:-720}"
WIDTH="${WIDTH:-1280}"
NUM_FRAMES="${NUM_FRAMES:-81}"
FPS="${FPS:-24}"
STEPS="${STEPS:-40}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
GUIDANCE_SCALE_2="${GUIDANCE_SCALE_2:-4.0}"
SEED="${SEED:-42}"

echo "Generating video..."
echo "Prompt: $PROMPT"
echo "Output: $OUTPUT"

curl -s "$SERVER/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [
      {\"role\": \"user\", \"content\": \"$PROMPT\"}
    ],
    \"extra_body\": {
      \"height\": $HEIGHT,
      \"width\": $WIDTH,
      \"num_frames\": $NUM_FRAMES,
      \"fps\": $FPS,
      \"num_inference_steps\": $STEPS,
      \"guidance_scale\": $GUIDANCE_SCALE,
      \"guidance_scale_2\": $GUIDANCE_SCALE_2,
      \"seed\": $SEED,
      \"num_outputs_per_prompt\": 1
    }
  }" | jq -r '.choices[0].message.content[0].video_url.url' | cut -d',' -f2 | base64 -d > "$OUTPUT"

if [ -f "$OUTPUT" ]; then
    echo "Video saved to: $OUTPUT"
    echo "Size: $(du -h "$OUTPUT" | cut -f1)"
else
    echo "Failed to generate video"
    exit 1
fi
