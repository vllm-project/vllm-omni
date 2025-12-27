#!/usr/bin/env bash
set -euo pipefail

SERVER_URL=${SERVER_URL:-"http://localhost:8093"}
PROMPT=${PROMPT:-"A cat surfing on waves"}

curl -s "${SERVER_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"messages\": [
      {\"role\": \"user\", \"content\": \"${PROMPT}\"}
    ],
    \"extra_body\": {
      \"height\": 480,
      \"width\": 640,
      \"num_frames\": 32,
      \"num_inference_steps\": 40,
      \"guidance_scale\": 4.0,
      \"guidance_scale_2\": 3.0,
      \"seed\": 42,
      \"fps\": 16
    }
  }" | jq -r '.choices[0].message.content[] | select(.type=="video_url") | .video_url.url' | head -n1 | cut -d',' -f2 | base64 -d > output.mp4

echo "Saved output.mp4"
