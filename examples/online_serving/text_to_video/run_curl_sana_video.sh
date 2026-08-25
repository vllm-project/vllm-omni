#!/bin/bash

BASE_URL="${BASE_URL:-http://localhost:8091}"
OUTPUT_PATH="${OUTPUT_PATH:-sana_video.mp4}"

curl -sS -X POST "$BASE_URL/v1/videos/sync" \
    -F "prompt=A cinematic tracking shot of a sailboat crossing the ocean at sunset" \
    -F "height=480" \
    -F "width=832" \
    -F "num_frames=81" \
    -F "fps=16" \
    -F "num_inference_steps=50" \
    -F "guidance_scale=6.0" \
    -F "seed=42" \
    -F 'extra_params={"motion_score":30}' \
    --output "$OUTPUT_PATH"
