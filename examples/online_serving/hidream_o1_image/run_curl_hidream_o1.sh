#!/bin/bash
# HiDream-O1-Image text-to-image via /v1/images/generations.
# Adjust HOST/PORT to match your server.

HOST="${HOST:-localhost}"
PORT="${PORT:-8095}"
OUTPUT="${OUTPUT:-hidream_o1_output.png}"

curl -s "http://${HOST}:${PORT}/v1/images/generations" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "A golden retriever running through a field of sunflowers at sunset",
        "size": "1024x1024",
        "num_inference_steps": 28,
        "seed": 42
    }' | jq -r '.data[0].b64_json' | base64 -d > "$OUTPUT"

echo "Saved to $OUTPUT"
