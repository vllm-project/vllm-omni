#!/bin/bash

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8099}"
OUTPUT_PATH="${OUTPUT_PATH:-lingbot_t2i.png}"

curl -sS -X POST "${BASE_URL}/v1/images/generations" \
    -H "Content-Type: application/json" \
    -d "{
        \"prompt\": \"a red fox standing in fresh snow\",
        \"size\": \"320x192\",
        \"num_inference_steps\": 2,
        \"guidance_scale\": 3.0,
        \"flow_shift\": 3.0,
        \"seed\": 42,
        \"response_format\": \"b64_json\"
    }" | jq -er '.data[0].b64_json' | base64 -d > "${OUTPUT_PATH}"

echo "Saved image to ${OUTPUT_PATH}"
