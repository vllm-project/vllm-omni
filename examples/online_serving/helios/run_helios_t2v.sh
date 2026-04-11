#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Helios Text-to-Video Generation Example
# This script demonstrates how to use the /v1/videos API with Helios models

API_URL="${API_URL:-http://localhost:8000}"
MODEL="${MODEL:-BestWishYsh/Helios-Base}"
PROMPT="${PROMPT:-A serene lakeside sunrise with mist over the water.}"

echo "==================================="
echo "Helios T2V Generation"
echo "==================================="
echo "API URL: $API_URL"
echo "Model: $MODEL"
echo "Prompt: $PROMPT"
echo ""

# Create video generation job
echo "Creating video generation job..."
RESPONSE=$(curl -s -X POST "$API_URL/v1/videos" \
  -F "prompt=$PROMPT" \
  -F "model=$MODEL" \
  -F "width=640" \
  -F "height=384" \
  -F "num_frames=99" \
  -F "num_inference_steps=50" \
  -F "guidance_scale=5.0" \
  -F "seed=42")

echo "Response: $RESPONSE"
echo ""

# Extract video ID
VIDEO_ID=$(echo "$RESPONSE" | grep -o '"id":"[^"]*"' | cut -d'"' -f4)

if [ -z "$VIDEO_ID" ]; then
  echo "Error: Failed to create video job"
  exit 1
fi

echo "Video job created: $VIDEO_ID"
echo ""

# Poll for completion
echo "Polling for completion..."
while true; do
  STATUS_RESPONSE=$(curl -s "$API_URL/v1/videos/$VIDEO_ID")
  STATUS=$(echo "$STATUS_RESPONSE" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
  PROGRESS=$(echo "$STATUS_RESPONSE" | grep -o '"progress":[0-9]*' | cut -d':' -f2)

  echo "Status: $STATUS, Progress: $PROGRESS%"

  if [ "$STATUS" = "completed" ]; then
    echo ""
    echo "Video generation completed!"
    echo "Full response:"
    echo "$STATUS_RESPONSE" | jq '.'
    break
  elif [ "$STATUS" = "failed" ]; then
    echo ""
    echo "Video generation failed!"
    echo "$STATUS_RESPONSE" | jq '.'
    exit 1
  fi

  sleep 2
done
