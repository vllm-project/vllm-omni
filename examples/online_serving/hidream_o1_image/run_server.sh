#!/bin/bash
# HiDream-O1-Image online serving startup script.
# Defaults to the Dev variant (28 steps, no CFG, faster iteration).
# Set MODEL to HiDream-ai/HiDream-O1-Image for the full 50-step variant.

MODEL="${MODEL:-HiDream-ai/HiDream-O1-Image-Dev}"
PORT="${PORT:-8095}"

echo "Starting HiDream-O1-Image server..."
echo "Model: $MODEL"
echo "Port:  $PORT"

vllm serve "$MODEL" --omni \
    --port "$PORT"
