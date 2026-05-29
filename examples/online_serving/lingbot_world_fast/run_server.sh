#!/bin/bash
# Lingbot World Fast online serving startup script

MODEL="${MODEL:-../../offline_inference/lingbot_world_fast/lingbot_world/lingbot-world-base-cam/Lingbot-World-Fast}"
PORT="${PORT:-8091}"

echo "Starting Lingbot World server..."
echo "Model: $MODEL"
echo "Port: $PORT"

vllm serve "$MODEL" --omni \
    --port "$PORT" \
    --model-class-name LingbotWorldFastPipeline \
    --stage-init-timeout 6000 \
    --init-timeout 6000 \
    --ws-max-size 268435456 \
    --ws wsproto
