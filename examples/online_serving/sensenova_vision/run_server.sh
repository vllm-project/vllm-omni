#!/bin/bash
# SenseNova-Vision online serving startup script

MODEL="${MODEL:-sensenova/SenseNova-Vision-7B-MoT}"
PORT="${PORT:-8092}"
DEPLOY_CONFIG="${DEPLOY_CONFIG:-vllm_omni/deploy/sensenova_vision.yaml}"

echo "Starting SenseNova-Vision server..."
echo "Model:         $MODEL"
echo "Deploy config: $DEPLOY_CONFIG"
echo "Port:          $PORT"

vllm serve "$MODEL" --omni \
    --deploy-config "$DEPLOY_CONFIG" \
    --port "$PORT"
