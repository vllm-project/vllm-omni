#!/bin/bash
# Bagel multi-stage online serving startup script
# Starts stage 0 as master with API server, and stage 1 in headless mode

MODEL="${MODEL:-ByteDance-Seed/BAGEL-7B-MoT}"
PORT="${PORT:-8091}"
MASTER_ADDRESS="${MASTER_ADDRESS:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-8092}"
STAGE_CONFIGS_PATH="$(dirname "$0")/../../../vllm_omni/model_executor/stage_configs/bagel.yaml"

echo "Starting Bagel multi-stage server..."
echo "Model: $MODEL"
echo "API Port: $PORT"
echo "Master Address: $MASTER_ADDRESS"
echo "Master Port: $MASTER_PORT"
echo "Stage Configs: $STAGE_CONFIGS_PATH"

# Start stage 1 (DiT) in headless mode first
echo "Starting Stage 1 (DiT) in headless mode..."
vllm serve "$MODEL" --omni \
    --stage-configs-path "$STAGE_CONFIGS_PATH" \
    --stage-id 1 \
    --headless \
    -oma "$MASTER_ADDRESS" \
    -omp "$MASTER_PORT" &

# Start stage 0 (Thinker) as master with API server
echo "Starting Stage 0 (Thinker) as master..."
vllm serve "$MODEL" --omni \
    --port "$PORT" \
    --stage-configs-path "$STAGE_CONFIGS_PATH" \
    --stage-id 0 \
    -oma "$MASTER_ADDRESS" \
    -omp "$MASTER_PORT"
