#!/bin/bash
# InternVL-U online serving startup script.
#
# THINK=1 selects the text-then-image deployment (internvlu_chat_think.yaml),
# where the VLM writes a chain-of-thought description before generating the
# image. Think mode is configured statically per deployment, not per request.

MODEL="${MODEL:-InternVL-U/InternVL-U}"
PORT="${PORT:-8091}"
THINK="${THINK:-0}"

DEPLOY_DIR="$(python -c 'import pathlib, vllm_omni; print(pathlib.Path(vllm_omni.__file__).parent / "deploy")')"
if [ "$THINK" = "1" ]; then
    DEPLOY_CONFIG="$DEPLOY_DIR/internvlu_chat_think.yaml"
else
    DEPLOY_CONFIG="$DEPLOY_DIR/internvlu_chat.yaml"
fi

echo "Starting InternVL-U server..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Deploy config: $DEPLOY_CONFIG"

vllm serve "$MODEL" --omni \
    --port "$PORT" \
    --deploy-config "$DEPLOY_CONFIG"
