#!/bin/bash
# Launch vLLM-Omni server for VoxCPM online speech serving.
#
# Usage:
#   ./run_server.sh                 # default: async_chunk stage config
#   ./run_server.sh async           # async_chunk stage config
#   ./run_server.sh sync            # no-async-chunk stage config
#   VOXCPM_MODEL=/path/to/model ./run_server.sh

set -e

MODE="${1:-async}"
MODEL="${VOXCPM_MODEL:-OpenBMB/VoxCPM1.5}"

case "$MODE" in
    async)
        STAGE_CONFIG="vllm_omni/model_executor/stage_configs/voxcpm_async_chunk.yaml"
        ;;
    sync)
        STAGE_CONFIG="vllm_omni/deploy/voxcpm.yaml"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Supported: async, sync"
        exit 1
        ;;
esac

echo "Starting VoxCPM server with model: $MODEL"
echo "Stage config: $STAGE_CONFIG"

vllm serve "$MODEL" \
    --stage-configs-path "$STAGE_CONFIG" \
    --host 0.0.0.0 \
    --port 8091 \
    --trust-remote-code \
    --enforce-eager \
    --omni
