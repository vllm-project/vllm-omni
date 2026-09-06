#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Launch two standalone Qwen3-TTS stages for disaggregated serving.
#
# Usage:
#   ./run_standalone_servers.sh                  # Default model, needs 2 GPUs
#   ./run_standalone_servers.sh CustomVoice      # CustomVoice model
#
# This starts talker (stage 0) on GPU 0 / port 8000
# and code2wav (stage 1) on GPU 1 / port 8001.
# Then use standalone_disagg_client.py to chain them.

set -e

TASK_TYPE="${1:-CustomVoice}"

case "$TASK_TYPE" in
    CustomVoice)
        MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        ;;
    Base)
        MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        ;;
    *)
        echo "Unknown task type: $TASK_TYPE"
        echo "Supported: CustomVoice, Base"
        exit 1
        ;;
esac

echo "Starting standalone talker (stage 0) on GPU 0, port 8000..."
CUDA_VISIBLE_DEVICES=0 vllm-omni serve "$MODEL" \
    --omni \
    --standalone \
    --stage-id 0 \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code &
TALKER_PID=$!

echo "Starting standalone code2wav (stage 1) on GPU 1, port 8001..."
CUDA_VISIBLE_DEVICES=1 vllm-omni serve "$MODEL" \
    --omni \
    --standalone \
    --stage-id 1 \
    --host 0.0.0.0 \
    --port 8001 \
    --trust-remote-code &
CODE2WAV_PID=$!

echo "Talker PID=$TALKER_PID (GPU 0), Code2wav PID=$CODE2WAV_PID (GPU 1)"
echo "Waiting for servers... use standalone_disagg_client.py to test."

trap "kill $TALKER_PID $CODE2WAV_PID 2>/dev/null" EXIT
wait
