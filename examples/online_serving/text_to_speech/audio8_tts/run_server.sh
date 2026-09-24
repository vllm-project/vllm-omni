#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Launch a vLLM-Omni server for Audio8 TTS Preview 0.6B.
#
# Usage:
#   ./run_server.sh
#   CUDA_VISIBLE_DEVICES=0 PORT=8092 ./run_server.sh

set -e

MODEL="${MODEL:-Audio8/Audio8-TTS-Preview-0.6b}"
PORT="${PORT:-8092}"

echo "Starting Audio8 TTS Preview server with model: $MODEL"

# --trust-remote-code is deliberately NOT passed: vllm-omni registers its own
# `arktts` config, and transformers would otherwise prefer the checkpoint's
# remote code and bypass it.
vllm serve "$MODEL" \
    --omni \
    --host 0.0.0.0 \
    --port "$PORT"
