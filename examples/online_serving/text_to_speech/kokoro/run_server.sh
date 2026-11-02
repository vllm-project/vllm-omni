#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Launch vLLM-Omni server for Kokoro-82M TTS (54 built-in voices, 24 kHz).
#
# Usage:
#   ./run_server.sh
#   CUDA_VISIBLE_DEVICES=0 ./run_server.sh
#
# Requires the G2P extra: pip install 'vllm-omni[kokoro]'.
# The packaged deploy config (vllm_omni/deploy/kokoro.yaml) is picked up
# automatically from the model name.
#
# On a host whose CUDA toolkit cannot JIT FlashInfer (no nvcc/ninja, or a
# consumer Blackwell sm_120 card), also:
#   export VLLM_USE_FLASHINFER_SAMPLER=0

set -e

MODEL="${MODEL:-hexgrad/Kokoro-82M}"
PORT="${PORT:-8091}"

echo "Starting Kokoro-82M server with model: $MODEL"

vllm-omni serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --omni
