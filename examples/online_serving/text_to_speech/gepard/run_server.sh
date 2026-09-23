#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Launch vLLM-Omni server for Gepard-1.0 TTS (zero-shot default voice).
#
# Usage:
#   ./run_server.sh
#   CUDA_VISIBLE_DEVICES=0 ./run_server.sh
#
# Packaged deploy config: vllm_omni/deploy/gepard.yaml (async_chunk=false,
# 22.05 kHz mono, default seed 42 until the YAML seed is removed).
# Cold start downloads the talker and NeMo NanoCodec; the serve default of
# --stage-init-timeout 300 is too tight (offline end2end.py uses 900).
#
# On a host whose CUDA toolkit cannot JIT FlashInfer (no nvcc/ninja, or a
# consumer Blackwell sm_120 card), also:
#   export VLLM_USE_FLASHINFER_SAMPLER=0

set -e

MODEL="${MODEL:-nineninesix/gepard-1.0}"
PORT="${PORT:-8091}"

echo "Starting Gepard-1.0 server with model: $MODEL"

vllm-omni serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --omni \
    --stage-init-timeout 900 \
    --deploy-config vllm_omni/deploy/gepard.yaml
