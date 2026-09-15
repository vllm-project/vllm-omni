#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# MiniMax H3 online serving startup script
# Requires 4x H20 (96GB) GPUs. Adjust --usp / --text-encoder-tp-size /
# --vae-patch-parallel-size to match your GPU count.

MODEL="${MODEL:-/path/to/MiniMax-H3/Ref2VA}"
PORT="${PORT:-8099}"
USP="${USP:-4}"
TEXT_ENCODER_TP="${TEXT_ENCODER_TP:-4}"
VAE_PP="${VAE_PP:-4}"

echo "Starting MiniMax H3 server..."
echo "Model:                ${MODEL}"
echo "Port:                 ${PORT}"
echo "USP (ulysses degree): ${USP}"
echo "Text-encoder TP:      ${TEXT_ENCODER_TP}"
echo "VAE patch parallel:   ${VAE_PP}"
echo ""
echo "Note: quality=lossless is a per-request flag (send it in each curl request,"
echo "      not as a server startup argument)."

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
vllm serve "${MODEL}" --omni \
    --port "${PORT}" \
    --usp "${USP}" \
    --trust-remote-code \
    --text-encoder-tp-size "${TEXT_ENCODER_TP}" \
    --vae-patch-parallel-size "${VAE_PP}" \
    --vae-use-tiling
