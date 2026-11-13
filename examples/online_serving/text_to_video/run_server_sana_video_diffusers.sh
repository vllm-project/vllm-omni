#!/bin/bash
# SANA-Video through the black-box Diffusers backend adapter.

MODEL="${MODEL:-Efficient-Large-Model/SANA-Video_2B_480p_diffusers}"
PORT="${PORT:-8091}"

vllm serve "$MODEL" \
    --omni \
    --diffusion-load-format diffusers \
    --diffusion-attention-backend TORCH_SDPA \
    --dtype bfloat16 \
    --port "$PORT"
