#!/bin/bash
# SANA-Video-2B native image-to-video serving.

set -euo pipefail

MODEL="${MODEL:-Efficient-Large-Model/SANA-Video_2B_480p_diffusers}"
PORT="${PORT:-8099}"

vllm serve "$MODEL" \
    --omni \
    --model-class-name SanaImageToVideoPipeline \
    --dtype bfloat16 \
    --port "$PORT"
