#!/bin/bash
# SANA-Video-2B image-to-video serving through the Diffusers adapter.
# Both 480p and 720p checkpoints are validated. For 720p:
# MODEL=Efficient-Large-Model/SANA-Video_2B_720p_diffusers bash run_server_sana_video_diffusers.sh
# INPUT_IMAGE=input.jpg WIDTH=1280 HEIGHT=704 bash run_curl_sana_video.sh

set -euo pipefail

MODEL="${MODEL:-Efficient-Large-Model/SANA-Video_2B_480p_diffusers}"
PORT="${PORT:-8099}"

vllm serve "$MODEL" \
    --omni \
    --model-class-name SanaImageToVideoPipeline \
    --diffusion-load-format diffusers \
    --diffusion-attention-backend TORCH_SDPA \
    --dtype bfloat16 \
    --port "$PORT"
