#!/bin/bash
set -euo pipefail

# Wan2.2 T2V online serving script.
# This script intentionally matches the old v0.16 squashed validation config:
#   single backend on :8091
#   --usp 8
#   --enable-layerwise-offload
#   --boundary-ratio 0.875
#   --vae-use-slicing
#   --vae-use-tiling

MODEL="${MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}"
PORT="${PORT:-8091}"
USP="${USP:-8}"
BOUNDARY_RATIO="${BOUNDARY_RATIO:-0.875}"
FLOW_SHIFT="${FLOW_SHIFT:-5.0}"
ENABLE_LAYERWISE_OFFLOAD="${ENABLE_LAYERWISE_OFFLOAD:-1}"
VAE_USE_SLICING="${VAE_USE_SLICING:-1}"
VAE_USE_TILING="${VAE_USE_TILING:-1}"
CACHE_BACKEND="${CACHE_BACKEND:-none}"
ENABLE_CACHE_DIT_SUMMARY="${ENABLE_CACHE_DIT_SUMMARY:-0}"

echo "Starting Wan2.2 T2V server..."
echo "Model: ${MODEL}"
echo "Port: ${PORT}"
echo "USP: ${USP}"
echo "Layerwise offload: ${ENABLE_LAYERWISE_OFFLOAD}"
echo "Boundary ratio: ${BOUNDARY_RATIO}"
echo "Flow shift: ${FLOW_SHIFT}"
echo "VAE slicing: ${VAE_USE_SLICING}"
echo "VAE tiling: ${VAE_USE_TILING}"
echo "Cache backend: ${CACHE_BACKEND}"
if [ "${ENABLE_CACHE_DIT_SUMMARY}" != "0" ]; then
    echo "Cache-DiT summary: enabled"
fi

CMD=(
    python3 -m vllm_omni.entrypoints.cli.main serve "${MODEL}" --omni
    --port "${PORT}"
    --usp "${USP}"
    --boundary-ratio "${BOUNDARY_RATIO}"
    --flow-shift "${FLOW_SHIFT}"
)

if [ "${ENABLE_LAYERWISE_OFFLOAD}" != "0" ]; then
    CMD+=(--enable-layerwise-offload)
fi
if [ "${VAE_USE_SLICING}" != "0" ]; then
    CMD+=(--vae-use-slicing)
fi
if [ "${VAE_USE_TILING}" != "0" ]; then
    CMD+=(--vae-use-tiling)
fi
if [ "${CACHE_BACKEND}" != "none" ]; then
    CMD+=(--cache-backend "${CACHE_BACKEND}")
fi
if [ "${ENABLE_CACHE_DIT_SUMMARY}" != "0" ]; then
    CMD+=(--enable-cache-dit-summary)
fi

exec "${CMD[@]}"
