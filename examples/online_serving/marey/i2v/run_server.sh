#!/bin/bash
# Marey MMDiT online serving startup script.
#
# The model directory must contain a config.yaml with text_encoder, vae, model,
# and scheduler sections (see examples/offline_inference/marey/text_to_video.py).
#
# The model directory has neither model_index.json nor config.json, so stage
# config auto-detection cannot resolve a model_type and falls through to the
# default single-stage diffusion factory. --model-class-name MareyPipeline
# names the pipeline class that the factory should instantiate (looked up in
# DiffusionModelRegistry at vllm_omni/diffusion/registry.py).
#
# Required env vars:
#   MODEL                - Path to the Marey checkpoint directory (with config.yaml).
#
# flow_shift is set per-request by the curl clients (it overrides the
# pipeline default), so this script no longer takes FLOW_SHIFT.
#
# Optional env vars:
#   PORT                       - Server port (default: 8098).
#   ULYSSES_DEGREE             - Sequence parallel degree (default: 8).
#   GPU_MEMORY_UTILIZATION     - GPU memory utilization (default: 0.98).
#   HF_HOME                    - HuggingFace cache directory.
#   VLLM_OMNI_STORAGE_PATH     - vllm-omni storage directory.
#   VLLM_OMNI_PROJECT          - Path to the vllm-omni checkout for `uv run --project`
#                                (default: repo root inferred from this script's location).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

: "${MODEL:?MODEL must be set to the Marey checkpoint directory}"

PORT="${PORT:-8098}"
ULYSSES_DEGREE="${ULYSSES_DEGREE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.98}"
VLLM_OMNI_PROJECT="${VLLM_OMNI_PROJECT:-${REPO_ROOT}}"

echo "Starting Marey server..."
echo "Model:              $MODEL"
echo "Port:               $PORT"
echo "Ulysses degree:     $ULYSSES_DEGREE"

env_args=(
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
)
[[ -n "${HF_HOME:-}" ]]                && env_args+=("HF_HOME=${HF_HOME}")
[[ -n "${VLLM_OMNI_STORAGE_PATH:-}" ]] && env_args+=("VLLM_OMNI_STORAGE_PATH=${VLLM_OMNI_STORAGE_PATH}")

# Debug / reproducibility toggles — uncomment and append to env_args to use:
# env_args+=("MAREY_DUMP_DIR=/path/to/pipeline_dump/")
# env_args+=("MAREY_LOAD_INITIAL_NOISE=/path/to/z_initial_noise.pt")
# env_args+=("MAREY_LOAD_STEP_NOISE_DIR=/path/to/step_noise_dir/")

env "${env_args[@]}" \
uv run --project "${VLLM_OMNI_PROJECT}" vllm-omni serve "$MODEL" --omni \
    --port "$PORT" \
    --model-class-name MareyPipeline \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --ulysses-degree "$ULYSSES_DEGREE" \
    --use-hsdp \
    --hsdp-shard-size "$ULYSSES_DEGREE"
