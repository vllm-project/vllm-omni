#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# FLUX 3 Image Klein online serving startup script.
#
# Usage:
#   MODEL=/path/to/flux-3-image-klein ./run_server.sh
#   MODEL=/path/to/flux-3-image-klein PORT=8099 TENSOR_PARALLEL_SIZE=4 ./run_server.sh

MODEL="${MODEL:-}"
PORT="${PORT:-8099}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
CFG_PARALLEL_SIZE="${CFG_PARALLEL_SIZE:-1}"
# Leave blank to let the platform auto-select the optimal attention backend.
# Override with e.g. DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN if your FA build matches the GPU.
DIFFUSION_ATTENTION_BACKEND="${DIFFUSION_ATTENTION_BACKEND:-}"

for arg in "$@"; do
    case "$arg" in
        -h|--help)
            echo "Usage: MODEL=/path/to/flux-3-image-klein ./run_server.sh"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $arg" >&2
            exit 1
            ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "ERROR: MODEL is not set. Usage: MODEL=/path/to/flux-3-image-klein ./run_server.sh" >&2
    exit 1
fi
if [ ! -d "$MODEL" ]; then
    echo "ERROR: MODEL path does not exist or is not a directory: $MODEL" >&2
    exit 1
fi

echo "Starting FLUX 3 Image Klein server..."
echo "Model:                $MODEL"
echo "Port:                 $PORT"
echo "Tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "CFG parallel size:    $CFG_PARALLEL_SIZE"

SERVE_CMD=(vllm serve "$MODEL" --omni
    --served-model-name flux-3-image-klein
    --port "$PORT"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --cfg-parallel-size "$CFG_PARALLEL_SIZE"
)

if [ -n "$DIFFUSION_ATTENTION_BACKEND" ]; then
    DIFFUSION_ATTENTION_BACKEND="$DIFFUSION_ATTENTION_BACKEND" "${SERVE_CMD[@]}"
else
    "${SERVE_CMD[@]}"
fi
