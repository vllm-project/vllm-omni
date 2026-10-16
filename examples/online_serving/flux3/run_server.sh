#!/bin/bash
# FLUX 3 Image Klein online serving startup script.
#
# Usage:
#   MODEL=/path/to/flux-3-image-klein ./run_server.sh
#   MODEL=/path/to/flux-3-image-klein VARIANT=fp8r ./run_server.sh
#   MODEL=/path/to/flux-3-image-klein ./run_server.sh --fp8-attention
#   MODEL=/path/to/flux-3-image-klein LORA_PATH=/path/to/adapter LORA_SCALE=1.0 ./run_server.sh
#   MODEL=/path/to/flux-3-image-klein PORT=8099 TENSOR_PARALLEL_SIZE=4 ./run_server.sh

MODEL="${MODEL:-}"
VARIANT="${VARIANT:-}"
LORA_PATH="${LORA_PATH:-}"
LORA_SCALE="${LORA_SCALE:-1.0}"
PORT="${PORT:-8099}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
CFG_PARALLEL_SIZE="${CFG_PARALLEL_SIZE:-1}"
# Leave blank to let the platform auto-select the optimal attention backend.
# Override with e.g. DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN if your FA build matches the GPU.
DIFFUSION_ATTENTION_BACKEND="${DIFFUSION_ATTENTION_BACKEND:-}"

FP8_ATTENTION=0
for arg in "$@"; do
    case "$arg" in
        --fp8-attention) FP8_ATTENTION=1 ;;
        -h|--help)
            echo "Usage: MODEL=/path/to/flux-3-image-klein ./run_server.sh [--fp8-attention]"
            echo "FP8 attention uses FlashInfer/cuDNN and is available on SM100."
            echo "Optional LoRA preload: LORA_PATH=/absolute/path/to/adapter LORA_SCALE=1.0"
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
if [ -n "$VARIANT" ] && [[ ! "$VARIANT" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: VARIANT must contain only letters, numbers, '.', '_', or '-': $VARIANT" >&2
    exit 1
fi

echo "Starting FLUX 3 Image Klein server..."
echo "Model:                $MODEL"
echo "Variant:              ${VARIANT:-bf16}"
echo "Port:                 $PORT"
echo "Tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "CFG parallel size:    $CFG_PARALLEL_SIZE"

SERVE_CMD=(vllm serve "$MODEL" --omni
    --served-model-name flux-3-image-klein
    --port "$PORT"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --cfg-parallel-size "$CFG_PARALLEL_SIZE"
)

if [ -n "$VARIANT" ]; then
    SERVE_CMD+=(--stage-overrides "{\"0\":{\"additional_config\":{\"variant\":\"$VARIANT\"}}}")
fi

if [ -n "$LORA_PATH" ]; then
    SERVE_CMD+=(--lora-path "$LORA_PATH" --lora-scale "$LORA_SCALE")
fi

if [ "$FP8_ATTENTION" -eq 1 ]; then
    SERVE_CMD+=(--diffusion-attention-config '{"default":{"backend":"FLASHINFER_ATTN","quant":{"dtype_qk":"fp8_e4m3"}}}')
    # The explicit FP8 config selects the backend for this launch.
    env -u DIFFUSION_ATTENTION_BACKEND "${SERVE_CMD[@]}"
elif [ -n "$DIFFUSION_ATTENTION_BACKEND" ]; then
    DIFFUSION_ATTENTION_BACKEND="$DIFFUSION_ATTENTION_BACKEND" "${SERVE_CMD[@]}"
else
    "${SERVE_CMD[@]}"
fi
