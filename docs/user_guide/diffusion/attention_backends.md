# Diffusion Attention Backends

This document describes the diffusion attention backends available in vLLM-Omni, how to select them, and how to use SageAttention.

## Overview

Diffusion attention backend selection is controlled by the `DIFFUSION_ATTENTION_BACKEND` environment variable and resolved in `vllm_omni.diffusion.attention.selector`.

This backend is used by diffusion attention layers such as the DiT attention in video and image generation models.

On CUDA, the practical choices today are:

- `FLASH_ATTN`: FlashAttention backend. This is the default on supported CUDA systems when FlashAttention is installed.
- `TORCH_SDPA`: PyTorch `scaled_dot_product_attention`.
- `SAGE_ATTN`: SageAttention backend, if `sageattention` is installed.

If `DIFFUSION_ATTENTION_BACKEND` is unset, vLLM-Omni asks the current platform to choose the default backend. On CUDA, that normally means `FLASH_ATTN` when available, otherwise `TORCH_SDPA`.

## Backend Options

| Value | Notes |
|---|---|
| `FLASH_ATTN` | Default on CUDA when FlashAttention is available. Good default for most diffusion workloads. |
| `TORCH_SDPA` | Most conservative fallback. Useful for debugging or compatibility. |
| `SAGE_ATTN` | Requires `sageattention`. Can improve performance on some workloads, but output quality must be validated model-by-model. |

## Selection Priority

Diffusion attention backend selection follows this order:

1. `DIFFUSION_ATTENTION_BACKEND`
2. Platform default

Example:

```bash
export DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN
```

## SageAttention Installation

vLLM-Omni expects SageAttention to be installed into the same Python environment as vLLM-Omni.

Build from source:

```bash
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention

export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32
pip install . --no-build-isolation
```

Quick check:

```bash
python -c "import sageattention; print(sageattention.__file__)"
```

## Usage

### Enable SageAttention

Example: HunyuanVideo-1.5 text-to-video

```bash
DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN python examples/offline_inference/text_to_video/text_to_video.py \
    --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \
    --prompt "A dog running across a field of golden wheat." \
    --height 480 --width 832 --num-frames 33 \
    --num-inference-steps 30 --seed 42 --guidance-scale 6.0 \
    --tensor-parallel-size 2 \
    --output ../tmp/hv15_modelopt_sage.mp4
```

Example: Wan2.2 TI2V 5B

```bash
DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN python examples/offline_inference/text_to_video/text_to_video.py \
    --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --prompt "A dog running across a field of golden wheat." \
    --height 704 --width 1280 --num-frames 49 \
    --num-inference-steps 30 --seed 42 --guidance-scale 5.0 \
    --tensor-parallel-size 2 \
    --output outputs/wan22_sage.mp4
```

### Compare Against FlashAttention

Unset the backend override, or explicitly use `FLASH_ATTN`:

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
    --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --prompt "A dog running across a field of golden wheat." \
    --height 704 --width 1280 --num-frames 49 \
    --num-inference-steps 30 --seed 42 --guidance-scale 5.0 \
    --tensor-parallel-size 2 \
    --output outputs/wan22_fa3.mp4
```

## Validation Guidance

Do not assume that a faster attention backend is numerically interchangeable with `FLASH_ATTN`.

Always compare:

- End-to-end runtime
- DiT / diffusion stage runtime
- Output quality against a known-good baseline

At minimum, keep the same:

- model
- prompt
- seed
- resolution
- frame count
- inference steps
- parallel config
