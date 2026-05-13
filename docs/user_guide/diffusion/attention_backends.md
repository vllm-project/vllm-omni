# Diffusion Attention Backends

This document describes the diffusion attention backends available in vLLM-Omni, how to select them globally and per-role, and how to use SageAttention.

## Overview

Diffusion attention backend selection is resolved in `vllm_omni.diffusion.attention.selector`. It looks up the backend from a structured `AttentionConfig` carried on `OmniDiffusionConfig` and falls back to the platform default when nothing is configured.

This backend is used by diffusion attention layers such as the DiT attention in video and image generation models.

On CUDA, the practical choices today are:

- `FLASH_ATTN`: FlashAttention backend. This is the default on supported CUDA systems when FlashAttention is installed.
- `TORCH_SDPA`: PyTorch `scaled_dot_product_attention`.
- `SAGE_ATTN`: SageAttention backend, if `sageattention` is installed.

If no attention backend is configured, vLLM-Omni asks the current platform to choose the default. On CUDA, that normally means `FLASH_ATTN` when available, otherwise `TORCH_SDPA`.

## Backend Options

| Value | Notes |
|---|---|
| `FLASH_ATTN` | Default on CUDA when FlashAttention is available. Good default for most diffusion workloads. |
| `TORCH_SDPA` | Most conservative fallback. Useful for debugging or compatibility. |
| `SAGE_ATTN` | Requires `sageattention`. Can improve performance on some workloads, but output quality must be validated model-by-model. |

## Configuration

Diffusion attention backends can be configured three ways, in priority order:

1. **`--diffusion-attention-config`** — structured per-role config (highest priority).
2. **`--diffusion-attention-backend` / `DIFFUSION_ATTENTION_BACKEND` env var** — global shorthand that sets the default backend.
3. **Platform default** — used when nothing is configured.

`--diffusion-attention-backend` is shorthand for `--diffusion-attention-config.default.backend`. It may be combined with `--diffusion-attention-config.per_role.*` overrides, but is mutually exclusive with `--diffusion-attention-config.default.backend`.

### Global default

Set the default backend for every diffusion attention layer:

```bash
# CLI flag
vllm-omni serve <model> --diffusion-attention-backend SAGE_ATTN

# Environment variable (also recognized for backwards compatibility)
export DIFFUSION_ATTENTION_BACKEND=SAGE_ATTN
```

### Per-role configuration

Roles are free-form strings declared by each diffusion model. The two common categories are `"self"` and `"cross"`; model-specific roles (e.g. `"ltx2.audio_to_video"`) may also be declared. A role string is matched in this order:

1. Exact `per_role[role]` match
2. `per_role[role_category]` fallback (e.g. `"ltx2.audio_to_video"` → `"cross"`)
3. `default`
4. Platform default

Use vLLM-style dotted flags or one JSON blob:

```bash
# Dotted flags
vllm-omni serve <model> \
    --diffusion-attention-config.default.backend FLASH_ATTN \
    --diffusion-attention-config.per_role.cross.backend TORCH_SDPA

# JSON
vllm-omni serve <model> \
    --diffusion-attention-config '{"default":{"backend":"FLASH_ATTN"},"per_role":{"cross":{"backend":"TORCH_SDPA"}}}'
```

Backends may also accept backend-specific parameters via `extra`:

```bash
--diffusion-attention-config.per_role.self.backend SPARSE_BLOCK \
--diffusion-attention-config.per_role.self.extra.block_size 128
```

### Programmatic API

When constructing `OmniDiffusionConfig` directly:

```python
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, OmniDiffusionConfig

config = OmniDiffusionConfig(
    attention_config=AttentionConfig(
        default=AttentionSpec(backend="FLASH_ATTN"),
        per_role={
            "cross": AttentionSpec(backend="TORCH_SDPA"),
        },
    ),
    ...,
)
```

A plain dict is also accepted and normalized to `AttentionConfig`.

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

### Mixed backends across roles

Use `FLASH_ATTN` for self-attention and `TORCH_SDPA` for cross-attention:

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
    --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --prompt "A dog running across a field of golden wheat." \
    --diffusion-attention-config.per_role.self.backend FLASH_ATTN \
    --diffusion-attention-config.per_role.cross.backend TORCH_SDPA \
    --tensor-parallel-size 2 \
    --output outputs/wan22_mixed.mp4
```

### Compare against FlashAttention

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
