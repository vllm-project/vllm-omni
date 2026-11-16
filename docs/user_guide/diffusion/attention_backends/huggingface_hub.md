# Hugging Face Hub Attention Backends

Hub backends load the same published kernels commonly used with Hugging Face
Diffusers. They are useful when training and serving must use matching kernel
implementations to reduce numerical drift and sampling divergence.

| Backend | Kernel | Platform |
| --- | --- | --- |
| `FLASH_ATTN_HUB` | `kernels-community/flash-attn2` | Compatible CUDA GPUs |
| `FLASH_ATTN_3_HUB` | `kernels-community/flash-attn3` | Hopper sm_90 or newer |

## Installation

```bash
pip install kernels==0.16.1
```

If `kernels` is unavailable, an explicit `FLASH_ATTN_HUB` or
`FLASH_ATTN_3_HUB` selection raises. Automatic / `FLASH_ATTN` selection in the
Diffusers adapter may still walk hub then local FlashAttention backends.

For FlashAttention 2, the loader tries Hub API versions 1, 2, then 3.
Version 3 provides Torch stable-ABI CUDA builds compatible with the Torch 2.14
runtime used by vLLM 0.33. An explicit Hub selection raises if none of these
versions has a compatible build.

## Usage

```bash
export DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN_HUB

# Equivalent CLI selection
vllm-omni serve <model> --diffusion-attention-backend FLASH_ATTN_HUB
```

For per-role kernel alignment, configure the Hub backend on only the relevant
role; see the [attention backend overview](../attention_backends.md#per-role-configuration).
