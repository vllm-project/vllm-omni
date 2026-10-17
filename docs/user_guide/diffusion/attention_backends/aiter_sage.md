# AITER SageAttention

`AITER_SAGE` runs SageAttention on AMD Radeon RDNA4 GPUs (gfx1201, for
example the Radeon AI PRO R9700) through
[aiter](https://github.com/ROCm/aiter)'s `gfx1201_sage_attention` operator.
On these GPUs the ROCm platform otherwise falls back to `TORCH_SDPA`, because
aiter FlashAttention targets gfx942 and gfx950 only.

The kernel computes INT8 QK^T and FP8 PV with FP32 accumulation. Q and K are
quantized per 32-row block and V per channel inside the kernel. Like
[`SAGE_ATTN`](sage.md), it is lossy: it is never selected by default, and you
should compare output quality against `TORCH_SDPA` at the same seed before
using it.

## Requirements

- ROCm platform with a gfx1201 GPU. Explicitly selecting `AITER_SAGE` on any
  other platform or architecture fails at startup with an actionable error.
- An aiter build that provides `aiter.gfx1201_sage_attention`
  ([ROCm/aiter#6390](https://github.com/ROCm/aiter/pull/6390)). The kernel is
  compiled by aiter's JIT on first use.

Check the installation:

```bash
python -c "import aiter; print(aiter.gfx1201_sage_attention)"
```

## Supported inputs

| Property | Supported |
| --- | --- |
| dtype | BF16 |
| Head size | 128 |
| Q/K/V shapes | Equal (self-attention, no GQA) |
| Causal / dropout | No |
| Masks | No `attn_mask`; a contiguous valid K/V prefix (`valid_kv_length`) is sliced instead, and padding rows return zeros |
| Packed multi-document batches | No |

Unsupported inputs raise an error naming the violated constraint instead of
falling back silently. Use [per-role configuration](../attention_backends.md#per-role-configuration)
to keep other attention calls on `TORCH_SDPA`.

## Usage

```bash
vllm serve <model> --omni --diffusion-attention-backend AITER_SAGE
```

## MiniMax-H3 on 2 x Radeon AI PRO R9700

The 2 x 32 GB TP2 profile from the [MiniMax-H3 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md)
(distributed layerwise offload, 20 resident DiT layers, eager), T2VA at
1344x768, 4 seconds (107 frames), 8 steps, seed 42. Timings are steady-state
requests after one warmup request in the same server.

| Backend | Denoise per step | Request latency | Video PSNR vs `TORCH_SDPA` |
| --- | ---: | ---: | ---: |
| `TORCH_SDPA` | 20.9 s | 186 s | reference |
| `AITER_SAGE` | 11.3 s | 109 s | 33.9 dB |
