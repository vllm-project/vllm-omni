# Sol-Attn

`SOL_ATTN` integrates NVlabs' training-free sparse self-attention kernel.
It is an explicit opt-in and does not change the platform default.
This integration covers the attention kernel, not the full Sol-Engine stack.

## Installation and scope

Install the optional upstream package into the vLLM-Omni environment:

```bash
pip install 'git+https://github.com/NVlabs/Sana.git@sol-engine#subdirectory=techniques/sparse_backends'
```

Requirements:

- NVIDIA CUDA SM80 or newer, BF16 activations, and `head_dim=128`.
- Noncausal self-attention with matching Q/K/V shapes and equal head counts.
- One real sequence per attention call. MiniMax-H3's structural suffix padding
  is supported through producer-owned `PackedPaddingMetadata`; multiple real
  packed documents are not supported.
- Local attention or pure Ulysses (`--ring 1`). Ring and AllGather-KV are rejected.
- No arbitrary masks, GQA/MQA, paged KV, quantized KV, or compact reference-KV
  attention with unequal Q/KV lengths.

Architecture dispatch is delegated to upstream `sol_attn`, which provides
CuTe DSL kernels for Hopper SM90 and Blackwell SM100/SM120. Install the
[upstream runtime dependencies](https://github.com/NVlabs/Sana/blob/sol-engine/techniques/sparse_backends/README.md#requirements)
for these specialized kernels; upstream falls back to Triton when the optional
CuTe runtime is unavailable. The default `kv_splits="auto"` uses one split on
all architectures; explicit splits of 2 or 4 require the Hopper CuTe path.

Full-model compilation and outer HSDP boundaries are outside this backend's
compilation guarantee.

## MiniMax-H3 configuration

Add this configuration to the hardware profile in the
[MiniMax-H3 recipe](../../../../recipes/MiniMaxAI/MiniMax-H3.md):

```bash
--diffusion-attention-config '{
  "default": {
    "backend": "SOL_ATTN",
    "sol_attn": {
      "tau": 1.0,
      "dense_steps": 10,
      "dense_layers": "0,1",
      "prefix_sink": true,
      "kv_splits": 1
    }
  },
  "per_role": {
    "minimax_h3.token_refiner": {"backend": "TORCH_SDPA"}
  }
}'
```

`prefix_sink` derives the exact prefix from the model's video layout rather
than assuming a fixed number of audio/text tokens. For Ref2VA it conservatively
protects every row before the target-video span, including reference videos.
The prefix is kept exact as K/V and recomputed densely as queries. Larger
reference prefixes increase dense work.

## Other models

The backend is not tied to MiniMax-H3. Other models can opt in when their
attention calls meet the geometry, dtype and metadata requirements above.
Wan exposes separate `self` and `cross` roles, so select SOL only for
self-attention and leave cross-attention dense:

```bash
--diffusion-attention-config '{
  "default": {"backend": "TORCH_SDPA"},
  "per_role": {
    "self": {
      "backend": "SOL_ATTN",
      "sol_attn": {"dense_steps": 10, "dense_layers": "0,1"}
    }
  }
}'
```

Do not apply SOL globally to models that also use cross-attention, masks,
GQA, or incompatible head dimensions. Models must publish denoising-step
metadata and parseable layer prefixes for the default guards to permit sparse
execution.

## Controls

Set these fields in `AttentionSpec.sol_attn`, under `default` or `per_role`:

| Key | Default | Meaning |
| --- | --- | --- |
| `tau` | `1.0` | Finite nonnegative routing threshold multiplier; higher values generally select fewer exact blocks. |
| `thresh_type` | `diag` | Routing estimator, `diag` or `exact`; `exact` does not mean fully dense attention. |
| `dense_steps` | `10` | First N zero-based denoising steps stay dense. Missing step metadata stays dense when N is positive. |
| `dense_layers` | `"0,1"` | Nonnegative layer index, list, or range string such as `"0-3,8"`; empty/null disables it. Unidentified layers stay dense when this guard is enabled. |
| `prefix_sink` | `true` | Derive a protected prefix from `video_layout` when available; without layout, use the explicit sink settings. |
| `sink_tokens` | `0` | Additional exact K/V and dense-query sink length, clamped to valid rows. |
| `sink_start` | `0` | Sink offset; `null` selects the last `sink_tokens` valid rows. With `prefix_sink`, the union is conservatively covered by one contiguous range. |
| `kv_splits` | `auto` | `auto` uses the conservative single-split path. Explicit `1`, `2`, or `4`; `2/4` require SM90, CuTe DSL, and at least one 4096-token route group per split. |

`SOL_ATTN` without a `sol_attn` block uses these defaults. Schedules with at
most ten denoising steps remain entirely dense with `dense_steps=10`.

Sparse attention is approximate, not dense-equivalent. Validate video and audio
quality against a matching dense baseline before reducing the guards or
deploying a configuration. Dense guards use PyTorch SDPA (cuDNN on SM120/SM121),
which can also differ numerically from `FLASH_ATTN`.
