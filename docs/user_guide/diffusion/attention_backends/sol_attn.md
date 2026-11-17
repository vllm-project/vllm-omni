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

The external kernel is wrapped in a PyTorch custom op with a fake implementation
for compilation. Dense guard steps and sink queries use PyTorch SDPA, pinned to
cuDNN on SM120/SM121. On other CUDA architectures SDPA selects its dense kernel.
Both paths trim structural padding first and return zero for discarded padding
rows. These dense kernels can differ numerically from `FLASH_ATTN`.
An outer HSDP boundary is not covered by the backend's fullgraph guarantee.

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
The prefix is kept exact as K/V and recomputed densely as queries; a K/V-only
sink does not protect generated audio queries. A larger reference prefix costs
more dense work, so measure reference-heavy workloads separately.

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

The CUDA regression suite exercises this configuration through a small Wan
transformer with synthetic weights, checking both dense guards and actual
sparse-kernel dispatch. This is forward-path compatibility coverage, not
pretrained-model quality or end-to-end speed evidence.

Do not apply SOL globally to models that also use cross-attention, masks,
GQA, or incompatible head dimensions. Models must publish denoising-step
metadata and parseable layer prefixes for the default guards to permit sparse
execution. Missing metadata keeps guarded calls dense, and schedules with
at most ten denoising steps remain dense with the default `dense_steps=10`.

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

`SOL_ATTN` without a `sol_attn` block uses these defaults. Use per-role selection
for models with unsupported cross-attention or other head dimensions.

## Validation and quality

Sparse attention changes outputs. Compare video and audio against the dense
baseline on the same prompts, seeds, resolution, duration, and denoising schedule.
Short distilled schedules can remain entirely dense with `dense_steps=10`;
change this only after validating quality.

Warm up both dense and sparse branches using a complete request before timing.
Report repeated latency, variation, peak GPU memory, video metrics, and audio
quality on the tested revision. The
[original-H3 E2E report](../../../../benchmarks/diffusion/sol_attn_h3_e2e_results.md)
records a matched dense/SOL comparison and its outstanding perceptual checks.
The old
[SM120 sweep](../../../../benchmarks/diffusion/sol_attn_sm120_results.md)
predates the current sink and metadata handling and is historical evidence only.

CPU contract regressions:

```bash
pytest -q tests/diffusion/attention/test_sol_attn_backend.py -m 'core_model and cpu'
```

With the optional kernel installed and one SM80+ GPU, verify the real kernel
and fullgraph compilation, including non-aligned padding and dense sink rows:

```bash
pytest -q tests/diffusion/attention/test_sol_attn_cuda.py
```

With two visible GPUs the same file also compares pure Ulysses execution
against local attention, including prefix protection and structural padding.
That case skips when fewer than two GPUs are available.
