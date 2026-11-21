# Sol-Attn

`SOL_ATTN` is an opt-in, training-free sparse attention backend. By default, H3
uses it throughout denoising while keeping text/audio blocks exact. Selecting it does
not change the checkpoint, sampling steps, precision, or platform defaults.

Sol-Attn is by **Haopeng Li et al.** ([paper](https://arxiv.org/abs/2607.24027),
[upstream implementation](https://github.com/NVlabs/Sana/tree/f4035e0a976d828ed41c7b0b799e9712b47cd60d/techniques/sparse_backends)).
This backend integrates their algorithm; it does not require VSA-trained weights.

## Requirements

Install the pinned Sol-Attn package with arbitrary exact-block protection
and automatic SM103 dispatch:

```bash
pip install 'git+https://github.com/lishunyang12/Sana.git@d9ef6321230cff13398cfc86aa107f97f093e925#subdirectory=techniques/sparse_backends'
```

This pin contains the generic `exact_block_mask` extension and SM103 dispatch
on top of upstream
`f4035e0a`; it is a feature branch pending
[upstream review](https://github.com/NVlabs/Sana/pull/593). Older releases with
only a contiguous KV sink cannot preserve H3's disjoint text/audio Q/KV protection
and fail dependency preflight.

The upstream package selects its CuTe kernel on SM90/SM100/SM103/SM120 when the
CuTe DSL runtime is available, and its Triton implementation otherwise (SM80+).
SM103 (B300/GB300) uses the upstream SM100 kernel family automatically.
Omni calls that implementation directly; no `provider` or `native` flag is used,
and no cuDNN BSA adapter is involved. Attention requires BF16 self-attention with
head dimension 128 and equal Q/KV heads. It supports one sequence at a time,
locally or with pure Ulysses; Ring, AllGather-KV, and quantized KV are unsupported.

## Quick start

Start T2VA or FL2VA with native checkpoints:

```bash
vllm serve /path/to/MiniMax-H3/FL2VA --omni \
  --task-type fl2va \
  --enforce-eager --vae-use-tiling \
  --diffusion-attention-backend SOL_ATTN
```

No JSON or model-specific attention settings are required. H3 supplies a GPU
block mask for all text/audio segments, including disjoint Ref2VA references.
The upstream fused kernel keeps those Q rows and KV columns exact; Omni does
not compute a separate omitted-block residual.

- **T2VA / FL2VA:** set request `extra_params.task` to `t2va` or `fl2va`.
- **Ref2VA:** replace `/FL2VA` with `/Ref2VA` and `--task-type fl2va` with
  `--task-type ref2va`; set request `extra_params.task` to `ref2va`.

For memory requirements, parallelism, and HTTP requests, see the
[MiniMax-H3 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/MiniMaxAI/MiniMax-H3.md).

## Advanced settings

To override defaults, replace the quick-start backend flag with:

```bash
--diffusion-attention-config '{"default":{"backend":"SOL_ATTN","sol_attn":{"tau":1.5}}}'
```

| Key | Default | Meaning |
| --- | --- | --- |
| `tau` | `1.0` | Controls sparsity. Higher values generally compute fewer blocks exactly; check output quality. |
| `tau_profile` | `""` | Per-block overrides, e.g. `"0-30=2.0;39-42=0.9"`. Unlisted blocks keep `tau`; later matching entries win. |
| `dense_blocks` | `""` | Blocks that always use dense attention, e.g. `"0-2,-1"` for the first three and last blocks. Overrides sparsity settings. |
| `min_tokens` | `0` | Use dense attention below this valid sequence length, excluding structural padding. Equality uses SOL unless another dense guard applies. |
| `sink_conditioning` | `exact_kv_and_rows` | Protect model-marked conditioning blocks on both Q and KV axes. `off` explicitly disables this protection. With no conditioning mask and no model requirement, SOL uses ordinary self-attention. |

The threshold estimator and KV splitting are internal kernel policy. The
implementation uses the diagonal estimator, with four KV splits on SM90 for
at least 65,536 tokens when CuTe is available, and one split otherwise.
They are not public configuration fields. Select dense denoising steps with
[attention schedules](schedules.md), and select dense layers with `dense_blocks`.

`sink_conditioning="exact_kv_and_rows"` means all queries read the marked
K/V blocks token by token, and queries in marked blocks attend densely to all
K/V. For H3, the model supplies disjoint text/audio blocks, including those in
Ref2VA references. Protection is block64-granular. It applies to the current
layer's Q/K/V; approximation in other layers can still change the final video
or audio. Preserve the default protection when comparing H3 speed and quality.
This is H3's default policy, not a universal requirement: audio tokens may also
be generation targets. `off` allows an explicit speed/quality ablation. Required
metadata is validated only while protection is enabled; this validation does
not certify output quality.
The current kernel cannot protect only KV columns: `exact_kv` is rejected
explicitly rather than being mapped to a different computation.

`dense_blocks` and selectors inside `tau_profile` accept inclusive ranges,
negative indices relative to the model's block count, and reversed ranges.
For a 50-block model, `-1` means block 49 and `-3--1` means blocks 47–49.
Ranges only affect existing blocks. Negative indices require model-owned block
counts; H3 supplies separate counts for its DiT and token refiner. Missing
model block indices or required counts fail explicitly instead of ignoring settings.

Separate `tau_profile` entries with semicolons or newlines; `#` starts a
comment within an entry. Values must be finite and non-negative. Selectors
and tau overrides are resolved once at layer construction, including scheduled
profiles, so no strings are parsed during denoising.

For example, this configuration keeps three early blocks and the last block
dense, applies a lower tau to blocks 39–42, and skips sparse execution below
4096 valid tokens:

```bash
--diffusion-attention-config '{
  "default": {
    "backend": "SOL_ATTN",
    "sol_attn": {
      "tau": 1.0,
      "tau_profile": "39-42=0.8",
      "dense_blocks": "0-2,-1",
      "min_tokens": 4096
    }
  }
}'
```

These values illustrate the API; they are not a validated speed/quality preset.
Higher tau increases approximation, while more dense blocks or steps generally
reduce it. Neither setting guarantees lower latency. Short sequences can cost
more to route than they save in attention computation.

### Enable SOL during part of denoising

Use the existing [attention schedule](schedules.md#sigma-windows) to keep
dense attention outside a sigma window. `start_percent` and `end_percent` are
not SOL backend fields. Percentages need the model scheduler's noise mapping;
they are not interchangeable with fractions of the executed step count.

For H3's shifted flow trajectory, the conversion is
`sigma = shift * (1 - percent) / (1 + (shift - 1) * (1 - percent))`.
With video `flow_shift=12`, percentages 0.2 and 0.9 correspond to sigma
approximately 0.979592 and 0.571429. A matching window can be configured as:

```bash
--diffusion-attention-backend TORCH_SDPA \
--diffusion-attention-schedule '{
  "profiles": {
    "sparse": {
      "default": {
        "backend": "SOL_ATTN",
        "sol_attn": {"tau": 1.0, "min_tokens": 4096, "dense_blocks": "0-2,-1"}
      }
    }
  },
  "sigma": [{"low": 0.5714285714285714, "high": 0.9795918367346939, "profile": "sparse"}]
}'
```

Windows use `[low, high)`, except a high bound of 1 includes 1. Recompute the
bounds when changing flow shift; do not reuse this conversion for other
schedulers. Existing integer step ranges remain available when the intention
is to select particular iterations rather than noise levels. A dense layer or short-sequence guard within the selected SOL profile
still wins.

### Model integration contract

H3 is the model adapter added and validated by this change. Shared SOL code
uses model-independent metadata; it does not inspect model names or infer
conditioning from token order.

A model already using shared `Attention` needs no SOL-specific implementation
for an eligible, unconditioned self-attention call. A model with protected
conditioning supplies `AttentionMetadata.conditioning_block_mask`. Build it
once during request preparation, using valid tokens before sequence sharding:

```python
from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionMetadata,
    ConditioningBlockMask,
)

# A one-dimensional Boolean tensor on the attention device, excluding padding.
protection = ConditioningBlockMask.from_token_mask(conditioning_tokens)
metadata = AttentionMetadata(conditioning_block_mask=protection)
```

`ConditioningBlockMask` also accepts an already prepared Boolean block vector.
It describes protection, not token visibility: it must never replace a causal
or arbitrary attention mask. The model retains ownership of its packing,
visibility rules and structural padding metadata.

Models pass their stack's `layer_count` to `Attention` for negative block
selectors. Stacks that require conditioning protection also set the internal
constructor argument `requires_conditioning_mask=True`; missing protection
then fails unless the user explicitly selects `sink_conditioning="off"`.
H3 supplies both arguments, including its separate token-refiner depth. These
are model contracts, not additional service configuration fields. Base and
scheduled implementations receive the same contracts.

Other architectures have not been certified by this adapter change. The kernel
still requires BF16, head dimension 128, a single mask-free noncausal sequence
with equal Q/KV lengths and heads, and local attention or pure Ulysses. Chunk
history, arbitrary visibility masks, paged KV and other head dimensions need
appropriate backend support before model integration.

### Precision and diagnostics

This backend uses BF16. `int8_qk`, `int8_pv`, `morton`, `morton_curve`,
`use_tma`, `verbose`, and `probe` are not accepted SOL
configuration fields. Unknown keys fail configuration parsing. Kernel dispatch
is automatic; no `native` or `provider` flag is needed.

Changing `sink_conditioning` changes which tokens remain exact;
it is not a precision switch. Per-block attention error diagnostics and
decoded video SSIM/PSNR/LPIPS measure different things. Any diagnostic that
runs both dense and sparse attention must be disabled for latency measurements.

## Validation limits

Full-model video/audio quality, HTTP serving, and speedup for the default
profile remain unvalidated. More sparsity does not guarantee a faster request;
compare against a matching dense baseline before deployment. Full-model
compilation and HSDP compatibility are not guaranteed.
