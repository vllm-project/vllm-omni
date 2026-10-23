# AITER Sol-Attn

`AITER_SOL` runs Sol-Attn block-sparse attention on AMD Radeon RDNA4 GPUs
(gfx1201, for example the Radeon AI PRO R9700) through
[aiter](https://github.com/ROCm/aiter)'s `gfx1201_sol_attention` operator. It
builds on [`AITER_SAGE`](aiter_sage.md): selected key blocks are computed
with the same INT8 QK^T / FP8 PV kernel, and the remaining blocks enter
through a block-summary correction.

Per 64-row query block, key block `j` is computed exactly if it passes the
Sol-Attn `diag` threshold
`qc . kc_j > qc . mu + tau * sqrt(sum(qc^2 * var) + 1e-6)` (`qc`/`kc` are
block-mean queries and keys, `mu`/`var` the per-dimension mean and variance
of `kc`), lies on the diagonal band, or overlaps the sink. Larger `tau` is
sparser. `AITER_SOL` is lossy and never selected by default. Compare output
quality against `TORCH_SDPA` at the same seed before using it.

## Requirements

- Everything `AITER_SAGE` requires.
- An aiter build that provides `aiter.gfx1201_sol_attention`.

```bash
python -c "import aiter; print(aiter.gfx1201_sol_attention)"
```

## Configuration

`sol_attn` sets the typed config of an `AITER_SOL` role:

| Field | Default | Meaning |
| --- | --- | --- |
| `tau` | `1.0` | Routing threshold in standard deviations (`>= 0`; larger is sparser) |
| `thresh_type` | `"diag"` | Routing rule; only `"diag"` is supported |
| `prefix_sink` | `true` | Use the non-video prefix the model publishes in `AttentionMetadata.video_layout` (text, conditions, audio) as the sink, computed exactly as keys and as queries |
| `sink_tokens` / `sink_start` | `0` / `0` | Static key sink `[sink_start, sink_start + sink_tokens)`, used when no layout prefix applies (`sink_start: null` places it at the end) |
| `dense_steps` | `10` | Denoise steps before this index run the dense `AITER_SAGE` kernel |
| `dense_layers` | `"0,1"` | DiT blocks (index selector such as `"0-1,38"`) that always run dense |

Sparse attention is applied only to the self-attention role you select, for
example:

```bash
vllm serve <model> --omni --diffusion-attention-backend AITER_SAGE \
  --diffusion-attention-config '{"per_role": {"self": {"backend": "AITER_SOL",
    "sol_attn": {"tau": 1.5, "dense_steps": 2, "dense_layers": "0"}}}}'
```

Supported inputs are the same as for [`AITER_SAGE`](aiter_sage.md#supported-inputs).

## MiniMax-H3 on 2 x Radeon AI PRO R9700

Same setup as the [`AITER_SAGE` measurement](aiter_sage.md#minimax-h3-on-2-x-radeon-ai-pro-r9700)
(T2VA 1344x768, 4 seconds, 8 steps, seed 42, steady-state requests). `AITER_SOL` is
used for `self` attention with `tau=1.5`, `dense_steps=2` and `dense_layers="0"`;
all other attention stays on `AITER_SAGE`.

| Backend | Denoise per step | Request latency | Video PSNR vs `TORCH_SDPA` | Audio cosine vs `TORCH_SDPA` |
| --- | ---: | ---: | ---: | ---: |
| `TORCH_SDPA` | 20.9 s | 186 s | reference | reference |
| `AITER_SAGE` | 11.3 s | 109 s | 33.9 dB | 0.950 |
| `AITER_SOL` | 10.0 s | 100 s | 29.4 dB | 0.865 |
