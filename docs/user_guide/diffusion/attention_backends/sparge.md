# SPARGE Attention

`SPARGE_ATTN` is an Intel XPU diffusion attention backend based on the
training-free sparse attention algorithm from
[SpargeAttn](https://github.com/thu-ml/SpargeAttn). The backend calls ARK's
`sparge_sage2_attn_meansim_topk_xpu` operator from
[auto-round](https://github.com/intel/auto-round), rather than the upstream CUDA
extension.

The original algorithm was introduced in
[SpargeAttention: Accurate Sparse Attention Accelerating Any Model Inference](https://arxiv.org/abs/2502.18137)
(ICML 2025). The reference implementation is Apache-2.0 licensed at
[thu-ml/SpargeAttn](https://github.com/thu-ml/SpargeAttn).

## Upstream algorithm

SpargeAttn predicts a dynamic block-sparse mask for each attention call instead
of relying on a fixed pattern such as a sliding window or attention sink. The
original design combines two online filters with quantized attention:

1. **Selective block compression.** Q and K are divided into routing blocks.
   Each block is reduced to a mean token, but only when the tokens in that block
   are sufficiently self-similar. Mean pairwise cosine similarity is used as the
   gate, controlled upstream by `simthreshd1`.
2. **Block-mask prediction.** Scaled products of the compressed Q/K tokens form
   a small attention map. After softmax, each query block retains the highest
   scoring K blocks. Upstream supports either a fixed block fraction
   (`topk`) or a cumulative-probability threshold (`cdfthreshd`).
3. **Selective densification.** If a Q or K block fails the self-similarity
   test, its whole row or column is kept dense. This prevents the compressed
   mask from dropping important pairs where one mean token is not
   representative of the block.
4. **Sparse online softmax.** The original CUDA kernel skips a PV update when
   the local row maximum is far below the running softmax maximum. This
   second-stage filter is controlled upstream by `pvthreshd`.
5. **Quantized execution.** The selected blocks are evaluated with per-block
   INT8 Q/K quantization and SageAttention-style online softmax. V stays in
   FP16 or BF16 in this XPU integration.

### XPU implementation mapping

The ARK XPU operator implements the routing and top-k portion of this design,
but it does not expose every upstream tuning option:

- The backend passes `smooth_k=True` (subtract the per-head K sequence mean
  before INT8 quantization), `simthreshd1=-1.0`, and `attention_sink=False`.
  The negative similarity threshold makes full routing blocks pass the
  self-similarity gate, so `topk` controls selection uniformly instead of
  enabling the upstream dense fallback.
- Only fixed-fraction `topk` selection is configurable. `cdfthreshd` and
  `simthreshd1` are not part of the vLLM-Omni configuration.
- The ARK function accepts `pvthreshd` for signature compatibility but ignores
  it in the current XPU implementation, so the upstream sparse-PV filter is not
  available through this backend.
- Upstream's custom block-mask API documents 128-token Q blocks and 64-token K
  blocks. This backend exposes its own routing geometry through
  `q_block_tokens` and `k_block_tokens`; defaults are shown below.

This is an approximate attention backend. Validate image or video quality
against `TORCH_SDPA` at the same seed and step count before using it in
production.

## Requirements

- Intel XPU with a working PyTorch XPU build and Triton XPU runtime.
- `auto-round-lib` installed in the same environment as vLLM-Omni.
- FP16 or BF16 Q/K/V with `head_dim` 64 or 128.
- Query heads equal to KV heads, or a positive multiple of KV heads (MHA/GQA).
- No attention mask, packed variable-length input, paged KV cache, dropout, or
  ring sequence parallelism. Ulysses SP is not yet advertised for
  guaranteed-fullgraph execution.

Install the kernel package:

```bash
pip install auto-round-lib
python -c "import auto_round_kernel as ark; print(ark.sparge_sage2_attn_meansim_topk_xpu)"
```

Select it globally:

```bash
vllm-omni serve <model> --diffusion-attention-backend SPARGE_ATTN
```

## Configuration

SPARGE options live in the typed `sparge` block:

| Field | Default | Meaning |
| --- | --- | --- |
| `topk` | `0.5` | Fraction of logical K blocks retained per query block; must be in `(0, 1]`. The measured selected ratio can differ slightly because of routing geometry and boundary blocks. |
| `q_block_tokens` | `256` for `head_dim=128`, `64` for `head_dim=64` | Query-row granularity used for sparse routing. |
| `k_block_tokens` | `64` | K-block granularity; the current kernel requires `64`. |
| `q_tile_override` | Derived from `q_block_tokens`; see below | Query prefill tile size. |

Lower `topk` values increase sparsity and speed but reduce the number of key
blocks available to softmax. Start from the default and compare several values
on the target checkpoint.

When `q_tile_override` is omitted, the backend uses `0` for
`head_dim=128, q_block_tokens=64`, `256` for
`head_dim=128, q_block_tokens=256`, and `128` for
`head_dim=64, q_block_tokens=64`. An explicit override still takes precedence
and must remain compatible with the selected block size.

```bash
vllm-omni serve <model> \
  --diffusion-attention-config \
  '{"default":{"backend":"SPARGE_ATTN","sparge":{"topk":0.25}}}'
```

Long self-attention calls usually benefit most. Cross-attention often has a
short K/V sequence, where block selection overhead can outweigh the sparse
compute. Configure roles independently when a model exposes both:

```bash
vllm-omni serve <model> \
  --diffusion-attention-config \
  '{"per_role":{"self":{"backend":"SPARGE_ATTN",
  "sparge":{"topk":0.25,"q_block_tokens":256}},
  "cross":{"backend":"TORCH_SDPA"}}}'
```

The same configuration through the Python API:

```python
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec

attention = AttentionConfig(
    per_role={
        "self": AttentionSpec(backend="SPARGE_ATTN", sparge={"topk": 0.25}),
        "cross": AttentionSpec(backend="TORCH_SDPA"),
    }
)
```

## Accuracy validation

Run the same prompt, seed, scheduler, and denoising-step count once with
`SPARGE_ATTN` and once with `TORCH_SDPA`. Compare the decoded output or latent
tensors and tune `topk` upward if quality degrades. Setting `topk=1.0` retains
all candidate blocks and is useful as a loose kernel and layout sanity check,
but it is not a bit-exact replacement for dense attention.

The attention-level top-k sweep is available at
`benchmarks/kernels/bench_sparge_attn.py`. It compares the dense
`FLASH_ATTN` backend, the dense FlashAttention kernel, and the full SPARGE
backend for a configurable shape. Its default `B=1, S=32760, H=12, D=128`
shape matches Wan2.1-VACE 480P self-attention.

For common backend selection and per-role precedence, see the
[attention backend overview](../attention_backends.md).
