# SPARGE Attention

`SPARGE_ATTN` exposes the ARK SPARGE top-k sparse attention kernel from
[auto-round](https://github.com/intel/auto-round) as a diffusion attention
backend on Intel XPU. The kernel quantizes Q and K to INT8, selects the most
relevant key blocks for each query block, and computes attention only on those
blocks. V and the output remain FP16 or BF16.

This is an approximate attention backend. Validate image or video quality
against `TORCH_SDPA` at the same seed and step count before using it in
production.

## Requirements

- Intel XPU with a working PyTorch XPU build and Triton XPU runtime.
- `auto-round-lib` installed in the same environment as vLLM-Omni.
- FP16 or BF16 Q/K/V with `head_dim` 64 or 128.
- Query heads equal to KV heads, or a positive multiple of KV heads (MHA/GQA).
- No attention mask, packed variable-length input, paged KV cache, dropout, or
  sequence-parallel attention.

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
| `topk` | `0.5` | Fraction of logical K blocks retained per query block; must be in `(0, 1]`. |
| `q_block_tokens` | `256` for `head_dim=128`, `64` for `head_dim=64` | Query-row granularity used for sparse routing. |
| `k_block_tokens` | `64` | K-block granularity; the current kernel requires `64`. |
| `q_tile_override` | `256` for `head_dim=128`, `128` for `head_dim=64` | Query prefill tile size. |

Lower `topk` values increase sparsity and speed but reduce the number of key
blocks available to softmax. Start from the default and compare several values
on the target checkpoint.

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
  "sparge":{"topk":0.25,"q_block_tokens":256,"q_tile_override":256}},
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

For common backend selection and per-role precedence, see the
[attention backend overview](../attention_backends.md).
