# FlashInfer selected-block adapter

This adapter adds sparse execution to `FLASHINFER_ATTN` using
`VariableBlockSparseAttentionWrapper`. It builds on the
[shared block-sparse foundation (#8335)](https://github.com/vllm-project/vllm-omni/pull/8335).
See the [shared contract](fa4_subblock_poc.md) for selection, protected prefixes,
padding, and request preparation.

## Provider mapping

Selected IDs/counts become a block mask with exact row/column lengths. The
mapping preserves independent batch/head patterns, logical blocks, and sequence
tails. It requires MHA and equal Q/K/V head dimensions: the provider's shared
KV-head patterns cannot represent arbitrary per-query-head GQA selections.
Cosmos3 GQA is therefore unsupported by this adapter.

`implementation` passes unchanged to the provider's `backend` argument, including
`auto`, `fa2`, and `fa3`. The installed provider validates device, dtype, and
numeric geometry support. Errors propagate without fallback, merging selections,
or expanding K/V.

Each invocation creates its own wrapper, plan, and 128 MiB workspace for the
current pattern. Planning, allocation, and layout conversion costs must be
included in performance measurements. An opaque custom op encloses execution.

## Configuration

Pass `recipes/attention/minimax-h3-flashinfer-subblock.json` to the attention
configuration option as JSON. It selects sparse MiniMax DiT with dense
`FLASH_ATTN` token refinement.

## Validation

GH200 validation uses the vLLM-Omni 0.30.0
image, PyTorch `2.13.0+cu130`, FA4 `4.0.0b33` and CuTe DSL `4.7.1`.
FlashInfer is `0.6.18.post1`.

**65 passed, 0 skipped**.

```bash
python -m pytest -q \
  tests/diffusion/attention/test_selector.py \
  tests/diffusion/attention/test_block_sparse_adapters.py \
  tests/diffusion/attention/test_flashinfer_sparse_adapter.py \
  tests/diffusion/attention/test_flashinfer_attn.py
```

Real-kernel tests cover selected-key numerics, changing patterns and shapes,
tails, inactive entries, prefixes, padding and multiple owners under
`torch.compile(fullgraph=True, dynamic=True)`. Platform selection and shared
FA4 regressions are included.

These results do not establish checkpoint quality or end-to-end speedups.
