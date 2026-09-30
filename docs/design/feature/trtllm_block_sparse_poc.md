# TRTLLM selected-block adapter

This adapter adds sparse plumbing to `TRTLLM_ATTN` using FlashInfer's
`flashinfer.attention.prims_ts.BlockSparseTSWrapper`. It builds independently on
the [shared foundation (#8335)](https://github.com/vllm-project/vllm-omni/pull/8335).
See the [shared contract](fa4_subblock_poc.md) for selection and preparation.
The existing dense TRTLLM API alone is insufficient for this sparse path.

## Provider mapping

Selected IDs/counts become UInt32 exact-block bits. Inactive entries are ignored;
logical blocks and actual sequence lengths are preserved. The API mapping
requires MHA and equal Q/K/V head dimensions. It cannot represent arbitrary
per-query-head GQA selections without changing their meaning.

Only `implementation: auto` is accepted because the wrapper has no public
kernel-ID parameter. The installed provider validates device, dtype, and numeric
geometry support. Errors propagate without fallback, merging selections, or
expanding K/V.

Each invocation owns its wrapper, plan, and route workspace. An opaque custom op
encloses conversion, planning, and execution. Future latency measurements must
include allocation and planning costs.

## Configuration

`recipes/attention/minimax-h3-trtllm-subblock.json` selects sparse MiniMax DiT
with dense `FLASH_ATTN` for other roles.

## Validation

Native GB200 validation uses FlashInfer Python/cubin `0.7.0` and JIT cache
`0.7.0+cu130`. Both runs use the vLLM-Omni 0.30.0 image, PyTorch
`2.13.0+cu130`, FA4 `4.0.0b33` and CuTe DSL `4.7.1`.

| GPU | Results | Sparse execution |
| --- | --- | --- |
| GB200 | 98 passed, no skips | Native FlashInfer `0.7.0` |
| GH200 | 87 passed, 11 Blackwell-only skips | Reference provider; FlashInfer `0.6.18.post1` |

Native tests cover selected-key correctness and dynamic fullgraph execution
across changing patterns/shapes and multiple owners.

```bash
python -m pytest -q \
  tests/diffusion/attention/test_selector.py \
  tests/diffusion/attention/test_block_sparse_adapters.py \
  tests/diffusion/attention/test_trtllm_sparse_adapter.py \
  tests/diffusion/attention/test_trtllm_attn.py \
  tests/diffusion/attention/test_trtllm_calibration.py
```

These results do not establish checkpoint quality or end-to-end speedups.
