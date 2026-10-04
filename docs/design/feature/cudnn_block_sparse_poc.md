# cuDNN selected-block adapter

This adapter adds sparse execution to `CUDNN_ATTN` using
`cudnn.BSA.block_sparse_attention_forward`. It builds on the
[shared foundation (#8335)](https://github.com/vllm-project/vllm-omni/pull/8335).
See the [shared contract](fa4_subblock_poc.md) for selection, protected prefixes,
padding, and request preparation.

## Provider mapping

The adapter forwards per-query-head block IDs/counts with `pack_gqa=False`,
preserving native MHA/MQA/GQA patterns. It supplies valid lengths for partial KV
blocks and requests BSHD input/output layout.

Q/KV block sizes must be equal because the API accepts one scalar block size.
Only `implementation: auto` is accepted because there is no kernel-ID parameter.
The installed provider validates device, dtype, and numeric geometry support.
Errors propagate without padding queries, retiling, merging patterns, expanding
K/V, or falling back to dense attention.

An opaque custom op encloses the provider call. Request metadata and workspaces
remain invocation-local; only dependency callables are cached.

## Configuration

For example, select cuDNN sparse attention for Cosmos3 generation while keeping
dense Flash Attention as the default:

```json
{
  "default": {"backend": "FLASH_ATTN"},
  "per_role": {
    "cosmos3.gen": {
      "name": "block_sparse",
      "config": {
        "block_size": [64, 64],
        "backend": {"require": "CUDNN_ATTN", "implementation": "auto"}
      }
    }
  }
}
```

Actual model geometry must pass provider preparation.

## Validation

GH200 validation uses the vLLM-Omni 0.30.0
image, PyTorch `2.13.0+cu130`, FA4 `4.0.0b33` and CuTe DSL `4.7.1`.
cuDNN frontend is `1.29.0`.

**52 passed, 0 skipped**.

```bash
python -m pytest -q \
  tests/diffusion/attention/test_selector.py \
  tests/diffusion/attention/test_block_sparse_adapters.py \
  tests/diffusion/attention/test_cudnn_sparse_adapter.py
```

Real-kernel tests cover selected-key numerics, changing patterns and shapes,
tails, inactive entries, prefixes, padding and multiple owners under
`torch.compile(fullgraph=True, dynamic=True)`. Platform selection and shared
FA4 regressions are included.
Dynamic cuDNN cases use query lengths divisible by 64 on SM90.

These results do not establish checkpoint quality or end-to-end speedups.
