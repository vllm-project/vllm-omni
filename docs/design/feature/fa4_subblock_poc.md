# SubBlock attention with FA4: proof of concept

This PoC adds shared block selection and an FA4 execution adapter on top of
[the attention execution contract (#7379)](https://github.com/vllm-project/vllm-omni/pull/7379).
The selector chooses logical KV blocks; the adapter executes attention over those
blocks. Dense attention remains the default.

The selector adapts the original
[SGLang MiniMax H3 SubBlock work (#34148)](https://github.com/sgl-project/sglang/pull/34148),
with native GQA routing and protected-prefix extensions. The shared adapter
interface lets other providers consume the same selection; separate drafts cover
[FlashInfer (#8336)](https://github.com/vllm-project/vllm-omni/pull/8336),
[TRTLLM (#8337)](https://github.com/vllm-project/vllm-omni/pull/8337), and
[cuDNN (#8338)](https://github.com/vllm-project/vllm-omni/pull/8338).

## Installation

Install the CUDA 13 extra from this checkout:

```bash
pip install '.[fa4]'
```

This pins the published `flash-attn-4[cu13]==4.0.0b33` wheel.

## Configuration

Pass the JSON contents of a recipe to `--diffusion-attention-config`:

| Recipe | Sparse role | Dense roles |
| --- | --- | --- |
| `recipes/attention/fa4-subblock.json` | `cosmos3.gen` | Understanding and multi-control |
| `recipes/attention/minimax-h3-fa4-subblock.json` | `minimax_h3.dit` | Token refiner |

Both recipes target Hopper and use dense `FLASH_ATTN` by default, 64×64 routing blocks, and
`target_sparsity: 0.75`. Sparse execution pins the `FLASH_ATTN` adapter with
`implementation: auto`, which uses FA4's sparse entry point. FA4 exposes no
kernel-ID selector. Provider preference lists are not supported.

The selected role uses sparse attention on every invocation. There is no layer
or denoising-step schedule. Exact roles take precedence over legacy `self`
overrides; Cosmos3 multi-control requires its own explicit sparse override.

### What 75% sparsity means

The target applies to **unprotected candidate blocks**. The retained candidate
budget rounds upward to eight and is capped by the candidate count. Protected
blocks are added to that budget, so overall sparsity can be lower than 75%.

Cosmos3 supplies its actual understanding KV-prefix length. Every block touching
that prefix is retained in full. For example, 301 blocks with nine protected
blocks retain 80 candidate blocks plus the nine protected blocks: 70.4% overall
block sparsity.

MiniMax currently supplies a zero protected prefix, so conditioning tokens can
also be sparsified. Protecting them requires the packer to expose their actual
boundary.

## Execution contract

Q/K/V use BSHD layout. `BlockSelection` contains contiguous device-local int32
`indices` shaped `[B, Hq, ceil(Sq / Bq), capacity]` and `counts` shaped
`[B, Hq, ceil(Sq / Bq)]`. Each row's active prefix contains sorted, unique,
in-range KV-block IDs; trailing storage is ignored.

The adapter may translate that representation, but must preserve the selected
blocks and geometry. It does not select additional blocks, expand K/V, or fall
back to another backend. FA4 supports native MHA, MQA, and GQA through this path.
The selector and installed kernel must both accept the requested geometry;
provider support is checked against the installed implementation rather than a
copied hardware/dtype/head-size allowlist.

Sparse provider selection follows the [platform selection contract](attention_backend_selection.md#registry-and-platform-boundary).

`BlockSparseAttention` owns routing, validation, and request preparation.
The first actual execution of each request signature runs and synchronizes the
kernel before recording support. Capability queries are read-only and report
preparation as required until that succeeds. Provider errors propagate.

An opaque custom op keeps request preparation and provider execution outside
Dynamo tracing, with a fake implementation for output shape inference. Eager
execution and ordinary `torch.compile`, including `fullgraph=True`, are covered.
This demonstrates the execution contract; it does not implement the entirety of
[the broader capability proposal (#7226)](https://github.com/vllm-project/vllm-omni/issues/7226).

Typed `PackedPaddingMetadata` supports a single document with suffix padding
at batch size one. Real Q/K/V lengths are trimmed before routing, and padded
output rows are restored as zeros. MiniMax supplies this metadata automatically.

## Current validation

Validation used the vLLM-Omni 0.30.0 image, published FA4 `4.0.0b33`,
PyTorch `2.13.0+cu130` and CuTe DSL `4.7.1`.

| GPU | Results | Sparse block size |
| --- | --- | --- |
| GH200 | 745 passed, 12 skipped | 64×64 |
| GB200 (SM100) | 84 passed, 6 skipped | 256×128 |

Hardware-specific tests account for all skips except two unavailable MiniMax
APIs on GH200. Native dense and sparse compilation checks passed on both GPUs.
For the tested Blackwell path, explicitly set `block_size: [256, 128]`;
the recipes' 64×64 geometry targets Hopper.

These runs validate correctness and compilation. Earlier quality/performance
measurements used a development FA4 revision on Hopper and do not establish
results for this wheel or Blackwell's different selection granularity.

```bash
python -m pytest -q \
  tests/diffusion/attention/test_block_selection.py \
  tests/diffusion/attention/test_block_sparse.py \
  tests/diffusion/attention/test_block_sparse_owner.py \
  tests/diffusion/attention/test_block_sparse_adapters.py \
  tests/diffusion/attention/test_block_sparse_ops.py \
  tests/diffusion/attention/test_flash_attn.py \
  tests/diffusion/attention/test_flash_attn_compile.py \
  tests/diffusion/attention/test_attention_config.py \
  tests/diffusion/attention/test_selector.py \
  tests/diffusion/models/cosmos3/test_cosmos3_transformer.py \
  tests/diffusion/models/minimax_h3/test_minimax_h3_contract.py \
  tests/diffusion/models/minimax_h3/test_minimax_h3_fasth3.py \
  tests/diffusion/models/minimax_h3/test_fasth3_checkpoint.py \
  tests/diffusion/diffusion_backend/test_diffusers_backend.py
```

Coverage includes selected-key numerical references, MHA/MQA/GQA, tails,
prefixes, padding, owner isolation, dynamic/fullgraph compilation, model-role
integration and dense regressions.

For an attention-call benchmark, including routing and metadata conversion:

```bash
python benchmarks/diffusion/fa4_subblock.py --capture /path/to/capture.pt \
  --sparsity 0.75 --block-size 64 64 --iterations 30 --output /tmp/fa4-subblock.json
```

Captures contain `q`, `k`, `v`, and optional `meta.scale` and `meta.prefix_len`.
This measures attention calls; it does not establish checkpoint-level role
activation, end-to-end speedup, or video quality. Historical timings are omitted
because they have not been refreshed for the current implementation.

## Limitations and follow-up

Arbitrary masks, causal sparse attention, multi-document packing,
paged or quantized KV, and sequence-parallel wrappers are unsupported. Diffusers
rejects sparse configurations before loading weights; sparse execution needs a
native model integration.

Checkpoint generation quality and end-to-end latency remain unqualified.
A separate attention-strategies RFC will follow for mixing dense and sparse
attention across roles, layers, and denoising steps to improve quality.
