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
These are experimental execution examples; 75% is not a quality-qualified default.

### Composition with scheduling and sparse plugins

The current configuration selects either an `AttentionSpec` (a complete backend)
or a `BlockSparseAttentionSpec` (shared selection plus a provider-owned execution
adapter) for each role. `AttentionSpec.block_sparse` is an existing typed option
for complete backends such as RainFusion; it is not an alias for
`BlockSparseAttentionSpec`. Mixed `backend` and `name/config` fields are rejected.

[RFC #8382](https://github.com/vllm-project/vllm-omni/issues/8382) proposes
`AttentionSpec.sparse` with `backend`, `start_step`, and `options`. That field is
not implemented here. The proposed composition, to agree before stabilizing the
public API, is:

- Normalize the step policy into a dense method, a sparse method, and one shared
  schedule. The sparse method may be a complete backend or this selected-block
  method. Reuse `BlockSparseAttentionSpec` for the latter's geometry, selector,
  and provider options; do not introduce a second set of equivalent knobs.
- Resolve role precedence once, then resolve both methods through platform-owned
  dispatch. A provider plugin supplying a selected-block adapter uses the existing
  backend registry and `get_block_sparse_adapter()` contract. A complete sparse
  method retains its own backend contract and is not assumed to accept
  `BlockSelection`. No second provider registry is needed.
- Keep scheduling outside the selector and kernel adapter. Exactly one policy
  owns logical denoising progress; reject combinations with an independent backend
  schedule unless that backend exposes an explicitly externally scheduled mode.
  Sparse steps must not silently fall back to dense execution.
- Preserve model-facing metadata capability queries for every reachable method.
  Require metadata valid for both phases (or model-owned phase-specific metadata)
  rather than advertising only the dense backend's capabilities. Query execution
  support for the chosen method using actual local tensors after communication;
  selection-time capability queries launch no kernels or collectives, and sparse
  preparation remains an explicit execution requirement.

The [Cosmos3 strategy PoC (#8583)](https://github.com/vllm-project/vllm-omni/pull/8583)
demonstrates separate presets, layouts, and schedules on this foundation. It is
an example of composition, not an implementation of the proposed
`AttentionSpec.sparse` spelling or a finalized common schema. Parsing,
serialization, plugin dispatch, metadata queries, and missing/boundary step
handling need shared acceptance tests when that schema is implemented.

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

## Strict Ulysses and CPU offloading

The Cosmos3 and MiniMax H3 recipes support pure strict Ulysses with ordinary static
`per_role` configuration. Set `ulysses_degree=2` and `ulysses_mode="strict"`.
No attention strategy or denoising schedule is required. Q and KV head counts
must both be divisible by the Ulysses degree. Replicated multi-control attention
keeps its full local heads and does not enter the Ulysses wrapper. MiniMax H3's
token refiner likewise stays replicated and dense.

The shared attention layer uses the existing all-to-all and prepares sparse
kernels with the resulting local head counts. Cosmos3 marks synthetic sequence
padding; it is removed after all-to-all, before block selection, and zero rows
are restored before the reverse communication. Real understanding tokens remain
a protected KV prefix. MiniMax H3 instead uses its existing single-document
`PackedPaddingMetadata`: padding is trimmed after all-to-all by the sparse
executor and zero output rows are restored before the reverse communication.
Its packed total must divide evenly across ranks, and multi-request packed
batches remain unsupported. Other models must supply equivalent padding metadata or
use sequences that need no padding; this does not enable arbitrary sparse masks.

Capability queries for the composed wrapper require `inputs_are_local=True`
and the actual post-communication tensors. They execute no collectives or
kernels. Local request preparation is still required. The wrapper reports
`EAGER_ONLY`; this does not promise fullgraph capture of communication. Regional
compilation may compile the local work and leave graph breaks around collectives.

Both models reuse their existing model and layerwise CPU offloading. These modes
work with sparse attention on one GPU or with pure strict Ulysses, in eager or
regional compilation. Distributed layerwise offloading and combined parallel
modes are outside this validation.

| Execution combination | Scope and evidence |
| --- | --- |
| Single GPU, no offload | Local FA4 correctness and compilation tests, including dynamic/fullgraph execution |
| Single GPU, model or layerwise CPU offload | Small-model Cosmos3 and MiniMax H3 tests, eager and regional compilation |
| Pure strict Ulysses, no offload | Two-rank Hopper/FA4 small-model parity, padding, and sparse-dispatch tests for both models |
| Pure strict Ulysses, model or layerwise CPU offload | The same two-rank tests with each offload mode, eager and regional compilation |
| Fullgraph capture of the Ulysses wrapper, or full-model compilation with offload | Not established by this PoC; use eager or regional compilation |
| Distributed layerwise offload, Ring, AllGather-KV, advanced Ulysses, HSDP, or hybrid parallelism | Outside supported scope |
| Cosmos3 tensor parallelism | Rank-local projections and attention tested; full TP collectives and generation remain unvalidated |

The two-rank evidence uses Hopper 64×64 geometry. Local Blackwell 256×128
adapter tests do not qualify Blackwell Ulysses/offload combinations. Ordinary
layerwise CPU offload on each Ulysses rank is distinct from the distributed
layerwise offloader and its separate weight-transfer policies. With the current ordinary
layerwise offloader, MiniMax streams its DiT blocks and keeps the token refiner
resident on the device.

Run the validation on two Hopper GPUs with FA4 `4.0.0b33`:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m pytest -v \
  tests/diffusion/models/cosmos3/test_cosmos3_sparse_ulysses.py \
  tests/diffusion/models/minimax_h3/test_minimax_h3_sparse_ulysses.py
```

The tests spawn both ranks themselves. They compare single-device and two-rank
outputs, exercise padded and unpadded requests, verify sparse dispatch per layer,
and cover eager/regional execution with no, model, and layerwise offloading.

## Current validation

After merging `main` at `a3d7a0444`, validation on one GH200 with the
vLLM-Omni `0.31.0rc1` ARM64 image and FA4 `4.0.0b33` passed 761 distinct tests
across the regression suite and focused reruns. Coverage includes sparse
selection/adapters, configuration, capability queries, sequence-parallel hooks,
Cosmos3/MiniMax integration, and single-GPU model/layerwise offloading. Eight
tests skipped: six require two GPUs and two concern unavailable MiniMax APIs.
Two-rank execution was not rerun after this merge; the following distributed
results describe the earlier foundation revision.

Before integration with `main` at `a3d7a0444`, all six Cosmos3 cases and five
MiniMax H3 Ulysses/offloading cases passed
on Hopper GPUs with FA4 `4.0.0b33`. Coverage includes standalone model and
layerwise CPU offloading, two-rank Ulysses with each offload mode, and eager and
regional compilation. These tests use small randomly initialized models, not
full checkpoints. Test lengths exceed the selector's minimum retained-block
budget, so the parity checks exercise actual sparse selection.

The earlier adapter validation used the vLLM-Omni 0.30.0 image, published FA4 `4.0.0b33`,
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
paged or quantized KV, Ring, AllGather-KV, advanced Ulysses, and hybrid
parallel modes are unsupported. Diffusers
rejects sparse configurations before loading weights; sparse execution needs a
native model integration.

Full-checkpoint generation quality remains unqualified for these static recipes.
The [Cosmos3 strategy PoC (#8583)](https://github.com/vllm-project/vllm-omni/pull/8583)
reports preliminary generation latency with the pinned wheel for one prompt and
seed; it does not establish general quality preservation or distributed throughput.

The [upstream MiniMax H3 evaluation](https://www.lmsys.org/blog/2026-08-27-minimax-h3-h200)
motivates a mixed schedule: ten initial dense steps followed by SubBlock with
64×64 blocks. Its measured speed/similarity trade-off is upstream evidence, not
validation of this port's always-sparse recipes or of other block geometries.
Follow-up qualification should compare matched dense and mixed generations with
the pinned wheel, fixed prompts/inputs and seeds, viewable outputs, similarity
measurements, and repeated generation latency. Record conditioning-token pruning
and qualify Cosmos3 and each advertised geometry separately. Neither universal
quality preservation nor the necessity of a mixed schedule is claimed.
