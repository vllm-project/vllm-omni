# MiniMax H3 attention strategies

MiniMax H3 uses the shared attention strategy configuration to select a fixed
attention layout at each denoising iteration. Operations identify their component,
layer and role: `minimax_h3.token_refiner` or `minimax_h3.dit`. Both the primary
transformer and optional reference transformer expose separate inventories.

## Example recipe

`recipes/attention/minimax-h3-subblock-strategy.json`
keeps the first twenty iterations dense, then uses FA4 SubBlock sparse attention for
all DiT layers. Token refinement remains dense FLASH_ATTN throughout. Sparse
attention uses 64×64 blocks and target sparsity 0.75, including conditioning keys
in selection. This is an illustrative PoC recipe; the schedule and layer selection
are not tuned for optimal speed or quality.

Pass the recipe as `diffusion_attention_config` when constructing `Omni`. With
CPU offloading, use `diffusion_compile_granularity="regional"` and
`diffusion_compile_dynamic=True`. Resident execution also supports full-forward
compilation. For one graph per layout, disable the exact AdaLN cache with
`cache_config={"minimax_h3_adaln_cache": False}`; its default enabled mode
introduces eager cache boundaries. Run a representative warmup before measuring steady-state latency;
new input shapes or metadata can require additional compilation.

## Execution and limits

The chosen layout flows explicitly through token refinement and DiT blocks.
Packed-attention metadata is built for the concrete executor selected by that
layout. The strategy path is traceable; legacy providers keep their existing eager
attention boundary. Denoising progress advances once per Euler iteration and is
cleared when the loop exits, including cancellation and exceptions.

This integration supports single-device and pure strict Ulysses strategies. Packed sparse execution
supports one document per forward. Multi-request sparse packing, VDN-H3 checkpoints and approximate
Cache-DiT acceleration are unsupported. Set `cache_backend="none"` and omit the
quality override or use `quality="lossless"`. Requests selecting Cache-DiT are
rejected before changing its runtime hooks. Existing non-strategy behavior is
preserved.

## Validation

```bash
python -m pytest -q tests/diffusion/models/minimax_h3/test_minimax_h3_attention_strategy.py \
  tests/diffusion/models/minimax_h3/test_minimax_h3_strategy_schedule.py \
  tests/diffusion/models/minimax_h3/test_minimax_h3_contract.py \
  tests/diffusion/models/minimax_h3/test_minimax_h3_quantization.py \
  tests/diffusion/models/minimax_h3/test_minimax_h3_fasth3.py \
  tests/diffusion/models/minimax_h3/test_fasth3_checkpoint.py
```

Small real-model tests cover CPU eager/Inductor and CUDA resident, component
and layerwise offload. They compare eager and compiled outputs, warm each layout,
inspect sparse graph calls, and replay schedule changes without recompilation.
CUDA tests also warm and replay a second packed geometry. Contract tests cover
recipe dispatch, operation inventories, packed padding and quality-policy
rejection without changing Cache-DiT hooks.

Full-model benchmark results and their scope are recorded in the PR description.
A generated example does not establish general quality parity.

## Strict Ulysses

Set `num_gpus=2` and
`parallel_config={"ulysses_degree": 2, "ulysses_mode": "strict"}`. Use eager
execution or `diffusion_compile_granularity="regional"`; full-forward compiled
Ulysses is rejected. Head counts must divide evenly across Ulysses ranks.
The same path admits no offload, model-level CPU offload, or ordinary layerwise
CPU offload. Distributed layerwise offload, Ring/hybrid parallelism, and
multi-request sparse packing remain unsupported.

H3 retains its existing rank-local embedding/RoPE preparation, DiT sequence
sharding and output gathering. Token refinement remains replicated and skips
sequence-parallel attention. Each selected DiT executor uses the shared strict
Ulysses wrapper; packed suffix padding is removed before sparse selection.
Capability queries at the local attention boundary report the composed Ulysses
path via `resolve_execution_path(..., inputs_are_local=True)`.

```bash
python -m pytest -q tests/diffusion/models/minimax_h3/test_minimax_h3_strategy_ulysses.py
```

The test spawns two ranks; do not launch it with `torchrun`. CPU/Gloo coverage
uses SDPA in place of CUDA providers and checks production communication,
dense/mixed/sparse layout switching, padded/unpadded inputs, prepared/on-demand
RoPE, changed conditioning, eager/regional parity and warmed graph reuse.
CUDA cases add actual FA4 dispatch, head-shard geometry, capability queries,
and no/model/layerwise offload. Run those with two Hopper GPUs and FA4
`4.0.0b33`.

All three CUDA cases passed on two NVIDIA H100 NVL GPUs in 191.57 seconds
(3m11s), using the x86 v0.31.0rc1 image, vLLM `0.31.0`, PyTorch
`2.13.0+cu130` and FA4 `4.0.0b33`.
This validates the small-model two-GPU strategy cases with resident, model-level
and ordinary layerwise CPU offload. Full-checkpoint distributed generation
quality and latency have not been measured.
