# Cosmos3 mixed attention PoC

This experimental PoC builds on #8335 / RFC #8396 and explores the scheduling
requirements discussed in #8382.

## Design

Named presets configure attention methods. Layouts assign presets to model-declared
roles and layers; a schedule selects a layout by logical denoising iteration.
Assignments are resolved at startup, rejecting unknown or overlapping selectors.
Each layout has a transformer callable sharing the same model weights.
Models explicitly dispatch from `forward` to the strategy runner, which invokes
`forward_with_attention_layout`. Finalization builds the runner without replacing
model methods. Cosmos3 shares request preparation and tensor preprocessing between
ordinary and strategy execution; strategies move request preparation outside compilation.

Cosmos3 prepares conditioning and positional metadata before compiled execution.
The runner selects the layout outside the graph. Attention retains the shared
`pre_attention` → `_run_local_attention` → `post_attention` path and platform-owned
provider selection. Full compilation applies `torch.compile` to one transformer callable per layout.
Regional compilation captures the existing GEN decoder blocks, keeping layout
selection, request preparation, and CPU offload transfers outside those graphs.

Configuration parsing normalizes presets and schedules without loading backends.
Executor construction checks the selected method's local-execution declaration;
backend-specific options are passed through to the provider's implementation.
Methods must declare local execution without independent request-context scheduling.
Compilation is best effort: the strategy does not require backend compilation
capabilities or fullgraph capture. PyTorch and each attention backend determine
which regions compile and where graph breaks occur. Recompilation limits and
compiler failures follow PyTorch's configured behavior.

Dense FA3 uses its native execution and compilation path, including the FA3
package's existing custom operation. The strategy adds no FA3 kernel wrapper
or runtime shape restrictions. FA4/SubBlock retains its actual-request
preparation boundary.

The PoC supports single-device execution and pure strict Ulysses for standard
Cosmos3. Ulysses requires Q and KV head counts divisible by `ulysses_degree`,
and eager execution or regional compilation. Ring, AllGather-KV, TP, PP, CFG/DP
parallelism, HSDP, and distributed layerwise offloading remain outside its scope.

Warmup exercises every reachable layout. Sparse attention retains its existing
per-executor preparation bookkeeping and prepares new geometries on first use.

## Usage

Pass [cosmos3-subblock-strategy.json](../../../recipes/attention/cosmos3-subblock-strategy.json)
as `diffusion_attention_config`. It keeps ten iterations dense, then uses sparse
attention in 28 of 36 generation layers; understanding and multi-control stay dense.
The presets are `fa3_dense` (dense FlashAttention-3) and `fa4_subblock`
(SubBlock selection through FA4 `4.0.0b33`, with Hopper 64×64 blocks).
Use a Hopper environment with FA3 installed through `fa3-fwd` or a
`flash_attn_interface` source build. `FLASH_ATTN` is the provider selector;
the preset name does not pin a kernel version. The CUDA validation launcher
checks the resolved dense implementation is FA3. Sparse execution uses the
separate FA4 adapter; its `implementation: auto` is required by that API.

For a uniform switch in every generation layer after ten dense iterations, use
[cosmos3-dense-to-sparse-strategy.json](../../../recipes/attention/cosmos3-dense-to-sparse-strategy.json).
Understanding and multi-control attention remain FA3 dense in both recipes.

For a dense → mixed → dense schedule, keep the first recipe's presets/layouts
and replace its `schedule` with:

```json
{
  "coordinate": "step_fraction",
  "phases": [
    {"until": 0.3, "layout": "dense"},
    {"until": 0.8, "layout": "mixed"},
    {"until": 1.0, "layout": "dense"}
  ]
}
```

For 30 iterations, this uses FA4/SubBlock in the selected generation layers
during zero-based iterations 9–23, with FA3 elsewhere.

Use `diffusion_compile_granularity="full"` and `diffusion_compile_dynamic=True`, or
`enforce_eager=True`. For model-level CPU offloading, set `enable_cpu_offload=True`
and `diffusion_compile_granularity="regional"`; eager execution is also supported.
For layerwise CPU offloading, use `enable_layerwise_offload=True` instead, also
with regional compilation or eager execution. The existing Cosmos3 offloader
streams reasoner and generator blocks; transfer hooks remain outside compiled regions.
Full compilation remains incompatible with offloading. Cosmos3's existing mixed-precision schedule remains independent:
it selects linear precision while this strategy selects attention. Both use the
logical denoising iteration. Precision changes may specialize a layout's compiled
graph; each layout has its own recompilation budget.
Active strategies cannot also specify root-level `default`, `per_role`, or
`diffusion_attention_backend`.

### Checkpoint defaults and deployment overrides

A transformer checkpoint may declare `runtime.attention_strategy` in its component
`config.json`. This is a PoC schema, independent of `quantization_config` and
ModelOpt's algorithm-specific `sparse_attention_config`:

```json
{
  "runtime": {
    "attention_strategy": {
      "schema_version": 1,
      "config": {
        "presets": {"fa3_dense": {"backend": "FLASH_ATTN"}},
        "layout": {"default": "fa3_dense"}
      }
    }
  }
}
```

The `config` object accepts the same `presets`, `layout`, `layouts` and `schedule`
as runtime strategies. An explicit runtime strategy, `default`, or `per_role`
configuration replaces the entire checkpoint policy. An explicit backend from
the CLI or environment also takes precedence. Empty runtime configuration permits
checkpoint discovery; `diffusion_attention_config={"checkpoint_policy": "ignore"}`
disables checkpoint attention-policy loading without disabling quantization or
mixed precision. No metadata preserves existing defaults.

Resolution runs before attention construction, using the loaded transformer
configuration. Invalid effective policies fail loading, including when discovered
after the deployment config is constructed. Runtime overrides are still subject
to method, model and execution validation. Startup logs report the policy source.
No ModelOpt dependency is
needed at inference time. This PoC reads the declared schema; it does not yet
provide a ModelOpt exporter or translate existing `sparse_attention_config` metadata.

The simple dense-to-sparse policy in #8382 uses two layouts and a `step_index`
schedule: dense until `start_step`, then sparse until `null`. For `start_step=0`,
omit the dense phase; a boundary at or beyond the request's iteration count keeps
the request dense. Schedules require valid zero-based logical progress even with
one reachable layout. Repeated CFG/solver evaluations at the same index reuse the
same layout. A static `layout` needs no denoising progress. The policy is resolved
before transformer execution, so local attention does not read the step index.

Distributed modes other than strict Ulysses, cache acceleration, paged/quantized
KV, step serving and MiniMax integration are outside this PoC.

## Validation

```bash
python -m pytest -q tests/diffusion/attention/test_attention_strategy*.py \
  tests/diffusion/attention/test_attention_checkpoint_policy.py \
  tests/diffusion/attention/test_cosmos3_action_strategy.py \
  tests/diffusion/models/cosmos3/test_cosmos3_attention_strategy.py
```

Tests cover layout switching, conditioning refresh, real FA4 execution
and graph reuse. The pre/local/post regression also records actual compiled-graph
invocations across layout switches and changing lengths; this is test evidence,
not a production per-kernel dispatch trace.
Generation quality and performance comparisons are separate validation.

Mixed-precision implementation and end-to-end quantized generation are outside
this PoC. The existing precision scheduler is unchanged; this PoC does not validate
combined quantized-model execution.

### Ulysses PoC validation

Set `num_gpus=2`, `parallel_config={"ulysses_degree": 2, "ulysses_mode": "strict"}`,
and `diffusion_compile_granularity="regional"` (or `enforce_eager=True`).
The normal Cosmos3 split/gather hooks and Ulysses all-to-all are reused.
Strategies build on the shared static subblock-sparse Ulysses support; sparse
executors use the same post-all-to-all head counts and padding metadata. Synthetic suffix padding is
removed before attention and restored before reverse all-to-all; replicated
UND keys remain a protected prefix. Multi-control keeps its existing replicated
execution. Full-graph Ulysses is deliberately rejected.

Sparse capability reporting composes the strict Ulysses wrapper with the local
kernel contract. Call `Attention.resolve_execution_path(..., inputs_are_local=True)`
with the actual tensors and metadata at the local attention boundary, after
all-to-all, joint-KV assembly and suffix trimming. Queries execute no collectives
or kernels; an unprepared local geometry remains unsupported until it executes
successfully. Pre-communication inputs are rejected explicitly. The composed
path reports Ulysses with `EAGER_ONLY` compilation: regional block compilation
is supported, but the entire communication wrapper cannot promise fullgraph
capture. The sparse kernel itself retains its local-only contract.

From the repository root in an environment with the project dependencies,
FA4 `4.0.0b33`, and two Hopper GPUs, run:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m pytest -v -s --tb=short \
  tests/diffusion/models/cosmos3/test_cosmos3_strategy_ulysses.py -m cuda
```

The test spawns two ranks itself: do not wrap it in `torchrun`. It needs no model
checkpoint. It compares a small, same-weight Cosmos3 model at Ulysses degrees
one and two across dense/mixed/sparse layouts, padded and unpadded sequences,
changed conditioning, eager and regional execution, and no/model/layerwise
offloading. It checks that warmed regions are reused. This is a numerical PoC
gate, not a full-checkpoint generation or performance benchmark.
Each rank also checks actual sparse dispatch for every scheduled generation
layer and step, including dense steps with zero sparse calls. Explicit warmup
is checked separately because it intentionally exercises every layout.

A CPU/Gloo version checks the communication and model wiring with SDPA in place
of CUDA providers:

```bash
python -m pytest -v -s --tb=short \
  tests/diffusion/models/cosmos3/test_cosmos3_strategy_ulysses.py -m cpu
```

Validation on two Hopper GPUs passed all three strategy CUDA cases: Ulysses
without CPU offload, with model-level offload, and with layerwise offload.
Shared sparse head-shard and padding coverage lives in
`tests/diffusion/models/cosmos3/test_cosmos3_sparse_ulysses.py`.
The distributed cases cover eager and regional
execution, output parity, and graph reuse after warmup. The CuTe `AuxData`
argument warning and PyTorch collective deprecation warnings did not prevent
these checks from passing; this does not establish full-checkpoint generation
quality or throughput.
