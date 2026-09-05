# Production Feature Patterns

## Contents

1. [Compatibility matrix](#compatibility-matrix)
2. [Online FP8](#online-fp8)
3. [Strict fused-weight loading](#strict-fused-weight-loading)
4. [Distributed layerwise offload](#distributed-layerwise-offload)
5. [Per-request Cache-DiT](#per-request-cache-dit)
6. [Combination gates](#combination-gates)

## Compatibility matrix

Create rows at the granularity below. Add evidence links and a reason for every
non-validated state.

| Task/shape | Execution mode/capacity | Backend/layout | Quant | Cache policy | Offload | Topology | Hardware/dtype | Output contract/transport | State | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|
| `<task>/<shape>/<schedule>` | serial request / 1 | SDPA, padded | BF16 | none | resident | 1 device | `<card>` BF16 | raw float / local | `not tested` | — |
| `<task>/<shape>/<schedule>` | step / N | `<fast>`, packed | BF16 | none | resident | `<TP/SP>` | `<card>` BF16 | encoded / subprocess | `not tested` | — |
| `<task>/<shape>/<schedule>` | step / N | `<fast>`, packed | FP8 | high | DLO no-AG | `<TP/SP>` | `<card>` | device uint8 / SHM | `not tested` | — |

Initialize every row as `not tested`; the table is a work queue, not evidence.
Promote the dense BF16 row to the oracle only after its scoped parity passes.
Add one axis at a time. If the runtime cannot
enforce a known-incompatible combination, add construction/admission
validation before writing a recipe.

## Online FP8

### Component routing

The public builder accepts one method or a longest-prefix component map:

```python
from vllm_omni.quantization import build_quant_config

config = build_quant_config({
    "transformer": {
        "method": "fp8",
        "ignored_layers": ["transformer.final_layer"],
    },
    "text_encoder": None,
    "vae": None,
    "default": None,
})

assert config.resolve("transformer.blocks.0.attn.qkv_proj") is not None
assert config.resolve("text_encoder.layers.0.mlp") is None
assert config.resolve("vae.decoder.conv_out") is None
```

Use actual runtime prefixes. Multi-DiT or nested pipelines may not use the
generic names `transformer` and `vae`. A prefix that falls through to a default
is not component validation. Add tests for every included and ignored prefix.

For CLI serving, first validate resident DiT-only FP8:

```bash
vllm serve '<model>' --omni --quantization fp8 --max-num-seqs 1
```

For scoped routing:

```bash
vllm serve '<model>' --omni \
  --diffusion-quantization-config \
  '{"<dit-runtime-prefix>":{"method":"fp8","ignored_layers":["<exact-linear-prefix>"]},"<vae-runtime-prefix>":null,"default":null}' \
  --max-num-seqs 1
```

Do not publish these placeholders. A recipe must contain the target model's
verified prefixes.

### Quantizable module wiring

Pass `quant_config` and a stable prefix through every vLLM linear. Keep
precision-sensitive modulation/norm layers safe unless proven otherwise.

```python
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)

self.qkv_proj = QKVParallelLinear(
    hidden_size=hidden_size,
    head_size=head_dim,
    total_num_heads=num_heads,
    total_num_kv_heads=num_kv_heads,
    bias=False,
    quant_config=quant_config,
    prefix=f"{prefix}.attn.qkv_proj",
)
self.gate_up_proj = MergedColumnParallelLinear(
    hidden_size,
    [intermediate_size, intermediate_size],
    bias=False,
    quant_config=quant_config,
    prefix=f"{prefix}.mlp.gate_up_proj",
)
self.act_fn = SiluAndMul()
self.down_proj = RowParallelLinear(
    intermediate_size,
    hidden_size,
    bias=False,
    input_is_parallel=True,
    quant_config=quant_config,
    prefix=f"{prefix}.mlp.down_proj",
)
```

Set required checkpoint parameters to fail rather than silently initialize if
the current loader supports `missing_param_init = "error"`.

### FP8 evidence

For each task/hardware row, capture:

- startup proves the FP8 quant method is attached to intended linear modules;
- no unexpected BF16 fallback and no meta/uninitialized parameters;
- same-seed component, trajectory, and final-artifact comparison with BF16;
- HBM at load, resident, encode, denoise, decode, and peak;
- cold load time; fixed-work warm-latency raw runs; and serving p50/p95/p99 plus
  throughput only from a declared arrival load with enough samples;
- named ignored layers and why their BF16 retention is required.

DiT FP8 evidence does not cover the text encoder or VAE. CUDA evidence does not
cover ROCm/NPU/XPU. A pre-quantized checkpoint is a different loading path from
online FP8 and needs its own state. If it is also pruned, rotated, distilled, or
adapter-modified, report that model delta explicitly; BF16-versus-artifact
quality and speed are deployment comparisons, not pure quantization ablations.

## Strict fused-weight loading

Never use `param.data.copy_()` for quantizable vLLM parameters. Transform the
checkpoint layout before invoking the parameter loader so the online quant
loader remains outermost:

```python
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

param = params[target_name]
weight_loader = getattr(param, "weight_loader", default_weight_loader)

if target_name.endswith(".qkv_proj.weight"):
    packed_qkv = reorder_checkpoint_qkv(checkpoint_weight)
    weight_loader(param, packed_qkv)
elif target_name.endswith(".gate_up_proj.weight"):
    gate, up = checkpoint_weight.chunk(2, dim=0)
    weight_loader(param, gate, 0)
    weight_loader(param, up, 1)
else:
    weight_loader(param, checkpoint_weight)
```

Track source shards per fused destination:

```python
expected = {
    "blocks.0.attn.qkv_proj.weight": {"q", "k", "v"},
    "blocks.0.mlp.gate_up_proj.weight": {0, 1},
}
seen = {name: set() for name in expected}

# After mapping/loading each source:
seen[target_name].add(shard_id)

incomplete = {
    name: sorted(shards - seen[name], key=str)
    for name, shards in expected.items()
    if shards - seen[name]
}
if incomplete:
    raise RuntimeError(f"incomplete fused checkpoint parameters: {incomplete}")
```

Also compare retained model parameters with the returned loaded set. Explicit
checkpoint ignores need a reason and test; an unknown or missing weight is not
an ignore policy.

Load-time adapter fusion creates a dedicated model, not a request-switchable
LoRA. Pin the base and adapter revisions, validate every low-rank and full-rank
delta target, freeze the adapter-declared schedule/task, and fuse before
sharding only when the loader contract proves identical rank-local weights.
Reject offload or host-weight paths that bypass that fusion rather than serving
the unfused base model silently.

## Distributed layerwise offload

### Declare component topology

Use `SupportsComponentDiscovery` and `OffloadPlan`. Validate all dotted paths
at construction/test time because discovery may otherwise warn and skip.

Treat a future/requested `--layerwise-offload-components` selector as a public
API only if it exists in the target revision. Today, express component topology
with `OffloadPlan` (`on_demand_component_paths`, `block_attrs`,
`encoder_block_attrs`, `resident_dit_paths`, and `offload_submodules`). Do not
invent a CLI flag or claim component-selective DLO from discovery declarations
alone; validate the selector separately when it lands.

```python
from operator import attrgetter

import torch

def assert_offload_path(root, path):
    value = attrgetter(path)(root)
    if not isinstance(value, torch.nn.Module):
        raise TypeError(f"offload path {path!r} is not an nn.Module")
    return value

for path in (
    *pipeline._dit_modules,
    *pipeline._encoder_modules,
    *pipeline._vae_modules,
    *pipeline._offload_plan.on_demand_component_paths,
):
    assert_offload_path(pipeline, path)
```

For `block_attrs`, resolve the DiT first and then each declared block attribute.
Require an indexable block container with at least one `nn.Module`. Validate
`offload_submodules`, `resident_dit_paths`, and `encoder_block_attrs` similarly.

### Keep host storage loader-owned

Current DLO accepts a loader-produced `HostWeightPlan`. Pipeline authors should
not instantiate this plan or add ad hoc mmap capability flags. The loader must
preflight complete runtime/checkpoint bindings, names, shapes, dtypes, persistent
buffers, and represented transforms before it skips ordinary materialization.
The exact accepted plan is consumed once by DLO; a rejected plan records an
observable fallback reason and continues through the ordinary loader.

Direct checkpoint mmap currently requires TP1 without HSDP or online
quantization. TP>1 falls back to the ordinary TP-aware loader and may feed DLO
AllGather or no-AllGather; the current design records a bounded DP2xTP2
AllGather smoke, but that path does not receive direct-mmap host-page sharing.
Per-tensor online FP8 must use the ordinary loader. Current revisions can admit
its finalized weight/scale layout to DLO AllGather, including transposed runtime
weights, while unsupported online quantizers still fail closed. Direct mmap
remains unavailable, so capture the temporary full-model host materialization
before rank sharding.

Host Weight Runtime is a distinct opt-in path. Current final-layout consumers
use no-AllGather DLO and exact model-declared BF16 representations. In
`preferred` mode, consume an exact local hit or canonically load and publish for
a future startup; in `required` mode, consume only and fail on a miss or unusable
artifact. HWR must not interact with DLO AllGather, online quantization, HSDP,
LoRA/adapted weights, or an unrecognized load format unless the target revision
adds an exact producer/restore contract. It is immutable node-local CPU backing,
not zero-copy GPU execution; the DLO transport still owns staging or direct H2D.
Qualify identity inputs, cold population, warm reuse, atomic concurrent
publication, source digests, corruption/quarantine, lease cleanup, capacity,
node-local PSS, H2D, and E2E latency. A host-memory saving that materially
regresses transfer or request latency is a separate memory-first deployment.

### DLO deployment modes

Treat these paths separately:

| Path | Weight/runtime behavior | Required boundaries |
|---|---|---|
| DLO AllGather | Rank-sharded pinned host tensors; reconstruct layer through collective | TP>1 uses ordinary TP-aware loader output; HSDP and unsupported online quantizers are rejected; finalized per-tensor online FP8 is eligible only when recognized by the target revision; concurrent ranks must participate consistently |
| DLO no-AllGather | Loader-approved checkpoint mmap or ordinary runtime tensors; each rank streams a complete block | Direct mmap is TP1/non-HSDP/non-online-quant only; TP/HSDP/online quant use ordinary runtime tensors and require scoped E2E |
| SP + DLO | SP group can supply the DLO collective group | Packed boundaries and collective order require E2E |

Single-stage examples:

```bash
# SP + DLO AllGather candidate; validate exact model/card before publishing.
vllm serve '<model>' --omni \
  --enable-distributed-layerwise-offload \
  --usp 4 \
  --max-num-seqs 1

# TP + DLO rank-local candidate; not a support statement.
vllm serve '<model>' --omni \
  --tensor-parallel-size 2 \
  --enable-distributed-layerwise-offload \
  --dlo-no-use-allgather \
  --max-num-seqs 1
```

Under `--omni`, configure DP in a deploy YAML rather than passing vLLM DP CLI
flags:

```yaml
pipeline: <registered-pipeline>
async_chunk: false
data_parallel_size: 4
stages:
  - stage_id: 0
    devices: "0,1,2,3"
    max_num_seqs: 1
    enable_distributed_layerwise_offload: true
    dlo_use_allgather: true
    parallel_config:
      tensor_parallel_size: 1
      sequence_parallel_size: 1
```

```bash
vllm serve '<model>' --omni --deploy-config /path/to/dlo_dp4.yaml
```

Keep the YAML next to the validated recipe. Match its schema to the target
revision; do not assume a model accepts another pipeline's deploy config.

### DLO validation sequence

1. Resident BF16 reference.
2. Ordinary layerwise offload parity.
3. DLO enable, warmup, same-seed generation, disable/restart.
4. AllGather and no-AllGather separately.
5. Concurrent requests with different prompts and identical collective-affecting
   parameters; then reject or schedule incompatible steps/shapes safely.
6. Queued/in-flight abort, error, idle rank, worker exit, and next request.
7. TP, SP, cache, compile, online FP8 one axis at a time.
8. Loader-plan selection/fallback reason, cold/warm storage lifecycle, and
   corruption/stale-entry recovery when mmap/cache backing is advertised.
9. Replica-local request preparation plus DLO-group-only weight collectives in
   DP; exercise concurrent replicas to catch accidental WORLD rendezvous.
10. Sweep resident block counts; publish raw latency/HBM pairs, dominated
    points, and the non-dominated Pareto frontier.
11. Per-rank HBM, host PSS for the full process tree, H2D and collective trace.

On a small-HBM card, measure transient encode/VAE peaks as well as resident DiT
weights. If a non-DiT component still exceeds HBM, combine only independently
validated on-demand component staging, VAE tiling/patching, or another topology;
do not hide the remaining peak behind the phrase “DLO enabled.”

## Per-request Cache-DiT

### Policy and lifecycle skeleton

First verify that the target revision contains the shared protocol/runtime.
If not, follow the nearest current model's lifecycle without inventing a new
public API.

The following is a **serial/exclusive request-mode sketch only**. Pipeline-wide
hooks are mutable; concurrent requests need an admission/batching key and
runner-owned transition serialization before this shape is safe.

```python
import torch.nn as nn

from vllm_omni.diffusion.cache.cachedit import (
    CacheDiTBackend,
    CacheDiTRequestSpec,
    RequestScopedCacheDiTRuntime,
)

class MyPipeline(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # ... components ...
        self._cache_dit_runtime = RequestScopedCacheDiTRuntime(self)

    def adopt_cache_dit_backend(self, backend: CacheDiTBackend) -> None:
        self._cache_dit_runtime.adopt(
            backend,
            installation_key="my_model.generic",
        )

    def is_cache_dit_enabled(self) -> bool:
        return self._cache_dit_runtime.is_enabled

    def _cache_spec(self, quality, steps):
        if quality == "lossless":
            return None
        if quality == "high":
            return CacheDiTRequestSpec(
                installation_key="my_model.high.v1",
                cache_config=self._validated_high_config,
                num_inference_steps=steps,
            )
        raise ValueError(f"unsupported quality tier: {quality!r}")

    def forward(self, req):
        quality, steps = validate_and_resolve_request(req)
        self._cache_dit_runtime.prepare(self._cache_spec(quality, steps))
        return self._run_validated_request(req)
```

The snippet shows the model-owned transition for exclusive execution. Put
disable/restore handling in the runtime/runner lifecycle that owns success,
exception, disconnect, and asynchronous abort; a `try/finally` inside
`forward()` alone cannot observe every server cancellation boundary.

The `installation_key` must include every policy dimension that changes hook
installation; changes in step count can use runtime refresh. `None` disables
installed hooks. Do not mutate the global startup cache config for a request.

Centralize cleanup in the owning request/runner boundary. An exception, abort,
disconnect, or admission failure must not leave a changed hook policy visible
to the next request. The exact owner depends on the target revision; add a
failure-injection test rather than relying on `forward()` alone to observe
asynchronous abort.

Example model-specific request form after calibration:

```bash
curl --fail-with-body -sS -X POST http://127.0.0.1:8091/v1/videos/sync \
  -F 'model=<model>' \
  -F 'prompt=<prompt>' \
  -F 'quality=lossless' \
  -F 'num_inference_steps=<validated-steps>' \
  -o lossless.mp4

curl --fail-with-body -sS -X POST http://127.0.0.1:8091/v1/videos/sync \
  -F 'model=<model>' \
  -F 'prompt=<prompt>' \
  -F 'quality=high' \
  -F 'num_inference_steps=<validated-steps>' \
  -o high.mp4
```

Do not expose these tier names merely because another model uses them.

### Cache-DiT evidence

For every tier/task/shape/schedule:

- show real cache hits/skipped blocks, not only “backend enabled” logs;
- measure fixed-work speed and quality against lossless/native;
- publish a speed/quality frontier and chosen tier mapping;
- alternate lossless/high/lossless and vary steps to prove restore/refresh;
- interleave tasks and shapes; run concurrent requests only when scheduling
  guarantees exclusive compatible hook state;
- inject validation error, denoise error, disconnect, and abort, then prove the
  next request starts with its requested policy;
- bound any cached tensors and clear request-local state.

## Combination gates

Start every combination as `not tested`.

| Combination | Default production posture |
|---|---|
| Direct checkpoint mmap + TP>1/HSDP/online quant | Preflight falls back to the ordinary loader; do not claim direct-mmap savings |
| Per-tensor online FP8 + DLO AllGather | Generic compatibility exists for finalized weights/scales; each model/card/topology remains a candidate until E2E, and other online quantizers fail closed |
| Online FP8 + DLO no-AllGather | Candidate only after model/card/topology E2E |
| TP>1 + DLO AllGather | Ordinary TP-aware loader only; bounded DP2xTP2 smoke exists, but each new model/card/topology still needs E2E |
| TP>1 + DLO no-AllGather | Ordinary TP-aware loader only; candidate with limited generic coverage |
| HSDP + DLO AllGather | Reject; it would double-shard HSDP-managed parameters |
| HSDP + DLO no-AllGather | Configuration-compatible candidate with limited E2E coverage |
| Cache-DiT + online FP8 | Separate trajectory, hit-rate, HBM, latency test |
| Cache-DiT + DLO | Verify hooks, streamed blocks, memory peaks, abort cleanup |
| Cache-DiT + step execution | Do not advertise until lifecycle/batching support is explicit |
| Sparse attention + Ring | Verify backend execution; reject silently ignored sparse settings |
| Load-time fused adapter + offload/HWR | Reject unless the selected loader path applies and validates the same fusion before sharding/storage |
| Packed attention + any topology | Unequal-sample boundary parity is mandatory |
| Device-side uint8 preparation + offline caller | Version the dtype/range/layout change; HTTP MP4 parity does not preserve the raw tensor contract |
| Device-side preparation + remote codec | Validate route selection, strided/contiguous layouts, payload bytes, ordering, fallback, and encoded-media parity |

A startup-only test never upgrades a combination. Require request output,
parity/quality, actual fast-path evidence, resource cleanup, and a benchmark on
the named hardware.
