---
name: production-add-diffusion-model
description: Productionize a vLLM-Omni diffusion model after its Day-0 vertical slice works. Use when work requires official API parity and input limits, feature-combination evidence, online FP8, distributed layerwise offload, per-request Cache-DiT, CUDA/ROCm/NPU/XPU recipes, faster or sparse attention, fused operators, USP or disaggregation analysis, continuous batching and abort handling, long-running RPS stability, or accuracy/performance/reliability CI. For initial architecture porting, registry wiring, and basic weight loading, use add-diffusion-model first.
---

# Productionizing a Diffusion Model

## Scope

Use this skill to move a working model integration from Day-0 support to a
measured, fail-closed production deployment. It is deliberately separate from
[`add-diffusion-model`](../add-diffusion-model/SKILL.md); do not modify or copy
that skill when productionizing a model. If registry, pipeline, or basic loader
work is incomplete, read that skill first and finish the Day-0 vertical slice.

MiniMax-H3 [PR #5691](https://github.com/vllm-project/vllm-omni/pull/5691)
is the merged Day-0 case study. Its open optimization
[issue #5700](https://github.com/vllm-project/vllm-omni/issues/5700) is an
actively maintained roadmap and evidence map, not a list of already supported
features. It can lag merged implementation: read its current revision, cited
PRs, target source, and maintained recipes, then leave every unverified row
`not tested` until the target revision has scoped evidence.

Read these references before the corresponding gate:

- [API and hardware recipes](references/api-and-recipes.md)
- [Production feature patterns](references/feature-patterns.md)
- [Performance patterns and code examples](references/performance-patterns.md)
- [Serving, soak, and CI validation](references/production-validation.md)

## Non-negotiable evidence contract

Create a matrix whose row key is at least:

```text
task x API mode x execution mode/capacity x shape/schedule x attention backend x
packed layout x cache policy x quantization x offload mode x topology x
hardware x dtype x output representation/transport
```

Use exactly these states:

| State | Meaning |
|---|---|
| `validated` | The exact row passed correctness and measured deployment tests, with reproducible artifacts. |
| `limited` | A bounded subset passed; the limitation and rejection/fallback behavior are explicit. |
| `unsupported` | A stable architectural/platform restriction is proven and fails early with an actionable error. |
| `not tested` | No adequate evidence exists. This is the default. |

Never infer compatibility from import success, server startup, another task,
another card, or another topology. Record the model/checkpoint revision,
vLLM-Omni commit, commands, request assets and hashes, seed/schedule/shape,
raw output artifacts, environment, and result for every `validated` row.

## Workflow

Follow the gates in order. Keep a dense BF16 single-device path as the
correctness oracle until every advertised fast path has parity evidence.

### Gate 0: Freeze the official contract

Pin the official implementation and checkpoint revision. Build a matrix for:

- constructor/call arguments, defaults, scheduler and sigma schedule;
- task-to-source modality/count/order rules;
- prompt, MIME, bytes, pixels, frames, duration, FPS, steps, seed, guidance,
  output count, and output format limits;
- offline API and every public sync/async serving endpoint;
- checkpoint partitions and task selection.

Normalize offline, sync, and async requests through one validation contract.
Reject an invalid task/source combination before downloads, upload persistence,
temporary files, async job creation, or engine submission. Use 400 for invalid
semantics and 413 for payload limits; clean partial resources on every failure.
See [API and hardware recipes](references/api-and-recipes.md).

### Gate 1: Re-prove correctness under production shapes

1. Make loading strict: track expected, loaded, unexpected, and intentionally
   ignored tensors. A fused destination is complete only after every source
   shard arrives. Missing retained parameters abort startup.
2. Fix reference revision, asset hashes, seed, schedule, and shape. Compare
   component outputs, intermediate denoise latents, and final artifacts.
3. Test unequal packed samples. Verify `cu_seqlens`, padding exclusion,
   modality/CFG ownership, split/gather boundaries, and one output per request.
4. Test partial process groups with non-member ranks and group size smaller
   than world size. Do not construct a private group per component or request.
5. Run negative API cases before positive E2E tests to prove side-effect-free
   rejection and cleanup.

### Gate 2: Wire shared fast primitives

Prefer explicit, testable vLLM-Omni/vLLM operators with native fallbacks.
`torch.compile` remains a separately validated regional optimization; it is
not a substitute for correct operator selection or lifecycle-safe boundaries.

#### RMSNorm + RoPE baseline

Use the shared cross-platform operators, preserving the official Q/K norm and
RoPE order:

```python
from vllm_omni.diffusion.layers.norm import RMSNorm
from vllm_omni.diffusion.layers.rope import (
    RotaryEmbedding,
    apply_rope_to_qk,
)

self.norm_q = RMSNorm(head_dim, eps=eps)
self.norm_k = RMSNorm(head_dim, eps=eps)
self.rope = RotaryEmbedding(is_neox_style=False)

query = self.norm_q(query)  # [B, S, Hq, D] or packed [T, Hq, D]
key = self.norm_k(key)      # [B, S, Hkv, D] or packed [T, Hkv, D]
query, key = apply_rope_to_qk(
    self.rope,
    query,
    key,
    (cos, sin),
)
```

Verify `is_neox_style`, `half_head_dim`, partial rotary dimensions, cos/sin
layout, and packed row ownership against the official model. `False` means
interleaved/GPT-J style; do not assume the default is NeoX. Shared RMSNorm and
RoPE are two dispatched operators, not automatically one fused kernel.

For packed non-interleaved RoPE, use the shared fused boundary when its contract
matches the model:

```python
from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_qk_norm_rope

query, key = fused_qk_norm_rope(
    query,                    # [T, Hq, D]
    key,                      # [T, Hkv, D]
    self.norm_q.weight,       # [D]
    self.norm_k.weight,       # [D]
    rope_table,               # [T, rotary_dim] = [cos | sin]
    self.norm_q.variance_epsilon,
)
```

The current CUDA fast path specializes BF16 `head_dim=128`, `rotary_dim=96`;
the public function keeps an eager fallback for unsupported inputs. Verify the
packed row/frequency contract and trace the actual fast path. Do not reshape an
incompatible official RoPE layout merely to enter this kernel.

MiniMax-H3-style partial NeoX RoPE rotates a prefix and passes the suffix
through:

```python
self.q_norm = RMSNorm(head_dim, eps=eps, dtype=torch.bfloat16)
self.k_norm = RMSNorm(head_dim, eps=eps, dtype=torch.bfloat16)
self.rope = RotaryEmbedding(is_neox_style=True, half_head_dim=False)

def apply_partial_rope(self, x, freqs, rot_dim):
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    cos, sin = torch.cos(freqs).to(x.dtype), torch.sin(freqs).to(x.dtype)
    return torch.cat((self.rope(x_rot, cos, sin), x_pass), dim=-1)
```

The partial-RoPE snippet assumes `import torch` and lives on the owning
attention module; adapt names and validated frequency shapes rather than
copying it at module scope.

Call the modules normally so `CustomOp` selects CUDA/HIP/NPU/XPU/native
implementations. Do not call `forward_cuda()` directly. Compare the shared
path with native BF16 at operator, block, denoise-trajectory, and artifact
levels before claiming either accuracy or speed.

#### Faster BF16 attention and fused projections

Use local TP head counts and role-aware shared attention:

```python
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm_omni.diffusion.attention.layer import Attention

self.qkv_proj = QKVParallelLinear(
    hidden_size=hidden_size,
    head_size=head_dim,
    total_num_heads=num_heads,
    total_num_kv_heads=num_kv_heads,
    bias=False,
    quant_config=quant_config,
    prefix=f"{prefix}.qkv_proj",
)
self.attn = Attention(
    num_heads=self.qkv_proj.num_heads,
    num_kv_heads=self.qkv_proj.num_kv_heads,
    head_size=head_dim,
    softmax_scale=head_dim**-0.5,
    causal=False,
    qkv_layout="BSND",
    role="self",
    prefix=prefix,
)
```

Prove the selected backend from runtime logs/trace and compare it with dense
SDPA. Packed metadata must express real sample boundaries; do not build masks
that the selected backend ignores. A backend that is faster for one shape or
platform is not a global default. See [performance patterns](references/performance-patterns.md)
for QKV, SwiGLU, AdaLN, sparse attention, redundancy, and fusion examples.

### Gate 3: Add online FP8 without breaking the loader

Route the active config to each component, then pass a stable runtime prefix
and config into every quantizable vLLM linear:

```python
def resolve_component_quant_config(quant_config, component):
    return quant_config.resolve(component) if hasattr(quant_config, "resolve") else quant_config

transformer_quant_config = resolve_component_quant_config(
    od_config.quantization_config,
    "transformer",
)
self.transformer = MyDiT(
    od_config,
    quant_config=transformer_quant_config,
    prefix="transformer",
)

self.qkv_proj = QKVParallelLinear(
    hidden_size=hidden_size,
    head_size=head_dim,
    total_num_heads=num_heads,
    total_num_kv_heads=num_kv_heads,
    quant_config=quant_config,
    prefix=f"{prefix}.qkv_proj",
)
```

Use `build_quant_config()` for direct construction/testing:

```python
from vllm_omni.quantization import build_quant_config

quant_config = build_quant_config({
    "transformer": {"method": "fp8"},
    "text_encoder": None,
    "vae": None,
    "default": None,
})
```

Preserve each parameter's vLLM `weight_loader`; perform checkpoint layout
conversion before invoking it. Account for every QKV/gate-up source shard.
Keep precision-sensitive norm/modulation/embedders in BF16 unless separately
proven, using the current `safe_quant_config` pattern where applicable.

Validate resident DiT FP8 first: prove which modules are quantized, HBM saved,
BF16-vs-FP8 trajectory/artifact quality, latency, and throughput. Text encoder,
VAE, other hardware, TP/cache/DLO combinations are separate rows.
Pre-quantized, pruned, rotated, or adapter-modified checkpoints are separate
model/loading lanes; compare them with the released BF16 model without calling
the result a pure quantization ablation. Read
[`../quantization/SKILL.md`](../quantization/SKILL.md) if present and prefer the
target revision's implementation/docs when sibling guidance differs.

### Gate 4: Make every component offloadable

Declare topology instead of relying on heuristic discovery:

```python
from typing import ClassVar

import torch.nn as nn

from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.offloader import OffloadPlan

class MyPipeline(nn.Module, SupportsComponentDiscovery):
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _resident_modules: ClassVar[list[str]] = []
    _offload_plan: ClassVar[OffloadPlan] = OffloadPlan(
        on_demand_component_paths=frozenset({"text_encoder", "vae"}),
        block_attrs={"transformer": ("blocks",)},
        offload_submodules={"token_refiner": "blocks"},
        resident_dit_paths=frozenset({"transformer"}),
        encoder_block_attrs={"text_encoder": ("encoder.layers",)},
    )
```

Add a test that resolves every dotted component/block path and fails if it is
missing or is not the expected module/container; discovery may otherwise warn
and skip. Validate ordinary layerwise offload and distributed layerwise
offload (DLO) independently.

Keep component lifecycle and host-weight storage separate. `OffloadPlan`
declares what can be streamed; the diffusion loader owns any `HostWeightPlan`
and hands the exact prevalidated plan to DLO. Do not make the offloader rescan
checkpoint files or duplicate loader name/shape/dtype decisions. Direct mmap is
a fail-closed optimization: current preflight requires TP1, no HSDP, no online
quantization, complete bindings, and represented transforms. Otherwise the
ordinary loader remains authoritative.

If the target revision supports Host Weight Runtime (HWR), treat it as a
separate, exact-identity startup/cache path rather than generic offload or
zero-copy execution. Qualify `preferred` population/fallback and `required`
consume-only failure independently; verify immutable manifests, final-layout
restore, lease cleanup, corruption/quarantine, and node-local sharing. Current
model contracts may be BF16 no-AllGather-only, so reject AllGather, online
quantization, HSDP, LoRA/adapted weights, and unrecognized load formats unless
the exact target revision explicitly adds them.

On small-HBM cards, record load/materialization peak, resident peak, encode,
each denoise wave, decode, transient per-rank HBM, host PSS, H2D, and collective
time. Test DLO AllGather and `--dlo-no-use-allgather` as different deployments.
Sweep the supported resident-block policies and publish the non-dominated
latency-HBM frontier; a single minimum-memory or fastest point is not the DLO
trade-off. Keep request-specific preparation replica-local. Only weight
materialization may enter the declared DLO collective group; never use WORLD
for per-request work in a DP deployment.
Direct checkpoint mmap preflight is limited to TP1 without HSDP or online
quantization; TP>1 falls back to the ordinary TP-aware loader and can still feed
DLO AllGather or no-AllGather after scoped E2E. HSDP remains incompatible with
DLO AllGather. A target revision may allow finalized per-tensor online FP8
through the ordinary loader and AllGather; keep every other online quantizer
fail-closed, and require model/card/topology E2E before promotion.
Use deploy YAML for DP under `--omni`; vLLM DP CLI flags are rejected.

### Gate 5: Add request-scoped acceleration safely

Server-wide Cache-DiT support does not imply a per-request quality API. Only
add request policy after defining calibrated quality tiers and a safe hook
lifecycle using `SupportsRequestScopedCacheDiT`, `CacheDiTRequestSpec`, and
`RequestScopedCacheDiTRuntime`. Validate alternating and concurrent quality
tiers, real cache hits, refresh/reset, success, error, disconnect, abort, and
the next uncached request. Do not batch different quality policies unless the
batch compatibility key and hook ownership make that safe.

Then test Cache-DiT against FP8, DLO, TP/SP, compile, and batching one
combination at a time. See [production feature patterns](references/feature-patterns.md)
for the lifecycle skeleton and compatibility matrix.

### Gate 6: Optimize only from profiles

Freeze one canonical workload manifest before profiling. Keep lossless runtime
A/Bs, accelerated paths with numerical/quality trade-offs, and production
topology studies in separate result lanes. Reconcile the client boundary as
queue + encode + denoise + video/audio decode + output transport/codec +
residual; do not hide a material residual in another stage.

Profile stage time, GPU kernels, CPU/GPU synchronization, allocations, H2D,
output payload/copies, and collectives before choosing work. Apply and A/B one
change at a time:

1. Remove synchronization and redundant denoise-loop work: hoist static prompt
   conditioning/token refinement, RoPE tables, masks/metadata, bounded AdaLN
   schedule projections, and reference scans; remove per-layer `.item()` and
   unused masks/allocations.
2. Select the fastest correct BF16 attention backend for each role and shape.
3. Prefer explicit fused QKV, gate-up + `SiluAndMul`, RMSNorm/RoPE, AdaLN, and
   layout/residual operators with guarded native fallbacks. Preserve every
   materialized dtype/rounding boundary consumed by the unfused graph.
4. Enable sparse attention only when platform, shape, topology, realized
   sparsity, quality, fallback, and end-to-end speedup are all proven.
5. Profile USP all-to-all count/bytes, packing, contiguous copies, overlap, and
   scaling efficiency. Audit replicated cross-attention before using
   `skip_sequence_parallel`. Qualify regular and accelerated Ulysses transport
   independently; activation logs, JIT/readiness time, workspace growth,
   stream ownership, maximum-shape warmup, and numerical drift are gates.
   Preserve the checkpoint's Q:KV head ratio when padding GQA for Ulysses, and
   validate strided QKV staging rather than assuming packed contiguity.
6. Evaluate text-encoder or VAE disaggregation only if stage share, transfer
   volume, reuse/concurrency, and independent scaling justify it. Tiling,
   offload, or patch parallelism is not disaggregation.

Do not combine independent optimizations into one performance claim. Preserve
raw before/after runs with fixed workload and warmup policy.

### Gate 7: Make serving interruptible and load-stable

Treat request-mode batching and step-wise execution as separate capabilities:

- request batching requires `supports_request_batch = True` and
  `DiffusionRequestBatch -> list[DiffusionOutput]`, exactly one output per
  request;
- step execution requires the complete `SupportsStepExecution` contract:
  `prepare_encode`, `denoise_step`, `step_scheduler`, and `post_decode`.

Keep all mutable scheduler, RNG, latent, mask, cache, and step state request
scoped. First validate `--step-execution --max-num-seqs 1` and step-boundary
abort; only then test heterogeneous multi-request waves. Step continuous
batching is experimental and is not automatically compatible with Cache-DiT.
Before recommending it, predeclare a useful case and success threshold, then
A/B it against request mode under the same arrival process. Structural support
or a generic bridge is not latency, throughput, or cancellation evidence for a
model.

Test queued and in-flight abort, client disconnect, timeout, OOM, worker error,
and restart. Require one terminal result, idempotent cleanup, no post-decode
after abort, no cache/temp/VRAM/request-ID leak, and a successful next request.
Inject late worker results/exceptions after cancellation and prove the result
pump stays alive instead of raising `InvalidStateError`. For large batched or
long-window video outputs, freeze dtype, range, layout, contiguity, ownership,
payload size, serializer/codec limits, and offline-versus-HTTP semantics.
Validate device-side output preparation, D2H/IPC or shared handles, remote
encoding, event-loop responsiveness, and client materialization as separate
stages, including abort and consumer failure cleanup.
A model/VAE callback that publishes ordered chunks is only a producer contract;
it is not transport, backpressure, cancellation, public streaming, or
time-to-first-frame evidence. Distinguish complete-response faster-than-playback
(`client E2E / output duration <= 1`) from streaming and report first-chunk,
steady cadence, finalization, and complete-artifact latency separately.
Run below-, near-, and above-saturation mixed-RPS soaks and report success/error
rate, throughput, queue time, memory slope, temp growth, and worker health.
Report p50/p95/p99 only from a declared arrival model with enough measured
requests for the claimed percentile. See
[serving and CI validation](references/production-validation.md).

### Gate 8: Publish recipes and four independent CI tracks

For every target vendor/card/topology, publish exact environment, serve/deploy
command, one complete JSON or multipart curl per supported task, expected
output checks, benchmark command, warmup/repeat/concurrency policy, raw metrics,
per-rank HBM/host PSS, accuracy artifact, and limitations. Mark each
CUDA/ROCm/NPU/XPU row `validated`, `unsupported`, or `not tested`; never inherit
support from the platform abstraction or another card. Re-run DLO on each
small-HBM recipe.

Maintain separate CI gates:

1. Function: strict load, API/offline positive and negative contracts.
2. Accuracy: fixed reference revision/assets/seed/schedule and scoped tolerance.
3. Performance: fixed best deployment plus memory-constrained/DLO row, raw JSON,
   explicit warmup, enough samples for each claimed statistic, and owned
   regression thresholds.
4. Reliability: long mixed-RPS soak, abort/disconnect/fault cleanup, memory and
   temporary-resource trend.

## Production Definition of Done

A model is production-ready only when:

- every official advertised task passes API parity, offline/online parity,
  strict loading, fixed-reference quality, and negative validation;
- the recommended deployment row passes correctness, isolation, abort/error
  cleanup, benchmark, and soak gates on named hardware;
- online FP8 and DLO are each implemented and either validated in scoped rows
  or explicitly limited/unsupported with fail-fast behavior;
- per-request cache quality tiers have measured speed/quality frontiers and
  request isolation, if advertised;
- CUDA/ROCm/NPU/XPU recipes describe evidence rather than inferred support;
- Function, Accuracy, Performance, and Reliability CI have owners, artifacts,
  and actionable failure output;
- accelerated paths identify every precision/quality trade-off, and the raw
  offline plus online encoded output contracts are versioned and tested;
- all unverified task/backend/cache/quant/offload/topology/hardware combinations
  remain `not tested` and are not presented as supported.

Day-0 support is still a valid milestone, but title, docs, and matrix must say
which representative task passed and which production gates remain.
