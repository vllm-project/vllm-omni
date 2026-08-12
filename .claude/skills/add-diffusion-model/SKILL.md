---
name: add-diffusion-model
description: Add or productionize a diffusion model in vLLM-Omni, including native ports and external adapters, multimodal request contracts, accuracy and performance baselines, quantization, caching, parallelism, offload, continuous batching, CI, recipes, and hardware qualification. Use when adding a new image, video, audio, editing, or omni diffusion model, or closing post-Day-0 correctness, performance, and production gaps for an existing model.
---

# Add a Diffusion Model to vLLM-Omni

Treat model support as a progression from a correct Day-0 path to measured,
production-ready support. Keep the main PR reviewable; record follow-up work by
capability area instead of mixing unvalidated optimizations into the initial port.

Use the MiniMax-H3 Day-0 support PR #5691 and follow-up roadmap #5700 as the model
for this split. Reinspect current repository code and docs before copying any path or
API from those references.

## 1. Write the Integration Contract

Record these decisions before editing code:

| Decision | Required result |
|---|---|
| Support type | **Native port** or **adapter-based integration** |
| Tasks | Exact task × input-source × output matrix |
| Checkpoints | Repository, pinned revision, partitions, and weight format |
| Baseline | Official/reference command, inputs, seed, schedule, and artifacts |
| Day-0 scope | Smallest correct online and offline path |
| Follow-ups | Owners or issues grouped by the four readiness pillars below |

A native port owns execution in vLLM-Omni. Use the official repository as a pinned
correctness oracle, not a runtime dependency. If another inference package remains
required at runtime, label the support adapter-based and document its boundary.

Classify the source:

- **Diffusers-native**: reuse pipeline components and port only incompatible pieces.
- **Custom**: port the required architecture and loading logic; avoid runtime clone
  or path injection. Read [custom-model-patterns.md](references/custom-model-patterns.md).
- **Hybrid**: reuse standard components and port the custom backbone or fusion path.

For transformer and attention changes, read
[transformer-adaptation.md](references/transformer-adaptation.md).

## 2. Land the Smallest Correct Vertical Slice

Implement one representative task end to end:

```text
config/model_index -> registry -> load -> normalize request -> encode
-> denoise -> decode -> DiffusionOutput -> shared offline/online example
```

Keep model-specific code under `vllm_omni/diffusion/models/<model>/`. Reuse shared
loaders, request types, attention layers, schedulers, and distributed helpers when
their contracts fit; do not fork framework infrastructure into the model folder.

Prefer extending the shared `examples/**/x_to_y.py` flow through typed model extras.
Add a model-specific example only when the shared request protocol cannot represent
the model, and state that incompatibility in the PR.

Before optimizing, require:

- strict weight and fused-shard accounting with missing weights failing startup;
- a normalized capability check before downloads, persistence, or engine submission;
- deterministic single-device parity against the reference at component, latent,
  and final-artifact levels;
- one online and one offline smoke using the same model contract;
- clear errors for unsupported tasks, sources, shapes, schedules, or partitions.

## 3. Drive Readiness Through Four Pillars

After the vertical slice works, use
[readiness-roadmap.md](references/readiness-roadmap.md) as the central plan. Track
status and evidence under four representative areas:

1. **Correctness and feature coverage** — API matrix, strict loading, reference
   parity, packed boundaries, schedules, and backend compatibility.
2. **Performance engineering** — reproducible baseline, redundant work, kernels,
   compile, quantization, and cross-step caching.
3. **Parallelism and memory scaling** — TP/SP/CFG/HSDP, offload, disaggregation,
   device-count Pareto frontiers, and hardware recipes.
4. **Production and CI** — request isolation, continuous batching, abort/cleanup,
   Function/Accuracy/Perf gates, and supported combinations.

Do not claim a capability from successful initialization alone. Every checked item
must link to a test, benchmark, artifact, trace, or documented limitation.

## 4. Optimize in Evidence Order

Freeze a representative workload and profile before changing implementation. Apply
the following order unless evidence supports another choice:

1. Remove device synchronization, repeated scans, copies, and allocations.
2. Hoist proven loop invariants into request-scoped state.
3. Select or implement the correct attention/backend path.
4. Extend compile and operator fusion without changing numerics.
5. Choose quantization, cache, offload, or parallel topology for the measured
   bottleneck and quality budget.

Measure each change independently. Report warm latency, throughput per device,
quality, and phase-specific peak memory. For caching, read
[cache-dit-patterns.md](references/cache-dit-patterns.md) and prove a measured
post-warmup step actually hit the cache.

For distributed implementation details, read
[parallelism-patterns.md](references/parallelism-patterns.md). Reject unvalidated
backend × packed-layout × topology combinations rather than silently falling back.

## 5. Add Proportional Tests and Delivery Evidence

Add one parametrized model case per distinct contract: task/input family, checkpoint
partition, scheduler path, packed layout, or topology. When the port exposes a bug
in shared infrastructure, add one focused framework regression there; do not copy
broad framework coverage into the model suite.

Cover the cheapest stable layer first:

| Layer | Evidence |
|---|---|
| L1 contract | registry/config, loader completeness, request validation, tensor logic |
| L2 smoke | shared offline and online path, startup and one small generation |
| L3 distributed | each advertised topology/backend with unequal packed samples |
| L4 hardware | reference accuracy, performance baseline, peak memory, output artifacts |

Use the repository's current test skill and CI markers instead of copying commands
from an old PR. Update the supported-model table, relevant acceleration tables, and
one runnable recipe containing only validated options.

The PR summary should stay focused:

- support type and representative tasks;
- what Day-0 or readiness gap this change closes;
- accuracy/performance evidence actually collected;
- unsupported combinations and linked follow-up issues.

## Definition of Done

Day-0 is complete when the representative task loads strictly, matches the reference,
runs through shared offline and online surfaces, and has proportional tests and docs.

Production-ready is complete only when the four-pillar roadmap has evidence for the
advertised feature combinations, hardware, quality, performance, concurrency, abort,
and cleanup behavior. Keep unchecked items visible as follow-ups; do not turn them
into implied support.

## Reference Map

- [Readiness roadmap](references/readiness-roadmap.md) — central post-Day-0 plan and evidence contract
- [Transformer adaptation](references/transformer-adaptation.md) — attention, loading, and transformer porting
- [Custom model patterns](references/custom-model-patterns.md) — non-Diffusers layouts and custom components
- [Parallelism patterns](references/parallelism-patterns.md) — TP, SP/USP, CFG, HSDP, and VAE patch details
- [Cache-DiT patterns](references/cache-dit-patterns.md) — cache integration and validation
- [Troubleshooting](references/troubleshooting.md) — common integration failures
