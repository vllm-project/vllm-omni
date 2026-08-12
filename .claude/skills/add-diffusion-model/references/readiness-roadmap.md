# Diffusion Model Readiness Roadmap

Use this file as the single post-Day-0 roadmap. Keep one table per capability area,
link every completed row to evidence, and keep unsupported combinations explicit.

## Contents

1. [Correctness and feature coverage](#1-correctness-and-feature-coverage)
2. [Performance engineering](#2-performance-engineering)
3. [Parallelism and memory scaling](#3-parallelism-and-memory-scaling)
4. [Production and CI](#4-production-and-ci)
5. [Roadmap table](#roadmap-table)

## Integration Summary

Record the model, support type, pinned reference revision, representative workloads,
target hardware, and Day-0 PR. Then maintain the four pillars below.

Status values: `done`, `in progress`, `limited`, and `todo`.

## 1. Correctness and Feature Coverage

| Area | Completion evidence |
|---|---|
| API alignment | Exact task × source × shape × schedule × output matrix; shared validation before side effects |
| Native/adapter boundary | Native code has no reference-runtime dependency; adapters name and pin the external runtime |
| Weight loading | Expected, loaded, unexpected, ignored-with-reason sets; every fused destination proves all source shards |
| Reference parity | Frozen inputs and seeds; component, denoise trajectory, final latent, and artifact comparison |
| Packed attention | Unequal multi-sample boundaries, cumulative lengths, padding, modality and CFG ownership preserved per backend |
| Compatibility | Each advertised scheduler, attention backend, task partition, cache, quantization, and topology is tested |

Define tolerances from the reference implementation's own variance. Use metrics
appropriate to the modality, and retain failure artifacts. Never infer correctness
from tensor shape, a successful server start, or a visually plausible sample alone.

Validate request count, per-file and aggregate bytes, decoded duration/frames/pixels,
timing offsets, output count, and checkpoint partition before persistence or engine
submission. Clean partial resources on rejection.

## 2. Performance Engineering

Freeze model revision, task, prompt/media hashes, resolution/duration, seed, scheduler,
step count, dtype, attention backend, device topology, and concurrency. Exclude one
declared warmup and retain raw run data.

Report these phases separately:

- load/materialization peak HBM and host memory;
- post-load resident HBM and host memory;
- warm-generation peak HBM;
- encode, denoise, and decode peaks and latency;
- full process-tree host PSS when offload or HSDP is enabled.

For each optimization, show before/after warm latency, throughput, throughput per
device, peak memory, and quality. Profile synchronization, allocation, copies,
communication, and graph breaks before choosing work.

Prioritize representative opportunities:

| Area | Typical target |
|---|---|
| Redundant work | Static conditioning, token refinement, packed metadata/masks, reference scans, AdaLN schedules |
| Attention and kernels | Correct packed backend, native GQA, sparse attention, RMSNorm/RoPE/SwiGLU/gated-residual fusion |
| Compile | Stable repeated regions with explicit eager boundaries for hooks and lifecycle code |
| Quantization | DiT and encoder precision selected by measured quality and memory trade-offs |
| Cross-step cache | TeaCache/Cache-DiT with real hit evidence, isolation, reset, and quality/speed frontier |

Do not combine several unmeasured changes into one speed claim.

## 3. Parallelism and Memory Scaling

Select topology from the bottleneck, not from feature availability:

| Goal | Candidate paths |
|---|---|
| Lower latency | TP, SP/USP, CFG parallel, faster attention, compile/fusion |
| Fit limited HBM | HSDP, layerwise/distributed offload, quantization, VAE tiling |
| Scale throughput | Replication/DP, continuous batching, component disaggregation |
| Isolate heavy stages | Text-encoder or VAE disaggregation; component-selective residency/offload |

Create process groups collectively and guard component construction and collectives
by membership. Preserve packed sample metadata through every split and gather. Test
Ulysses, Ring, CFG, TP, and HSDP combinations separately when advertised.

For topology selection, create a separate Pareto frontier for each inference-step
count. Put user latency on the x-axis and throughput per device on the y-axis. Plot
only device counts actually validated, and report per-rank HBM plus communication
time. A lower resident weight footprint does not prove a lower transient peak.

Publish hardware recipes only for measured configurations. Record device model and
count, interconnect, driver/runtime, dtype, topology, concurrency, latency,
throughput, per-rank peak HBM, host PSS, and limitations.

## 4. Production and CI

Keep mutable state request-scoped: latents, scheduler/timestep, generator,
conditioning, packed metadata, cache state, output chunks, and temporary resources.
For step execution, validate concurrent admission, independent completion, and abort
at each boundary. Cleanup must be idempotent after success, validation failure,
engine failure, disconnect, and abort.

Use four evidence tracks:

| Track | Required signal |
|---|---|
| Function | Tasks, source combinations, limits, schedules, backends, topology, abort |
| Accuracy | Frozen reference, per-hardware tolerance, artifacts on failure |
| Performance | Fixed workload, warmup policy, repeated runs, regression threshold |
| Reliability | Concurrency, request isolation, OOM/timeout behavior, cleanup and restart |

Keep model-specific tests proportional: add one parametrized case per distinct model
contract. If the work fixes shared infrastructure, add one focused regression at the
framework layer and reuse it across models.

## Roadmap Table

Maintain one compact table in the issue or PR:

| Pillar | Item | Status | Issue/PR | Hardware | Evidence or limitation |
|---|---|---|---|---|---|
| Correctness | Representative task parity | todo | | | |
| Performance | Fixed-workload baseline | todo | | | |
| Scaling | Target topology/offload path | todo | | | |
| Production | Continuous batching and abort | todo | | | |
| CI | Accuracy and performance gates | todo | | | |

Split work into follow-up PRs by these capability areas. Update status only after
the linked evidence exists; keep partial hardware or feature coverage marked
`limited`.
