# Module and Feature Review Routing

Read [design-contracts.md](design-contracts.md) first. Route from the frozen
head's live producer-consumer behavior, then confirm the result against its
`docs/design/` metadata. Titles and paths are evidence, not the deciding rule.

## Primary module contract

Choose one primary module and a second only when the live call path crosses a
documented dependency or primary-path exception.

| Module | Live behavior signals | Read |
| --- | --- | --- |
| Entrypoints | Offline/CLI/API ingress, validation, rendering, streaming, sessions | [entrypoints.md](../modules/entrypoints.md) |
| Configuration | Schema, defaults, deploy/stage construction, registry, topology | [configuration.md](../modules/configuration.md) |
| I/O and Modality | Requests, messages, serialization, output types, accumulation, completion | [input-output-modality.md](../modules/input-output-modality.md) |
| Error Contracts | Classification, fatality, propagation, sanitization, public rendering | [error-contracts.md](../modules/error-contracts.md) |
| Engine Orchestration | Cross-stage routing, request state, ordering, RPC correlation, terminal convergence | [engine-orchestration.md](../modules/engine-orchestration.md) |
| Stage Runtime | Placement, startup, readiness, replicas, affinity, membership, shutdown | [stage-runtime.md](../modules/stage-runtime.md) |
| OmniConnector | Cross-stage/process/device/node transport and synchronization | [omni-connector.md](../modules/omni-connector.md) |
| Model Integration | Registration, preprocessing, loading, runners, model-specific execution | [model-integration.md](../modules/model-integration.md) |
| AR Runtime | Scheduling, request/cache state, adapters, workers, upstream vLLM semantics | [ar-runtime.md](../modules/ar-runtime.md) |
| Diffusion Family | Diffusion runtime, models, batching, parallelism, or offload | [diffusion.md](../modules/diffusion.md) |
| Execution Platforms | Hardware selection, capabilities, vendor workers, kernels, patches | [execution-platforms.md](../modules/execution-platforms.md) |
| Cache Management | Cache identity, validity, reuse, eviction, reset, teardown | [cache-management.md](../modules/cache-management.md) |
| Quantization | Method/metadata selection, layer mapping, reduced precision, constraints | [quantization.md](../modules/quantization.md) |
| Observability | Metrics, logs, labels, units, correlation, lifecycle | [observability.md](../modules/observability.md) |
| Profiling | Opt-in instrumentation, traces, start/stop lifecycle, overhead | [profiling.md](../modules/profiling.md) |
| Benchmarking | Workloads, metric computation, benchmark CLI, result metadata | [benchmarking.md](../modules/benchmarking.md) |

For tests-, docs-, CI-, or recipe-only changes, route to the production module
whose contract they protect. When none exists, use the applicable evidence
check without inventing a production owner.

## Feature-design overlays

Load every category whose documented behavior, enablement, default, support, or
compatibility changes.

| Feature-design section | Read |
| --- | --- |
| Runtime and stage execution | [runtime-stage-execution.md](../features/runtime-stage-execution.md) |
| Communication and concrete connectors | [communication.md](../features/communication.md) |
| Diffusion acceleration | [diffusion-acceleration.md](../features/diffusion-acceleration.md) |
| Infrastructure and performance | [infrastructure-performance.md](../features/infrastructure-performance.md) |

## Evidence and change overlays

| Signal | Read | Optional repo-local skill |
| --- | --- | --- |
| New or expanded model, loader, processor, registry, or stage config | [model-addition-checklist.md](../checks/model-addition-checklist.md) | [`add-tts-model`](../../../add-tts-model/SKILL.md) or [`add-diffusion-model`](../../../add-diffusion-model/SKILL.md) |
| Latency, throughput, memory, scaling, precision, or quality claim | [perf-verification.md](../checks/perf-verification.md) | [`diffusion-perf-opt`](../../../diffusion-perf-opt/SKILL.md) for diffusion |
| Tests changed, absent for risky behavior, or test-only | [test-quality-evaluation.md](../checks/test-quality-evaluation.md) | [`vllm-omni-test`](../../../vllm-omni-test/SKILL.md) |
| CI, examples, docs, public behavior, or contributor evidence | [tests-docs-checklist.md](../checks/tests-docs-checklist.md) | None |
| Suitable hardware/server and runnable affected path | [verification.md](../checks/verification.md) | None |
| User asks who should review or requests owner notification | [review-requests.md](../delivery/review-requests.md) | None |

Use [`quantization`](../../../quantization/SKILL.md) for a quantization module
change and [`$vllm-omni-npu-model-runner-upgrade`](../../../vllm-omni-npu-upgrade/SKILL.md)
for an NPU execution-platform change when those skills are available.

For a bug fix, require a reachable reproduction and regression test. For a
refactor, prove parity and remove obsolete paths. For a feature, verify the
public/config contract, module integration, compatibility/default behavior,
production dispatch, feature design, and docs.

## Calibrate findings

- **P0:** security exposure, data corruption, or broad project unusability.
- **P1:** reachable runtime failure, wrong output, compatibility break, or
  unsafe lifecycle in the changed behavior.
- **P2:** real non-blocking defect with a concrete future failure mode.

Treat draft candidate invariants, missing hardware, and unsupported claims as
questions or validation gaps unless current code, tests, or policy makes them a
merge requirement.
