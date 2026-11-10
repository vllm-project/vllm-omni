# vLLM-Omni execution and evidence map

Use this to resolve the assessed commit, not as a permanent claim about every
model. The anchors below were checked at `532b33544331`. If a symbol moves, use
`rg` in the selected revision. Actual code/configuration takes precedence over
draft design prose; start at `docs/design/index.md` for relevant contracts.

## 1. Build the run's execution map

The same model can have multiple deployments. Record logical stages separately
from semantic phases, processes, replicas and ranks:

```text
API / Omni.generate
  -> AsyncOmniEngine -> Orchestrator -> selected stage replica
       -> AR / generation runner, or diffusion scheduler -> worker -> runner
       -> model-owned phases and kernels
       -> optional stage payload / KV connector -> downstream stage
  -> output assembly / D2H / media encoding -> client-visible completion
```

For each participating stage record its resolved execution type, process/device
placement, replica, upstream/downstream dependencies and output modality. Record
the request/chunk correlation and trace timebase. A missing phase measurement is
UNKNOWN; N/A needs a configuration or source-backed reason.

| Concern | Current source anchors (repository-relative) | Evidence to seek |
| --- | --- | --- |
| Offline / online boundary | `vllm_omni/entrypoints/omni.py`, `async_omni.py`, `openai/` | Actual endpoint, preprocessing, response/stream completion; do not equate pipeline return with client receipt |
| Topology / placement | `vllm_omni/config/`, `vllm_omni/engine/stage_runtime.py` | Resolved PipelineConfig and typed stage configs, deployment overrides, replica/device assignments |
| Cross-stage control | `vllm_omni/engine/async_omni_engine.py`, `orchestrator.py` | Admission, stage submissions/results, routing and chunk ordering; Orchestrator owns routing, connector owns transport |
| AR / generation | `vllm_omni/core/sched/`, `vllm_omni/worker/gpu_ar_model_runner.py`, `gpu_generation_model_runner.py`, `gpu_model_runner.py` | Prefill versus decode, admitted token batch, KV-cache hits, output materialization; inherited vLLM work may be outside this repo |
| Diffusion execution | `vllm_omni/diffusion/sched/`, `worker/diffusion_worker.py`, `worker/diffusion_model_runner.py` | Request batch versus step batch, pipeline capability checks, per-request preparation and state, executed model calls |
| Model work | `vllm_omni/diffusion/models/`, `vllm_omni/model_executor/models/` | Executed conditioning/DiT/AR/VAE/vocoder phases; model-specific masks, CFG, packing and fusion |
| Attention / compile / cache | `vllm_omni/diffusion/attention/selector.py`, `attention/backends/`, `compile.py`, `cache/`, `diffusion_kv/` | Selected implementation AND executed kernel, compiled regions, graph misses, cache eligibility/hits/invalidation |
| Transfers / residency | `vllm_omni/distributed/omni_connectors/`, `vllm_omni/diffusion/distributed/`, `offloader/` | Stage payload versus collective versus offload copies; sender/receiver readiness and exposed waits |
| Output | Diffusion runner `prepare_output_for_transport`, worker `WorkerProc._return_result` / `_async_output_loop`; `vllm_omni/worker/output/`; serving endpoint | D2H stream, shared-memory packing, CPU encode/serialize, final output visibility and backpressure |

Paths in a row after a shared directory prefix are relative to that prefix.
Keep the run's resolved map with the evidence. Optional module `context` records
component, stage, phase and source paths in machine-readable/text reports.

## 2. Choose the native measurement surface

| Question | Reuse first | Boundary caveat |
| --- | --- | --- |
| Diffusion serving latency/throughput | `tests/dfx/perf/scripts/run_diffusion_benchmark.py` + a matching `tests/dfx/perf/tests/*.json`; `benchmarks/diffusion/diffusion_benchmark_serving.py` | Inspect endpoint and output-fetch behavior; native latency fields are seconds, evidence durations are ms |
| Omni/TTS serving | `tests/dfx/perf/scripts/run_benchmark.py`; `vllm_omni/entrypoints/cli/benchmark/` and `vllm_omni/benchmarks/serve.py` | Choose TTFT, TPOT, first media packet, RTF or E2EL explicitly; these are not interchangeable |
| Offline pipeline | Existing task runner using `Omni.generate`, with the same request and sampling parameters | Excludes server queue/network; instrument the return/output boundary actually under study |
| Device timeline | `docs/contributing/profiling.md`; `OmniTorchProfilerWrapper` in `vllm_omni/profiler/omni_torch_profiler.py` | Bounded start/stop, stage/rank trace names, verify compressed traces finished writing |
| Coarse diffusion phase costs | `DiffusionPipelineProfilerMixin` in `vllm_omni/diffusion/profiler/diffusion_pipeline_profiler.py` | Synchronizes before/after wrapped methods; diagnostic only, not natural overlap or baseline latency |
| Attention reference | `benchmarks/diffusion/bench_attention_backends.py` or a run-specific matched-shape collector | Verify mask, dtype, dimensions, layouts, actual version, call count and output correctness |

Inspect the current runner's arguments/config before launching. Do not copy a
model preset just because it fits GPU memory. A short regression workload is not
a normal-quality workload. Failed requests and successful requests must both be
counted; successful-only latency can conceal failures.

For source identity, record remote HEAD/dirty paths and the imported package's
file location in the actual worker environment, plus dependency versions and
any external overrides. A matching checkout does not exclude a different installed
package. Use a dedicated checkout when alignment would disturb shared work.

## 3. Route by executed mode

### Diffusion

Read the chosen pipeline's capabilities before assuming request batching or
step execution. Current runner anchors are `execute_model`, `execute_model_batch`
and `execute_stepwise`; request mode can call `run_stage` or `forward`.

Split conditioning, initial/prefix work, repeated denoising, scheduler update,
VAE and final output. Split positive/negative CFG branches and heterogeneous
step regimes when their work differs. Repetition counts come from this workload,
not `steps * layers` guessed from the model name.

Prompt embeddings, prefix KV and exact projections have validity keys and hit/miss
behavior; they differ from approximate denoising reuse. Keep cache priming and
hit rates in the workload contract. Approximate caching, fewer steps, quantization
and changed VAE precision need explicit quality/workload permission. Offload may
be required by memory capacity: include peak live/reserved memory and residency
when proposing removal or overlap.

Inspect both eager and captured/compiled paths. Already fused QKV, FFN, norms or
modulation are facts to recognize, not new candidates. A per-step invariant must
be invariant over timestep, branch, request inputs and cache invalidation before
hoisting it. Include replacement/materialization cost and memory lifetime.

### AR, generation and streaming

Separate prefill from decode and streaming first-packet from completion. Record
actual token batch, cache reuse, chunk size and downstream consumer readiness.
A high-utilization thinker may feed a starved talker/decoder; a GPU gap can be
waiting for a required producer rather than launch overhead. Profile stage-local
work and cross-stage critical path separately. A non-AR generation/vocoder stage
must not inherit AR MFU or KV-cache assumptions just because it shares a runner.

### Multiple ranks or replicas

Record rank groups and topology for TP, Ulysses/Ring SP, CFG and VAE partitioning
actually enabled. Distinguish those collectives from inter-stage payload/KV
transport and from replicated request throughput. Single-GPU local compute does
not prove a single-stage service has no host IPC or output-transfer overhead.

## 4. Trace coverage gate before bubble accounting

Record capture interval, device/rank, available event classes, dropped events,
graph capture mode and expected versus observed replay/node counts. Check a second
view when a kernel table conflicts with obvious graph activity.

- CUDA Graph envelopes, such as Nsight's `CUPTI_ACTIVITY_KIND_GRAPH_TRACE`,
  describe a replay span, NOT the union of its busy kernels. If graph nodes are
  absent, report only envelope-external uncovered time and leave internal gaps
  UNKNOWN. Neither treating the whole envelope as busy nor treating missing
  kernels as idle is valid.
- The bundled `trace` helper accepts torch Chrome kernel/memcpy/memset events.
  It does not parse Nsight SQLite or certify collection completeness. Its ledger
  is only as complete as the recorded events supplied to it.
- With complete node events, union overlapping intervals on ONE device/timebase;
  keep request head/tail separate from internal gaps. Retain unclassified events.
  Do not sum ranks or stage spans into E2EL, or infer cross-process causality from
  visually adjacent timestamps.
- CPU range duration, synchronized phase timing, GPU event time and unprofiled
  wall time have different boundaries. A diagnostic ledger closes to its own
  window; it need not equal the unprofiled median. Report perturbation explicitly.

Do not add synchronization to every operator to manufacture attribution. Start
with a coarse low-overhead timeline, then collect shapes/stacks/NCU only for
unresolved expensive regions. Label temporary instrumentation and hash it.

## 5. Close the Omni report

The report's component rows should cover the selected path through user-visible
output, including unknown work. Bind diagnostic buckets to reviewed components;
L1 phases and L4 kernels are different views of the same time, not extra costs.
Before judging a whole pipeline, verify that no required phase, rank, cold/warm
regime or completion boundary has silently disappeared.

Keep exact commands, source map, shape/count assertions, raw artifacts, failed
collections and quality results together in a versioned run directory. Missing
weights or denied counters are evidence gaps, not permission to substitute a
checkpoint or pretend another workload's efficiency applies.
