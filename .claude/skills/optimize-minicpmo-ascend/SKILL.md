---
name: optimize-minicpmo-ascend
description: Optimize and validate MiniCPM-o 4.5 inference in vLLM-Omni for the OpenBMB Ascend high-performance inference competition. Use when adapting MiniCPM-o 4.5 to Ascend NPU, tuning TTFT or first-response latency, single-chunk or end-to-end latency, throughput, concurrent sessions, NPU utilization, memory, stability, accuracy, multimodal quality, benchmarking, profiling, or preparing a reproducible competition submission.
---

# Optimize MiniCPM-o 4.5 for the Ascend Competition

Treat the competition as a gated multi-objective optimization problem:

1. Pass effect, correctness, functional, and stability checks.
2. Improve the official performance metrics without crossing those gates.
3. Reproduce the result in the official Ascend environment from complete code,
   configuration, benchmark scripts, and documentation.

Do not optimize a private proxy at the expense of the official objective. Do
not invent a composite score while the organizer has not published metric
weights.

## Required Context

Read [references/competition-rules.md](references/competition-rules.md) before
planning or evaluating an optimization. Read
[references/repo-map.md](references/repo-map.md) before editing this repository.

At the start of each competition task:

1. Recheck the official competition page, toolkit page, announcements, and
   latest starter kit.
2. Record the rule and toolkit version or retrieval date used by the run.
3. Let newer official material override this skill's dated rule snapshot.
4. Resolve the official hardware, image, driver, CANN, PyTorch, torch-npu,
   vLLM, vLLM-Ascend, model, benchmark, request schema, and pass thresholds.
5. List every unresolved item. Never silently replace an unpublished rule with
   a guess.

## Optimization Objective

Use this priority order until the official scoring formula says otherwise:

1. **Eligibility gate:** model loads and serves the required text, image,
   audio, and video workloads; outputs remain correct; multimodal and speech
   quality remain acceptable; the service is stable.
2. **Latency:** reduce TTFT or first response, first audio response when
   applicable, single streaming chunk latency, and E2E latency.
3. **Capacity:** increase throughput and stable concurrent sessions.
4. **Efficiency:** improve NPU utilization and reduce HBM/host memory or
   per-request resource cost when this helps official latency/capacity.
5. **Engineering:** keep the solution explainable, portable to the official
   environment, and reproducible from the submitted materials.

Maintain a Pareto table rather than hiding tradeoffs in an unofficial scalar:

| Config | Effect gate | TTFT p50/p95 | Chunk p50/p95 | E2E p50/p95 | Throughput | Stable sessions | NPU util | Peak HBM | Failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

Use the exact statistics emitted by the official benchmark once it is
available. Until then, report both central tendency and tail behavior, and
label every local metric as a proxy.

## Workflow

### 1. Freeze a Trustworthy Baseline

Capture all of the following before changing code:

- Git SHA and dirty-worktree diff.
- NPU model/count/topology and CPU/RAM limits.
- OS, container/image digest, driver, firmware, CANN, Python, PyTorch,
  torch-npu, vLLM, vLLM-Ascend, and vLLM-Omni versions.
- Model source, revision, checksum, dtype, quantization, and sampling settings.
- Exact deploy config and stage placement.
- Exact server, warmup, correctness, benchmark, profiling, and monitoring
  commands.
- Dataset/version, input modality and size distributions, output settings,
  concurrency/request-rate policy, timeout, warmup count, repeat count, and
  random seed.
- TTFT, first audio response if present, chunk latency/jitter, E2E latency,
  throughput, stable concurrent sessions, NPU utilization, peak HBM/host
  memory, error rate, and effect/quality results.

Separate text-only requests from text-plus-audio requests. Text-only isolates
the Thinker path; audio output includes Talker, Code2Wav, connectors, and
streaming behavior. Never compare those modes as if they were the same
workload.

Warm the service before measuring. Keep profiler runs separate from score
runs because profiling overhead invalidates final performance numbers.

### 2. Map Metrics to the Pipeline

Use the three-stage MiniCPM-o 4.5 architecture to narrow hypotheses:

- **Stage 0, Thinker:** multimodal preprocessing/encoding, prefill, text
  decoding, and text TTFT.
- **Stage 1, Talker:** codec-token generation, scheduling, batching, and delay
  before audio work can begin.
- **Stage 2, Code2Wav:** time to first audio packet, per-chunk compute,
  chunk continuity, vocoder throughput, and audio completion time.
- **Orchestrator/connectors/API:** queueing, inter-stage transfer, serialization,
  request admission, end-to-end latency, concurrency, and cleanup stability.

Measure per-stage queue, compute, transfer, and output time when possible. A
global E2E regression can otherwise hide a faster kernel behind worse
queueing or chunk scheduling.

### 3. Search in Risk Order

Test one hypothesis per comparison. Prefer this order:

1. **Measurement and compatibility:** make the official workload run; remove
   benchmark errors; verify NPU-specific paths and output contracts.
2. **Configuration:** stage placement, memory budgets, `max_num_seqs`, batched
   token limits, request admission, async chunk sizes/context, ACL graph or
   compilation mode, dtype, and framework-supported parallelism.
3. **Runtime:** reduce host-NPU synchronization, Python work in hot loops,
   repeated allocation/copy/conversion, unnecessary serialization, connector
   polling, and avoidable queueing.
4. **Model hot paths:** optimize only operators proven hot on the official
   shapes. Prefer supported Ascend/CANN kernels and platform-local patches.
5. **Precision or algorithm changes:** quantization, approximation, cache
   changes, or altered sampling only with a full effect and quality gate.

Do not port CUDA assumptions mechanically to NPU. Confirm operator support,
layout, dtype, graph behavior, dynamic-shape behavior, and synchronization on
the actual Ascend stack.

### 4. Run Controlled Experiments

For each candidate:

1. State the bottleneck evidence and expected metric impact.
2. Change one primary variable.
3. Keep environment, workload, sampling, seed, warmup, and repetition policy
   fixed.
4. Save raw benchmark JSON/logs, NPU monitoring, and output artifacts.
5. Compare against same-session baseline variance.
6. Run correctness/effect checks before accepting a performance result.
7. Retest concurrency and long-running stability; request-local audio state can
   fail only under load.
8. Record wins, losses, confidence, and rollback condition.

Never claim a win from a cold-start run, a profiler trace, a failed request,
different output length, truncated generation, disabled modality, or a
changed dataset.

### 5. Profile Only After Baseline Evidence

Use low-overhead service and NPU metrics first. When a bottleneck remains
unclear, use the repository's NPU profiler and analyze traces offline.

Inspect:

- CPU/API time before NPU work begins.
- Stage queue wait and inter-stage gaps.
- NPU idle regions, synchronization, copies, layout conversions, and small
  repeated operators.
- Dynamic-shape recompilation or graph misses.
- Rank/stage imbalance and HBM pressure.
- Talker/Code2Wav chunk cadence, batching compatibility, and cache lifetime.

Use traces to choose an experiment, not as the final benchmark result.

### 6. Apply Acceptance Gates

Accept an optimization only when all applicable gates pass:

- Official or closest available effect/correctness check passes.
- Required input and output modalities still work.
- Text and audio content are not empty, truncated, duplicated, or reordered.
- Streaming chunks are continuous enough for the official/user-facing target.
- Concurrent requests do not share or leak request-local state.
- The gain exceeds measurement noise on the target workload.
- Peak memory, error rate, and long-run stability do not become disqualifying.
- A clean environment can reproduce the commands and result.

For quantization or altered precision, compare task accuracy plus reviewable
text/audio/multimodal outputs. Treat an inconclusive quality check as a failed
gate.

## Implementation Rules

- Preserve existing repository patterns and NPU platform guards.
- Keep GPU behavior unchanged unless the task explicitly includes it.
- Add focused tests for changed scheduling, request state, chunk boundaries,
  cache lifetime, tensor shapes/dtypes, and output assembly.
- Run CPU/unit tests first, then an Ascend smoke test, then the official or
  closest benchmark and effect suite.
- Keep benchmark-specific code out of model logic. Do not hardcode evaluation
  samples, outputs, timings, or environment-specific shortcuts that violate
  fair evaluation or general inference behavior.
- Do not enable caches or alter request semantics unless the official rules
  and benchmark permit them.
- Keep submission artifacts below any official size/time limits published in
  the latest starter kit.

## Result Report

Every optimization report must include:

- Objective and official rule/toolkit version used.
- Baseline and candidate SHAs/config IDs.
- Exact commands and environment.
- A baseline-versus-candidate metric table.
- Effect/correctness and stability results.
- NPU utilization and memory evidence.
- Explanation of the mechanism, not only the measured number.
- Known tradeoffs, unsupported workloads, and residual risk.
- Reproduction steps and artifact paths.
- Decision: keep, reject, or investigate further.

## Submission Gate

Before calling the work competition-ready, verify:

- Reproducible source and configuration are complete.
- Benchmark scripts use the official request schema and metric definitions.
- Performance report includes raw and summarized results.
- Deployment documentation lists all dependencies and commands.
- Model/effect validation and multimodal output checks are included.
- The package runs in the official unified Ascend environment.
- No undeclared model replacement, hidden service, dataset leakage, or
  machine-local dependency is required.
- A second clean run reproduces the claimed result.
