---
title: Benchmarking
kind: module
status: draft
owners:
  - "@alex-jw-brooks"
  - "@Bounty-hunter"
primary_code_paths:
  - vllm_omni/benchmarks/**
  - vllm_omni/entrypoints/cli/benchmark/**
related_code_paths:
  - benchmarks/**
depends_on:
  - entrypoints.md
  - profiling.md
validation_paths:
  - tests/benchmarks/**
  - tests/dfx/perf/**
upstream_refs:
  - vllm.benchmarks
last_reviewed: 2026-07-16
---

# Benchmarking

Benchmarking defines reproducible workloads, metrics, warm-up, measurement,
and result metadata for comparing vLLM-Omni performance.

## Candidate invariants

### BENCH-INV-001: Results are reproducible

**Rule:** Results MUST identify the model, revision, commit, hardware,
configuration, workload, concurrency, warm-up, and measurement window.

### BENCH-INV-002: Correctness precedes performance

**Rule:** A benchmark MUST validate successful and semantically acceptable
outputs before using timings as performance evidence.

### BENCH-INV-003: Metric definitions are explicit

**Rule:** Metrics MUST state their unit, population, and aggregation.

## Safe-change guide

Test workload parsing and metric calculations with deterministic fixtures. Do
not compare materially different workloads or runtime settings.

The [LingBot World regression suite](https://github.com/vllm-project/vllm-omni/blob/main/tests/dfx/lingbot_world/README.md)
wraps the served realtime workload in blocking pytest jobs. It measures a frozen
reference on the same hardware/dependencies, calibrates from three independent
server starts, and checks candidate performance separately. Fixed-input VAE
replay and instrumented lifecycle/memory checks run outside timing windows.
The PR smoke executes its recorded chunk/warmup/frame-rate/timeout settings
explicitly, validates raw output with the shared metric checker, and retains
those metrics. CLI defaults must not silently change the recorded workload;
short smoke timings are functional diagnostics rather than calibrated limits.
CUDA generation may retry a lost Buildkite agent once while retaining the
original attempt. Numerical/quality failures, OOM, cancellation and timeout
remain blocking; consumers validate the current attempt's complete artifacts.
Calibration reuse requires recorded optimization settings: its nonempty
`engine_environment` string mapping must match the measured contract. Missing
settings are invalid even when both records omit the field.
Reference-noise calibration also binds recorded playback/mux frame rates and
both the measurement warmup and inference warmup lengths. Missing or changed
values reject reuse; held-out scenes and rollout lengths remain supported.
Nightly CUDA uses two recorded-zero H100 reference cohorts and pinned-decoder
pixel summaries. Its fixed population owns reference-spread ceilings; loading
that policy must preserve global weekly/NPU limits and candidate tolerances.
Historical unrecorded records remain separate diagnostic fixtures. Fitted
calibration and negative controls do not replace fresh target-hardware acceptance.
Calibration compatibility errors retain raw measurements, numerical results and
quality scores for review; candidate comparisons stay unjudged and block acceptance.
CI pixel comparisons execute in the same verified dependency environment as
native evaluators, using isolated Python workers with retained versions, logs and
incomplete results on failure. Fixed reference pixel summaries require recorded
decoder dependency versions matching that environment; missing metadata rejects reuse.
CUDA weekly generation and evaluation use six matching case shards, each with
three of the 18 fixed cases. Each GPU evaluator processes 18 videos within a
240-minute job budget after six-case scoring exceeded an observed 300-minute
limit. Job partitioning preserves three independent starts per role, every
case, rollout length, warmup and regression budget.
NPU weekly collection separates each case shard's three independent
reference/candidate pairs into bounded jobs. A mandatory CPU gate verifies the
complete matching population per shard before quality evaluation; missing or
cross-shard pairs fail that gate. Partitioning jobs must preserve all cases,
sessions, chunks, starts, hardware and regression budgets.
GPU evaluators consume checksum-verified CUDA or NPU videos and enforce native
per-dimension quality limits. Persist evaluator hardware, dependencies and input
hashes before native workers start, so forced termination retains a failed report
with its provenance. Incomplete populations, unstable references,
missing prerequisites and skipped selected CI tests cannot establish acceptance.
