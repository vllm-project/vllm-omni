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

## LingBot performance and accuracy

The packaged `realtime-video` backend uses the existing `vllm bench serve
--omni` lifecycle and records real streamed chunk arrivals with explicit
warmup/frame/playback policies. Typed video workload/results keep frames
separate from tokens. The common result retains `realtime_video` raw records;
the legacy script translates flags and projects that result without a separate
request loop.

LingBot performance is configured through `tests/dfx/perf` and its ordinary
recorded JSON hardware baselines. Candidate runs do not restart a live reference
or derive a calibration/CV limit. Each workload has a distinct excluded
eight-chunk warmup; six attention-window chunks and the terminal chunk are
excluded from steady cadence. Readiness does not inject an unrecorded full
rollout. The 12 FPS playback target is a measurement basis, not an absolute
acceptance gate. The H100 and A2 baselines come from recorded CI runs (latency
x 1.10, throughput and FPS x 0.90).

`tests/e2e/accuracy/lingbot_world` contains three independent numerical, WBench
and VBench tests. Each CI run captures one candidate server
lifetime, one fixed VAE replay and one 40-chunk video per existing case, with
excluded warmup. It does not check out or restart a live official reference.
After generation releases its server, the packaged `vllm bench serve --omni`
evaluation path records actual numerical and native WBench/VBench measurements.
Each test validates its own complete input population, prints measured values
and writes its own accuracy JSON before asserting. It does not read another
test's verdict. Raw tensors, complete videos and per-dimension/binary scores
remain CI artifacts. On CUDA, WBench and VBench scores are checked against
floors derived from a recorded CI run; numerical precision has no stored
reference yet and stays baseline pending. NPU runs in **record-only** mode until
its own measurements are reviewed; recording success there does not establish
an accuracy regression pass. Generation and evaluator errors always fail. No
performance verdict depends on accuracy results.

CUDA generation uses two H100 GPUs and SP=2. NPU uses its existing four-device A2
profile, followed by CUDA evaluation; result metadata distinguishes generation
and evaluator hardware. Every accuracy leaf has a timeout of at most 120
minutes, expressing a configured bound rather than a measured duration. There
is no long-stability, cohort, held-out or external calibration-delivery workflow.
CPU tests/rendering establish implementation contracts; matching-source native
runs are still needed to establish actual accuracy, performance and capacity.
