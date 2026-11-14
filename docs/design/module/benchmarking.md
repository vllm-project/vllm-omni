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

Realtime-video observations enter the shared `MultiModalsBenchmarkMetrics`
aggregation with their native populations: one full request latency, TTFC,
generated FPS and client-session RTF per successful measured rollout, and
pooled steady chunk intervals selected by the protocol adapter. Frames remain
frames; text token fields are not invented. Public names and units are
registered in `vllm_omni/metrics/definitions.py`. `--percentile-metrics` selects
`e2el`, `video_ttfc`, `video_fps`, `video_rtf` and `video_steady_interval`; all
are the realtime backend's default selection, without changing other backends.
`--metric-percentiles` keeps the shared default of 99 and controls the emitted
percentile fields. The performance recipe explicitly requests 50,95,99,
including its recorded p95 baseline. Failed warmups, measured rollouts or
finalization retain raw diagnostics and prevent the entire realtime summary
from being accepted.

RTF is a dimensionless cost ratio, with a declared timing and duration scope.
Realtime `mean_video_rtf` preserves its recorded definition: wall time after
the `session.start` send completes through receiving `session.done` and draining keepalive
cleanup, divided by `received frames / target_fps`. Connection setup is
excluded; first-chunk and terminal receive time are included. Target playback
FPS can differ from the mux FPS. Ordinary video RTF prefers server-reported
generation time over resolved content duration, falling back to full request
latency when server timing is absent. Content duration resolves payload
`duration_s`, payload `num_frames/fps`, request `seconds`, then request
`num_frames/fps`. Result `video_metric_scopes.video_rtf`
identifies these distinct boundaries; realtime results also record target and
mux FPS. The values are not interchangeable across scopes. Existing baseline
keys, numeric thresholds and legacy raw-result projections retain their meaning.

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
evaluation path measures numerical and native WBench/VBench results.
Each test validates its own complete input population, prints measured values
and writes its own accuracy JSON before asserting. It does not read another
test's verdict. Raw tensors, complete videos and per-dimension/binary scores
remain CI artifacts. On CUDA and NPU, WBench and VBench scores are checked
against per-platform floors derived from recorded CI runs. Numerical precision
has no stored reference tensors: it checks a complete, finite replay and does
not compute tensor error, PSNR or SSIM. Generation and evaluator errors always
fail. No performance verdict depends on accuracy results.

CUDA generation uses two H100 GPUs and SP=2. NPU uses its existing four-device A2
profile, followed by CUDA evaluation; result metadata distinguishes generation
and evaluator hardware. Every accuracy leaf has a timeout of at most 120
minutes, expressing a configured bound rather than a measured duration. There
is no cohort, held-out or external calibration-delivery workflow. These accuracy
and performance workflows add no stability lane. Existing DFX stability cases
retain their legacy benchmark entry and deploy configs; the entry delegates to
the shared serving benchmark.
CPU tests/rendering establish implementation contracts; matching-source native
runs are still needed to establish actual accuracy, performance and capacity.
