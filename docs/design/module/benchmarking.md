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

## LingBot CI measurements

The source-local LingBot benchmark records real streamed chunk arrivals and
explicit warmup/frame/playback policies. CI collects current-run paired
reference/candidate inputs without composing semantic acceptance. Numerical
precision, WBench, VBench and relative performance are independent pytest cases
and reports; a failure in one does not block the others through a semantic
prerequisite. Each case fails on its own missing, invalid or incomplete inputs.

Three independent paired starts own the per-run reference spread and 5%
relative performance limits. The 12 FPS playback target is a measurement basis,
not a new absolute pass threshold. Numerical checks retain VAE samples and
full-video/window/content checks. Native semantic evaluators retain their own
per-dimension/binary limits and complete input-frame coverage. No frozen cohort,
held-out or external calibration-delivery workflow is part of these cases.

Long stability uses two candidate-only 600-chunk sessions with observed memory,
camera, lifecycle and cleanup assertions. Its instrumentation is excluded from
speed measurements. CPU helper tests and CI rendering validate implementation
contracts; only matching-source real hardware execution can establish native
accuracy, speed or stability outcomes.
