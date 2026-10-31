---
title: Profiling
kind: module
status: draft
owners:
  - "@gcanlin"
primary_code_paths:
  - vllm_omni/profiler/**
  - vllm_omni/diffusion/profiler/**
  - vllm_omni/utils/profiler.py
related_code_paths:
  - vllm_omni/entrypoints/**
depends_on:
  - observability.md
validation_paths:
  - tests/profile/**
  - tests/utils/test_profiler.py
upstream_refs:
  - vllm.profiler
last_reviewed: 2026-07-16
---

# Profiling

Profiling owns opt-in instrumentation and trace collection used to locate
runtime bottlenecks without changing model results or execution policy.

The Qwen2.5-Omni and CosyVoice3 offline examples share the JSON argument parser
in `vllm_omni.utils.profiler`. The parser defaults to `None`, rejects malformed
JSON and non-object values before engine initialization, and leaves field
validation to the engine configuration. See the
[offline profiling guide](../../contributing/profiling.md#offline-qwen25-omni-and-cosyvoice3-examples)
for commands and the shared helper API.

AR worker profiling forwards vLLM's per-session `profiler_kwargs` to the
Torch profiler constructor. Overrides apply only while profiling is inactive;
omitting them on a later session restores the configured defaults. Stage/rank
trace naming and the default trace export handler remain in place unless the
caller explicitly overrides the handler.

## Candidate invariants

### PROFILE-INV-001: Profiling is opt-in

**Rule:** Expensive trace collection MUST be disabled by default and bounded by
an explicit start and stop lifecycle.

### PROFILE-INV-002: Instrumentation preserves semantics

**Rule:** Profiling MUST NOT change scheduling decisions, generated outputs, or
resource ownership beyond documented synchronization overhead.

### PROFILE-INV-003: Trace context is sufficient

**Rule:** Trace events SHOULD identify stage, rank, worker, operation, and
request where available without embedding user payloads.

## Safe-change guide

Test disabled overhead, repeated start and stop, cleanup, multi-rank trace
naming, and each supported profiler backend.
