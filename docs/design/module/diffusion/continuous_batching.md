---
title: Diffusion Continuous Batching
kind: module
status: draft
owners:
  - "@Isotr0py"
  - "@princepride"
primary_code_paths:
  - vllm_omni/diffusion/sched/**
related_code_paths:
  - vllm_omni/diffusion/executor/**
depends_on:
  - diffusion_runtime.md
validation_paths:
  - tests/diffusion/batching/**
upstream_refs: []
last_reviewed: 2026-07-16
---

# Diffusion continuous batching

Continuous batching defines when diffusion requests are compatible, how they
share an execution step, and how each request advances independently.

## Candidate invariants

### BATCH-INV-001: Compatibility is explicit

**Rule:** Requests MAY share a batch only when all properties required by the
selected pipeline, scheduler step, shape, precision, and execution features are
compatible.

### BATCH-INV-002: Per-request state remains isolated

**Rule:** Batch assembly MUST preserve each request's random generator, progress,
conditioning, outputs, cancellation, and errors.

### BATCH-INV-003: Admission is bounded

**Rule:** Admission MUST respect configured capacity and MUST NOT rely on an
unbounded waiting or active-request collection.

### BATCH-INV-004: Batch order is not completion order

**Rule:** Output association MUST use stable request identity rather than batch
position across execution steps.

## Safe-change guide

Test heterogeneous requests, partial completion, cancellation, deterministic
seeds, capacity limits, and batch sizes of one and many.
