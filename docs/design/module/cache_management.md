---
title: Cache Management
kind: module
status: draft
owners:
  - "@Isotr0py"
  - "@princepride"
  - "@SamitHuang"
primary_code_paths:
  - vllm_omni/diffusion/cache/**
related_code_paths:
  - vllm_omni/experimental/ar_diffusion/kv_cache/**
  - vllm_omni/core/**
depends_on:
  - ar_runtime.md
  - diffusion/diffusion_runtime.md
validation_paths:
  - tests/diffusion/cache/**
upstream_refs:
  - vllm.v1.core.kv_cache_manager
last_reviewed: 2026-07-16
---

# Cache management

Cache management defines reusable state, cache identity, validity, lifecycle,
and eviction across AR and diffusion execution.

When upstream reports a multimodal encoder-cache size mismatch, both AR and
generation schedulers drain that request into a terminal error output and
discard its pending prewarm payload. A rejected cache identifier must not
leave a request waiting indefinitely or replay the same error on later steps.

When HiSparse defers asynchronous KV loads, a completed nonempty scheduler
step advances the upstream fence and drains both deferred frees and loads.
Empty steps leave the fence unchanged so a reused block cannot be loaded
while an earlier GPU step still reads it.

Diffusion supplies worker attention specs to the native KV planner and uses
its default grouping. Its model-config adapter declares that it does not use
a registered model's custom `SupportsKVCacheGrouping` hook.

## Candidate invariants

### CACHE-INV-001: Cache identity is complete

**Rule:** A cache key MUST include every input and execution property that can
change the reused value.

### CACHE-INV-002: Reuse preserves correctness

**Rule:** A cache optimization MUST provide a disabled path and MUST NOT reuse
state after its validity conditions change.

### CACHE-INV-003: Ownership has an end

**Rule:** Request-scoped and model-scoped cache entries MUST have explicit
reset, eviction, or teardown behavior.

## Safe-change guide

Test hit, miss, invalidation, disabled, concurrent-request, and teardown paths.
