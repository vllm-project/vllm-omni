---
title: Execution Platforms
kind: module
status: draft
owners:
  - "@gcanlin"
  - "@tjtanaa"
  - "@xuechendi"
primary_code_paths:
  - vllm_omni/platforms/**
  - vllm_omni/attention/**
primary_code_path_owners:
  # Additional owners scoped to attention kernels (quantized/custom attention)
  - paths:
      - vllm_omni/attention/**
    owners:
      - "@david6666666"
      - "@lishunyang12"
  # Additional owner for the Ascend NPU platform tree
  - paths:
      - vllm_omni/platforms/npu/**
    owners:
      - "@FayeSpica"
  # Additional owner for the ROCm platform tree
  - paths:
      - vllm_omni/platforms/rocm/**
    owners:
      - "@haic0"
related_code_paths: []
depends_on:
  - ar_runtime.md
  - diffusion/index.md
validation_paths:
  - tests/platforms/**
upstream_refs:
  - vllm.platforms
last_reviewed: 2026-07-16
---

# Execution platforms

Execution platforms isolate hardware-specific capability detection, worker
selection, patches, kernels, and configuration adjustments.

External platforms use `Operator.register_oot(Implementation)` before model
construction and implement `forward_oot`. Overrides preserve the operator's
constructor, parameters, and outputs; `forward_native` is the default fallback.
Diffusion attention implementations dispatch external platforms to
`forward_oot` the same way. Only `SDPAImpl` provides it so far; resolution
rejects an explicitly selected backend whose implementation does not.

## Candidate invariants

### PLATFORM-INV-001: Capabilities are explicit

**Rule:** Hardware-dependent behavior MUST be guarded by platform selection or
capability detection and MUST fail clearly when unsupported.

### PLATFORM-INV-002: Portable code remains portable

**Rule:** Platform-neutral modules MUST NOT import a vendor implementation
directly when a common selection interface exists.

### PLATFORM-INV-003: Overrides are minimal

**Rule:** A platform implementation SHOULD override only behavior that differs
from the common or upstream implementation.

## Safe-change guide

Validate common import paths without the target accelerator and run focused
tests in a fresh `uv` environment on every affected device platform.

## MiniMax H3 image NPU adapter

The NPU model-patch hook owns opt-in image VAE graph replay, independent decoder
streams, and approximate encoder/decoder tile batching. Other platforms and
pipelines retain their existing behavior. Graph buffers belong to the VAE
instance, and weight relocation invalidates its captures. User configuration
and precision limits are documented in the
[Ascend image guide](../../user_guide/diffusion/minimax_h3_image_npu.md).

H3 deterministic execution restores the caller's enabled and warn-only modes
when forward returns or raises, so inline engine setup and teardown leave no
persistent deterministic-mode change. VAE graphs disable NPU internal formats;
`npu_disable_internal_format` exposes that worker setting independently for
matched graph-off controls. `npu_disable_jit_compile` independently matches the
graph path's disabled JIT compilation mode. Both options default to false and
apply to the worker, including the DiT. Graph replica outputs are copied on the producer
stream before reuse, then joined to the consumer stream.
