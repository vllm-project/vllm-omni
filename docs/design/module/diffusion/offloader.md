---
title: Diffusion Offloader
kind: module
status: draft
owners:
  - "@yuanheng-zhao"
  - "@david6666666"
  - "@lishunyang12"
primary_code_paths:
  - vllm_omni/diffusion/offloader/**
related_code_paths:
  - vllm_omni/diffusion/models/**
  - vllm_omni/diffusion/worker/**
depends_on:
  - diffusion_runtime.md
  - ../execution_platforms.md
validation_paths:
  - tests/diffusion/offloader/**
upstream_refs:
  - torch.nn.Module.to
last_reviewed: 2026-07-16
---

# Diffusion offloader

The offloader owns component residency, transfer scheduling, memory accounting,
prefetch, and teardown for diffusion execution.

## Candidate invariants

### OFFLOAD-INV-001: Residency is explicit

**Rule:** Each managed component MUST have one known residency state and one
owner responsible for transitions.

### OFFLOAD-INV-002: Use follows readiness

**Rule:** Runtime execution MUST NOT consume a component until its transfer to
the required device is complete.

### OFFLOAD-INV-003: Transfers preserve model state

**Rule:** Offloading MUST preserve parameter and buffer identity, dtype, device,
and correctness required by the selected execution mode.

### OFFLOAD-INV-004: Memory is bounded and released

**Rule:** Retained host and device copies MUST respect configured limits and
provide deterministic teardown.

## Safe-change guide

Test enabled and disabled paths, repeated execution, asynchronous transfer,
memory limits, failure, teardown, and numerical equivalence.
