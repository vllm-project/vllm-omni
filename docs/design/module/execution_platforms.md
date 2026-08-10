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
related_code_paths:
  - vllm_omni/attention/**
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
