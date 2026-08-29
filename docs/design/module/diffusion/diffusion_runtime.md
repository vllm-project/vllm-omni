---
title: Diffusion Runtime
kind: module
status: draft
owners:
  - "@Isotr0py"
  - "@princepride"
  - "@SamitHuang"
  - "@fhfuih"
primary_code_paths:
  - vllm_omni/diffusion/sched/**
  - vllm_omni/diffusion/executor/**
  - vllm_omni/diffusion/worker/**
related_code_paths:
  - vllm_omni/diffusion/hooks/**
  - vllm_omni/diffusion/postprocess/**
depends_on:
  - ../engine_orchestration.md
  - ../input_output_modality_contracts.md
  - diffusion_model_integration.md
validation_paths:
  - tests/diffusion/diffusion_backend/**
  - tests/diffusion/hooks/**
upstream_refs:
  - diffusers.DiffusionPipeline
last_reviewed: 2026-07-16
---

# Diffusion runtime

The diffusion runtime owns request admission, scheduling, execution, progress,
output, cancellation, and cleanup inside a diffusion stage.

## Candidate invariants

### DIFF-RUNTIME-INV-001: One lifecycle owner

**Rule:** Every admitted request MUST have exactly one scheduler-owned lifecycle
until completion, cancellation, or failure.

### DIFF-RUNTIME-INV-002: Execution follows scheduler output

**Rule:** Executors and workers MUST execute scheduler decisions without
admitting, reordering, or forwarding requests independently.

### DIFF-RUNTIME-INV-003: Terminal cleanup is complete

**Rule:** Every terminal path MUST release request state, temporary tensors,
hooks, and runtime-owned resources.

### DIFF-RUNTIME-INV-004: Optional features use runtime hooks

**Rule:** Cache, profiling, offload, and parallel features SHOULD integrate at
defined hooks instead of creating another request lifecycle.

## Safe-change guide

Test success, cancellation, failure, shutdown, multiple requests, and resource
cleanup.
