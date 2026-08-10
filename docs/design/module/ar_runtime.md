---
title: Autoregressive Runtime
kind: module
status: draft
owners:
  - "@tzhouam"
  - "@yinpeiqi"
  - "@fake0fan"
  - "@Sy0307"
primary_code_paths:
  - vllm_omni/core/**
  - vllm_omni/worker/**
related_code_paths:
  - vllm_omni/model_executor/**
depends_on:
  - engine_orchestration.md
  - model_integration.md
  - input_output_modality_contracts.md
validation_paths:
  - tests/core/**
  - tests/worker/**
upstream_refs:
  - vllm.v1.core
  - vllm.v1.worker
last_reviewed: 2026-07-16
---

# Autoregressive runtime

The AR runtime extends vLLM scheduling and worker execution for omni-stage
inputs and outputs while preserving vLLM scheduling and cache semantics.

## Candidate invariants

### AR-INV-001: vLLM owns base scheduling semantics

**Rule:** Omni schedulers MUST preserve upstream request-state and cache
transitions unless an Omni-specific difference is documented and tested.

### AR-INV-002: Omni data crosses explicit adapters

**Rule:** Modality-specific stage data MUST be converted at an input or output
adapter, not injected through unrelated scheduler state.

### AR-INV-003: Workers execute assigned work

**Rule:** Workers and model runners MUST NOT implement cross-stage routing.

## Safe-change guide

Test request lifecycle, abort, cache state, and every affected worker execution
mode against the supported upstream vLLM contract.
