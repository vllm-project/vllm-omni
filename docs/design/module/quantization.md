---
title: Quantization
kind: module
status: draft
owners:
  - "@david6666666"
  - "@Isotr0py"
  - "@lishunyang12"
primary_code_paths:
  - vllm_omni/quantization/**
  - vllm_omni/diffusion/quantization/**
related_code_paths:
  - vllm_omni/platforms/*/quant/**
depends_on:
  - model_integration.md
  - diffusion/diffusion_model_integration.md
validation_paths:
  - tests/quantization/**
  - tests/diffusion/quantization/**
upstream_refs:
  - vllm.model_executor.layers.quantization
last_reviewed: 2026-07-16
---

# Quantization

Quantization owns method registration, checkpoint interpretation, layer
replacement, and runtime constraints for reduced-precision execution.

## Candidate invariants

### QUANT-INV-001: Checkpoint and runtime methods agree

**Rule:** The selected method MUST validate checkpoint metadata, supported
layers, device capabilities, and required dependencies before use.

### QUANT-INV-002: Unsupported combinations fail clearly

**Rule:** Quantization MUST NOT silently fall back to a different precision or
method when a requested combination is unsupported.

### QUANT-INV-003: Numerical validation is required

**Rule:** A new method or layer mapping MUST include correctness comparison
against a defined reference result.

## Safe-change guide

Test loading, representative layers, metadata, platform guards, and numerical
tolerances separately from performance.
