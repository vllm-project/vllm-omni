---
title: Diffusion Model Integration
kind: module
status: draft
owners:
  - "@Isotr0py"
  - "@wtomin"
  - "@ZJY0516"
primary_code_paths:
  - vllm_omni/diffusion/models/**
  - vllm_omni/diffusion/model_loader/**
related_code_paths:
  - vllm_omni/diffusion/layers/**
  - vllm_omni/diffusion/lora/**
  - vllm_omni/diffusion/utils/**
depends_on:
  - diffusion_runtime.md
  - ../input_output_modality_contracts.md
validation_paths:
  - tests/diffusion/models/**
  - tests/diffusion/model_loader/**
  - tests/diffusion/layers/**
  - tests/diffusion/lora/**
upstream_refs:
  - diffusers.DiffusionPipeline
last_reviewed: 2026-07-16
---

# Diffusion model integration

Diffusion model integration owns pipeline contracts, registration, checkpoint
loading, adapters, shared layers, and model-specific processing.

## Candidate invariants

### DIFF-MODEL-INV-001: Pipelines implement one runtime contract

**Rule:** A pipeline MUST declare its supported modalities, configuration,
inputs, outputs, loading path, and runtime capabilities.

### DIFF-MODEL-INV-002: Registration is the selection boundary

**Rule:** Runtime code MUST select model implementations through the registry or
loader contract rather than scattered model-name conditionals.

### DIFF-MODEL-INV-003: Model code does not schedule requests

**Rule:** Pipeline code MUST NOT own admission, batching, cancellation, or
cross-stage routing.

### DIFF-MODEL-INV-004: Shared behavior stays shared

**Rule:** Model directories SHOULD contain only genuine model differences.

## Safe-change guide

Test registry selection, checkpoint loading, minimal inference, input and output
contracts, and every declared optional capability.
