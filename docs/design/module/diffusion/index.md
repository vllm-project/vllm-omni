---
title: Diffusion Modules
kind: module-index
status: draft
owners:
  - "@Isotr0py"
  - "@princepride"
  - "@SamitHuang"
  - "@wtomin"
  - "@ZJY0516"
  - "@RuixiangMa"
  - "@david6666666"
  - "@xuechendi"
primary_code_paths:
  - vllm_omni/diffusion/**
related_code_paths:
  - vllm_omni/platforms/**
depends_on:
  - ../engine_orchestration.md
  - ../input_output_modality_contracts.md
validation_paths:
  - tests/diffusion/**
upstream_refs:
  - diffusers
last_reviewed: 2026-07-16
---

# Diffusion modules

Diffusion is a module family containing runtime, model integration, batching,
distributed execution, and memory-management subsystems.

## Module documents

- [Diffusion runtime](diffusion_runtime.md)
- [Diffusion model integration](diffusion_model_integration.md)
- [Continuous batching](continuous_batching.md)
- [Parallelism](parallelism.md)
- [Offloader](offloader.md)

Cache, quantization, profiling, and benchmarking follow their top-level module
documents.
