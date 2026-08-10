---
title: Model Integration
kind: module
status: draft
owners:
  - "@tzhouam"
  - "@gcanlin"
primary_code_paths:
  - vllm_omni/model_executor/**
  - vllm_omni/model_extras/**
  - vllm_omni/plugins/**
primary_code_path_owners:
  # Additional owners scoped to a sub-path of model_executor/**
  - paths:
      - vllm_omni/model_executor/models/**
    owners:
      - "@Sy0307"
      - "@amy-why-3459"
      - "@linyueqian"
related_code_paths:
  - vllm_omni/transformers_utils/**
  - vllm_omni/tokenizers/**
depends_on:
  - input_output_modality_contracts.md
  - ar_runtime.md
validation_paths:
  - tests/model_executor/**
  - tests/model_extras/**
  - tests/plugin/**
upstream_refs:
  - vllm.model_executor
last_reviewed: 2026-07-16
---

# Model integration

Model integration adapts model-specific configuration, preprocessing, loading,
and execution behavior to stable runtime contracts.

## Candidate invariants

### MODEL-INV-001: Registration is explicit

**Rule:** A model integration MUST declare how its model class, loader, input
processor, and stage configuration are selected.

### MODEL-INV-002: Model code does not route stages

**Rule:** Model-specific code MUST NOT select or invoke a downstream omni stage.

### MODEL-INV-003: Upstream behavior is reused deliberately

**Rule:** An upstream vLLM implementation SHOULD be reused when its contract is
sufficient; an override MUST document the behavioral difference.

## Safe-change guide

Test registration, checkpoint loading, input conversion, and representative
model execution. Test shared utilities against more than one integration.
