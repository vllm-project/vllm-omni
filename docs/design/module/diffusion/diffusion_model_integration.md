---
title: Diffusion Model Integration
kind: module
status: draft
owners:
  - "@fhfuih"
  - "@Bounty-hunter"
  - "@wtomin"
  - "@RuixiangMa"
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

## Exact conditioning projection reuse

Models with repeated conditioning projections can use `ExactProjectionCache`
from `vllm_omni.diffusion.cache`. It stores actual projection outputs for identical
conditioning tensors and unchanged weights, with a byte budget and TP-wide hit
coordination. It has no model names, sampling schedules, projection layouts, or
artifact formats built in. MiniMax-H3 is its first model integration.

Keep one cache per model instance and prepare each conditioning tensor once
before its projections. For a leaf linear with fixed SiLU preprocessing:

```python
from torch.nn.functional import silu

from vllm_omni.diffusion.cache import ExactProjectionCache

cache = ExactProjectionCache(max_bytes=256 * 1024**2)

# Inside an inference forward, with gradients disabled:
cache.prepare(conditioning)
modulation = cache.project(
    "block.0.modulation",
    projection,
    conditioning,
    lambda: projection(silu(conditioning)),
)
```

The original computation must depend only on the supplied conditioning tensor,
the leaf projection's weights/buffers, and fixed preprocessing. Include prompt,
guidance, modality, or other request-dependent conditioning in the actual input;
a timestep-only key is unsafe when those affect the result. Do not mutate the
input between `prepare` and `project`, or modify outputs returned on cache hits.
Use separate cache instances for concurrent forwards with different inputs.

Call `clear()` when loading or switching adapters, changing projection behavior,
or moving the model. Weight-version checks invalidate individual entries.
Compilation, gradient-enabled execution, projection hooks, and unversioned
inference weights use the original computation. Validate cold/warm parity,
invalidation, and rank-local misses when integrating another model.
Hook checks include global hooks and hooks on wrapped submodules. Numerical
settings are captured at each projection call, so entering an autocast context
after `prepare()` cannot reuse an output from a different precision.

Validated offline results can be supplied through `_lookup_precomputed()`;
this extension is consulted only at TP1 on runtime-cache misses. Artifact
validation, weight binding, and model-specific layouts remain in the adapter.
H3's adapter retains its main/Ref2VA sidecars and their eager-only admission.

The cache retains projection results and leaves all original weights available.
It does not reduce weight residency or cancel layerwise prefetch. A future
integration that skips weight transfers must coordinate with `OffloadPlan`
before prefetch and provide a safe miss path; moving weight storage currently
causes conservative cache invalidation.
