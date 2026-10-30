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
processor, and pipeline configuration are selected.

### MODEL-INV-002: Model code does not route stages

**Rule:** Model-specific code MUST NOT select or invoke a downstream omni stage.

### MODEL-INV-003: Upstream behavior is reused deliberately

**Rule:** An upstream vLLM implementation SHOULD be reused when its contract is
sufficient; an override MUST document the behavioral difference.

## Safe-change guide

Test registration, checkpoint loading, input conversion, and representative
model execution. Test shared utilities against more than one integration.

## Platform-specific model adapters

Drama 3's Ascend attention compatibility, hybrid model state, and FP8 expert
adapter live under `platforms/npu/models/` and `platforms/npu/quant/`. Shared
model code selects these adapters lazily on NPU; CUDA keeps the upstream
implementations. Generic eager kernels and request-history snapshots remain
in the shared model implementation.

## DRVEXO component views

DRVEXO registers a `drvexo_unified` local-only loader through the existing
plugin and declares its Thinker/Talker/Codec topology in the pipeline
registry. Stage-local JSON views resolve namespaced tensors in one root
safetensors index; they do not materialize another checkpoint. Qwen3-TTS
subfolder reads select the configured loader as well, so Talker, Codec and
the optional first-frame decoder preserve the complete checkpoint format.
The Thinker
reuses upstream Qwen3 execution and adds an audio encoder/input projector
and explicit final-prefill candidate scoring. Stage transitions remain owned
by the pipeline/orchestrator. See [DRVEXO](../../models/drvexo_speech_8b.md)
for runtime restrictions and validation scope.

### Runner payloads at downstream entry stages

When processing a raw prompt for a downstream entry or handoff stage, preserve
`model_intermediate_buffer` on the upgraded `OmniEngineCoreRequest`, alongside
serialized routing metadata. A processed request's buffer is authoritative;
the raw prompt is only a fallback, including for other pipelines using this
shared path. On admission the runner combines that buffer with the same
request's decoded routing metadata (routing keys win conflicts), then installs
one snapshot. Omitted keys from an earlier admission are cleared, apart from
the existing streaming `meta.num_processed_tokens` / `meta.resumable` lifecycle
fields. Model-produced incremental updates still use the separate merge path.
A direct speech request carries its caller text and fixed-reply provenance in
the snapshot; dropping that state incorrectly selects the generated-token path.

Qwen3-TTS subfolder reads keep a selected `DefaultModelLoader` subclass,
including the unified-checkpoint loader. Other outer-model loader protocols
fall back to a separate native `auto` loader for tokenizer assets, without
mutating their outer load configuration or forwarding format-specific options.
