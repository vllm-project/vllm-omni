---
title: Input, Output, and Modality Contracts
kind: module
status: draft
architecture_state: current
owners:
  - "@Sy0307"
  - "@amy-why-3459"
document_stewards:
  - "@hsliuustc0106"
  - "@Gaohan123"
  - "@david6666666"
required_reviewers:
  - "@tzhouam"
  - "@linyueqian"
  - "@david6666666"
  - "@yenuo26"
  - "@NickCao"
primary_code_paths:
  - vllm_omni/inputs/**
  - vllm_omni/outputs/**
  - vllm_omni/request.py
  - vllm_omni/data_entry_keys.py
  - vllm_omni/engine/messages.py
  - vllm_omni/engine/serialization.py
  - vllm_omni/engine/mm_outputs.py
  - vllm_omni/engine/output_modality.py
  - vllm_omni/engine/output_processor.py
related_code_paths:
  - vllm_omni/model_executor/stage_input_processors/**
  - vllm_omni/core/sched/**
  - vllm_omni/distributed/omni_connectors/**
  - vllm_omni/diffusion/request.py
  - vllm_omni/diffusion/worker/request_batch.py
  - vllm_omni/errors.py
depends_on:
  - error_contracts.md
validation_paths:
  - tests/inputs/**
  - tests/engine/test_data_entry_keys.py
  - tests/engine/test_omni_request.py
  - tests/engine/test_omni_request_output.py
  - tests/engine/test_multimodal_accumulation.py
  - tests/engine/test_output_modality.py
  - tests/engine/test_output_processor.py
  - tests/engine/test_wire_multimodal_output.py
  - tests/utils/test_mm_outputs.py
  - tests/utils/test_mm_outputs_partition.py
  - tests/diffusion/test_diffusion_output_metadata.py
upstream_refs:
  - vllm.inputs/**
  - vllm.outputs/**
  - vllm.v1.request.Request
  - vllm.v1.engine.EngineCoreRequest
  - vllm.v1.engine.EngineCoreOutput
  - vllm.v1.engine.output_processor.OutputProcessor
invariant_namespace: IO-INV
last_reviewed: 2026-08-07
last_verified_commit: 3d7fc3b9ba3cac88d579d4dc35b78b0b641675fc
---

# Input, output, and modality contracts

These contracts define the data that may cross entrypoint, orchestration,
stage, connector, and model boundaries.

## Contract status

This document is a draft description of the current request, message,
serialization, modality, accumulation, and output types. It does not assign
semantic error ownership to the wire schema.

## Ownership boundary

This document owns request identity, prompt/input types, modality keys and
metadata, queue and wire message schemas, serialization, output types,
accumulation, completion semantics, and compatibility shims.

It does not own route-specific validation or rendering, model-specific
interpretation, connector transport mechanics, scheduling policy, or semantic
error classification. The `ErrorMessage` schema belongs here; the meaning and
public rendering of its error fields belong to `error_contracts.md`.

## Candidate invariants

These identifiers are proposals while the document is `draft`.

### IO-INV-001: Boundary data has an explicit modality

**Rule:** Data crossing a module or stage boundary MUST identify its modality
and use the corresponding validated contract.

### IO-INV-100: Request identity is stable

**Rule:** Request identity MUST be preserved across conversions, stages,
streaming updates, cancellation, and errors.

### IO-INV-101: Completion is explicit

**Rule:** Producers MUST distinguish partial updates from the terminal update
for every output modality.

### IO-INV-300: Internal objects do not leak into public protocols

**Rule:** Entrypoints MUST explicitly translate internal output objects into
public response types.

## Invariant namespace

`IO-INV` reserves `001-099` for schema ownership and dependency direction,
`100-199` for identity, modality, accumulation, ordering, and completion,
`200-299` for serialization failure and payload cleanup, and `300-399` for
upstream extension, deprecation, and compatibility. Numbers become
append-only after normative promotion.

## Safe-change guide

Test construction, validation, serialization, round trips, streaming,
accumulation, completion, optional fields, unknown fields, and compatibility
at every affected producer-consumer boundary.

## Promotion gate

- Add explicit CODEOWNERS for the primary I/O paths or confirm that fallback
  ownership is intentional.
- Document the compatibility window for deprecated `vllm_omni.engine`
  I/O-related imports.
- Verify wire round trips and rejection of unknown or incompatible schema
  fields.
- Define payload-versus-metadata rules and completion semantics for streaming
  and batched outputs.
- Obtain approval from a technical owner and independent AR, diffusion,
  connector, entrypoint, and validation reviewers as applicable.
