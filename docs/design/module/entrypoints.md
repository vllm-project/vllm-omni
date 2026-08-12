---
title: Entrypoints and Serving Boundaries
kind: module
status: draft
architecture_state: current-plus-in-flight
owners:
  - "@alex-jw-brooks"
  - "@linyueqian"
  - "@NickCao"
document_stewards:
  - "@hsliuustc0106"
  - "@Gaohan123"
  - "@david6666666"
required_reviewers:
  - "@tzhouam"
  - "@herotai214"
  - "@yenuo26"
  - "@NickCao"
primary_code_paths:
  - vllm_omni/entrypoints/omni.py
  - vllm_omni/entrypoints/async_omni.py
  - vllm_omni/entrypoints/omni_base.py
  - vllm_omni/entrypoints/cli/**
  - vllm_omni/entrypoints/openai/**
  - vllm_omni/entrypoints/openpi/**
  - vllm_omni/entrypoints/client_request_state.py
  - vllm_omni/entrypoints/stage_utils.py
  - vllm_omni/entrypoints/utils.py
primary_path_exceptions:
  - path: vllm_omni/entrypoints/openai/errors.py
    owner: error_contracts.md
related_code_paths:
  - vllm_omni/errors.py
  - vllm_omni/inputs/**
  - vllm_omni/outputs/**
  - vllm_omni/engine/async_omni_engine.py
  - vllm_omni/config/**
  - vllm_omni/deploy/**
depends_on:
  - input_output_modality_contracts.md
  - error_contracts.md
  - engine_orchestration.md
validation_paths:
  - tests/entrypoints/test_omni_entrypoints.py
  - tests/entrypoints/test_async_omni.py
  - tests/entrypoints/test_async_omni_duplex.py
  - tests/entrypoints/test_serve.py
  - tests/entrypoints/test_stream_finish_reason.py
  - tests/entrypoints/openai/**
  - tests/entrypoints/openai_api/**
  - tests/e2e/online_serving/**
upstream_refs:
  - vllm.engine.protocol.EngineClient
  - vllm.entrypoints.openai/**
  - vllm.entrypoints.serve/**
  - vllm.renderers/**
  - vllm.v1.engine.exceptions/**
invariant_namespace: ENTRY-INV
last_reviewed: 2026-08-07
last_verified_commit: 3d7fc3b9ba3cac88d579d4dc35b78b0b641675fc
---

# Entrypoints and serving boundaries

Entrypoints translate offline, CLI, and serving requests into stable engine
operations and translate engine outputs into public responses.

## Contract status

This document describes current entrypoint responsibilities plus the boundary
under review in the open roadmap
[#5227](https://github.com/vllm-project/vllm-omni/issues/5227) and helper-move
PR [#5453](https://github.com/vllm-project/vllm-omni/pull/5453). In-flight
helper locations are not treated as current paths.

## Ownership boundary

This document owns offline API semantics, CLI and serve composition,
supported OpenAI-compatible routes, request validation and normalization,
response conversion, streaming/session behavior, and engine handoff.

It does not own configuration precedence, cross-stage routing, stage
placement, payload implementation, or semantic error classification.
`entrypoints/openai/errors.py` is an explicit primary-path exception owned by
`error_contracts.md`.

## Candidate invariants

These identifiers are proposals while the document is `draft`.

### ENTRY-INV-001: Entrypoints adapt but do not orchestrate

**Rule:** Entrypoints MUST NOT implement cross-stage routing or stage lifecycle
policy.

### ENTRY-INV-100: Public requests are normalized once

**Rule:** Public protocol values MUST be validated and converted to internal
request contracts before engine submission.

### ENTRY-INV-101: Streaming preserves request identity

**Rule:** Every streamed response MUST remain associated with the request and
output modality that produced it.

The reviewer-proposed rule that model-specific behavior should stay behind a
common adapter or processor abstraction remains an unnumbered candidate until
the entrypoint refactor makes that boundary enforceable.

## Invariant namespace

`ENTRY-INV` reserves `001-099` for boundary and dependency direction,
`100-199` for normalization, sessions, and streaming, `200-299` for
disconnect, cancellation, rendering, and cleanup, and `300-399` for upstream
route and renderer compatibility. Numbers become append-only after normative
promotion.

## Safe-change guide

Test request validation, protocol conversion, model-adapter routing,
streaming, session identity, disconnect/cancellation, and error mapping for
each affected offline or serving entrypoint.

## Promotion gate

- Reconcile the page after #5227's P0 ownership work and #5453's final
  disposition.
- Publish a supported route/transport matrix rather than claiming blanket
  OpenAI parity.
- Verify one normalization and handoff path per route family and one terminal
  outcome per streaming transport.
- Promote the `api_server.py` composition-root rule only after helper moves and
  import boundaries enforce it.
- Obtain approval from a technical owner and an independent validation
  reviewer.
