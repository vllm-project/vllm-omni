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
  - vllm_omni/entrypoints/async_omni_base.py
  - vllm_omni/entrypoints/omni_base.py
  - vllm_omni/entrypoints/duplex_omni.py
  - vllm_omni/entrypoints/cli/**
  - vllm_omni/entrypoints/openai/**
  - vllm_omni/entrypoints/openpi/**
  - vllm_omni/entrypoints/client_request_state.py
  - vllm_omni/entrypoints/stage_utils.py
  - vllm_omni/entrypoints/utils.py
  - vllm_omni/entrypoints/duplex/**
  - vllm_omni/clients/**
  - vllm_omni/protocol/**
primary_path_exceptions:
  - path: vllm_omni/entrypoints/openai/errors.py
    owner: error_contracts.md
  - path: vllm_omni/entrypoints/duplex/**
    owner: ../fullduplex.md
  - path: vllm_omni/entrypoints/duplex_omni.py
    owner: ../fullduplex.md
  - path: vllm_omni/clients/**
    owner: ../fullduplex.md
related_code_paths:
  - vllm_omni/errors.py
  - vllm_omni/inputs/**
  - vllm_omni/outputs/**
  - vllm_omni/engine/omni_engine_base.py
  - vllm_omni/engine/async_omni_engine.py
  - vllm_omni/engine/duplex_omni_engine.py
  - vllm_omni/config/**
  - vllm_omni/deploy/**
depends_on:
  - input_output_modality_contracts.md
  - error_contracts.md
  - engine_orchestration.md
validation_paths:
  - tests/protocol/**
  - tests/entrypoints/test_omni_entrypoints.py
  - tests/entrypoints/test_async_omni.py
  - tests/entrypoints/test_async_omni_pause_sleep_routing.py
  - tests/entrypoints/test_duplex_omni.py
  - tests/entrypoints/duplex/**
  - tests/entrypoints/openai_api/test_duplex_api_server.py
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

When `--api-server-count` is greater than one, the serve composition root
starts the frontend process manager while a parent-owned `StageRuntime`
launches the shared local stage engines. Each frontend receives only its own
stage-client channel configuration and does not launch or retire backend
processes. CLI validation rejects unsupported distributed, diffusion,
fault-tolerant, elastic-EP, Ray, and runtime-LoRA combinations before startup.

It does not own configuration precedence, cross-stage routing, stage
placement, payload implementation, or semantic error classification.
`entrypoints/openai/errors.py` is an explicit primary-path exception owned by
`error_contracts.md`.

It also owns `vllm_omni/protocol/**`, the shared wire codec promoted by RFC
[#6592](https://github.com/vllm-project/vllm-omni/issues/6592) P0a. It follows the three tiers
`docs/serving/realtime_duplex_api.md` already defines. `protocol/realtime/**`
is Tier 1, identical to OpenAI: 30 server events, 10 client commands, their
typed fields and pure wire rendering, audio-format negotiation,
conversation-item rules and the error-envelope shape. `protocol/duplex/**` is
Tier 2 (OpenAI names carrying vLLM-Omni extensions, each a subclass of its
Tier 1 twin declaring only what it adds) and Tier 3 (ours alone, including our
error-code vocabulary), kept separate so a GA-only consumer is not handed a
vocabulary its clients never send. That package sits
outside `entrypoints/` on purpose --- the engine depends on it, so it belongs
to neither layer --- but deciding what a Realtime client may put on the wire is
this document's subject, and the same reasoning already places
`vllm_omni/errors.py`, `inputs/` and `outputs/` under contract documents rather
than under a consumer.

It is deliberately **not** owned by `../fullduplex.md`, even though duplex is
the codec's only consumer today. RFC #6592 asks for a named owner precisely so
a second Realtime consumer does not need duplex review to change the codec
(#6592 open question 8). Ownership here is review scope, not a dependency: the
codec may not import the engine, the entrypoints or the model code, which
`tests/protocol/realtime/test_protocol_import_boundary.py` enforces. When a
second consumer lands (#6592 P0b) the package is expected to graduate to its
own module document with owners drawn from both consumers.

## Candidate invariants

These identifiers are proposals while the document is `draft`.

### ENTRY-INV-001: Entrypoints adapt but do not orchestrate

**Rule:** Entrypoints MUST NOT implement cross-stage routing or stage lifecycle
policy.

### ENTRY-INV-002: The shared Realtime codec does not depend on its consumers

**Rule:** `vllm_omni/protocol/**` MUST NOT import `vllm_omni.engine`,
`vllm_omni.entrypoints`, `vllm_omni.model_executor`, `vllm_omni.worker` or
`vllm_omni.clients`; `protocol/duplex/**` MAY depend on `protocol/realtime/**`
but never the reverse, and a Tier 1 class MUST NOT carry a vLLM-Omni extension
field; and each protocol/codec behavior MUST have exactly one
implementation that every consumer uses. A wire type MUST NOT carry a
consumer's internal representation: the duplex mailbox rendering
(`DuplexCommand.payload()`, whose channel differs from the client event for
`session.update` and the `conversation.item.*` commands) stays engine-side.
A duplex consumer (`engine/duplex/**`, `entrypoints/duplex/**`, the duplex
clients) MUST import `vllm_omni.protocol.duplex` and MUST NOT import
`vllm_omni.protocol.realtime` directly, so the tier boundary has exactly one
extension point.

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
each affected offline or serving entrypoint. Sleep must wait for in-flight
`generate()` admission before EngineCore offload; `wake_up` does not resume
admission — callers must `resume_generation()`. Sleeping tags are tracked per
stage so `wake_up(stage_ids=[0])` does not skip a later `wake_up(stage_ids=[1])`.
Streaming input pumps take an admission slot immediately before each EngineCore
ADD or update, not while waiting for the next client chunk. Frontend abort
keeps `request_states` until `generate()` consumes the terminal output.
Multi-API serving must preserve one client rank and one channel set per
frontend, include every frontend in readiness/failure observation, and keep
stage lifecycle ownership in the parent composition root.

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
