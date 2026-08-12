---
title: Error Classification, Propagation, and Rendering
kind: module
status: draft
architecture_state: current-plus-rfc
owners:
  - "@alex-jw-brooks"
  - "@NickCao"
document_stewards:
  - "@hsliuustc0106"
  - "@Gaohan123"
  - "@david6666666"
required_reviewers:
  - "@yinpeiqi"
  - "@tzhouam"
  - "@yenuo26"
primary_code_paths:
  - vllm_omni/errors.py
  - vllm_omni/entrypoints/openai/errors.py
related_code_paths:
  - vllm_omni/engine/messages.py
  - vllm_omni/engine/orchestrator.py
  - vllm_omni/entrypoints/omni_base.py
  - vllm_omni/entrypoints/async_omni.py
  - vllm_omni/entrypoints/openai/api_server.py
  - vllm_omni/entrypoints/openai/serving_chat.py
  - vllm_omni/entrypoints/openai/serving_speech.py
  - vllm_omni/entrypoints/openai/serving_audio_generate.py
  - vllm_omni/entrypoints/openai/protocol/images.py
  - vllm_omni/entrypoints/openai/protocol/videos.py
depends_on: []
validation_paths:
  - tests/engine/test_orchestrator_error_handling.py
  - tests/engine/test_async_omni_engine_outputs.py
  - tests/entrypoints/test_omni_entrypoints.py
  - tests/entrypoints/test_stream_finish_reason.py
  - tests/entrypoints/openai_api/test_image_server.py
  - tests/entrypoints/openai_api/test_video_server.py
  - tests/entrypoints/openai_api/test_serving_audio_generate.py
  - tests/entrypoints/openai_api/test_serving_speech.py
  - tests/metrics/test_prometheus.py
upstream_refs:
  - vllm.v1.engine.exceptions.EngineDeadError
  - vllm.v1.engine.exceptions.EngineGenerateError
  - vllm.entrypoints.serve.utils.error_response
  - vLLM RFC #48227
  - vLLM PR #49665
invariant_namespace: ERR-INV
last_reviewed: 2026-08-07
last_verified_commit: 3d7fc3b9ba3cac88d579d4dc35b78b0b641675fc
---

# Error classification, propagation, and rendering

This contract separates the meaning of a failure from the transport used to
carry it and the public protocol used to render it.

## Contract status

Current behavior is distributed across engine, message, and entrypoint code.
The semantic hierarchy proposed by
[#5570](https://github.com/vllm-project/vllm-omni/issues/5570) remains an open
RFC. Proposed types and wire records from that RFC are not current contracts.

## Ownership boundary

This document owns semantic classification, fatal versus request-scoped
failure, cross-process error context, compatibility policy, sanitization, and
offline, HTTP, SSE, WebSocket, and asynchronous-job rendering rules.

`engine/messages.py` remains primarily owned by the I/O contract; this page
owns the semantics of its error fields. Route and serving files remain
primarily owned by `entrypoints.md`; this page owns their rendering
constraints.

## Candidate invariants

The following topics require owner review before receiving stable IDs:

- A request-scoped failure terminates only the affected request unless the
  underlying engine is no longer usable.
- Every public rendering path preserves a stable classification while
  removing sensitive internal context.
- Cancellation remains distinguishable from internal failure across process
  and protocol boundaries.

## Invariant namespace

`ERR-INV` reserves `001-099` for classification ownership, `100-199` for wire
context and compatibility, `200-299` for fatality, termination, sanitization,
and rendering, and `300-399` for upstream alignment and transport extension
points. Numbers become append-only after normative promotion.

## Safe-change guide

Exercise exception classification, stage wire round trips, fatal and
request-scoped propagation, cancellation, sanitization, offline exceptions,
HTTP status/body consistency, streaming terminal events, asynchronous-job
persistence, and metrics classification.

## Promotion gate

- Resolve #5570's hierarchy, wire record, sanitization, and mixed-version
  questions.
- Add table-driven exception/status/type tests and stage wire round trips.
- Verify all public transports terminate once and do not leak sensitive
  payload data.
- Obtain approval from both technical owners and an independent validation
  reviewer.
