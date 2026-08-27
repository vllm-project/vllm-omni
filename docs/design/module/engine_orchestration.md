---
title: Engine Orchestration
kind: module
status: draft
architecture_state: current-plus-in-flight
owners:
  - "@tzhouam"
  - "@fake0fan"
document_stewards:
  - "@hsliuustc0106"
  - "@Gaohan123"
  - "@david6666666"
required_reviewers:
  - "@yinpeiqi"
  - "@Sy0307"
  - "@yenuo26"
  - "@NickCao"
primary_code_paths:
  - vllm_omni/engine/async_omni_engine.py
  - vllm_omni/engine/async_engine_utils.py
  - vllm_omni/engine/orchestrator.py
  - vllm_omni/engine/cfg_companion_tracker.py
  - vllm_omni/engine/rpc_result_router.py
related_code_paths:
  - vllm_omni/engine/messages.py
  - vllm_omni/engine/orchestrator_monitor.py
  - vllm_omni/engine/stage_client.py
  - vllm_omni/engine/stage_pool.py
  - vllm_omni/outputs/output_processor.py
  - vllm_omni/distributed/omni_coordinator/**
depends_on:
  - input_output_modality_contracts.md
  - error_contracts.md
  - stage_runtime.md
  - omni_connector.md
validation_paths:
  - tests/engine/test_async_omni_engine_input.py
  - tests/engine/test_async_omni_engine_outputs.py
  - tests/engine/test_async_omni_engine_abort.py
  - tests/engine/test_async_omni_engine_stage_init.py
  - tests/engine/test_orchestrator.py
  - tests/engine/test_orchestrator_error_handling.py
  - tests/engine/test_orchestrator_stage_input_bridge.py
  - tests/engine/test_cfg_companion_lifecycle.py
  - tests/engine/test_rpc_result_router.py
  - tests/e2e/features/fullduplex/engine/**
upstream_refs:
  - vllm.v1.engine.EngineCoreRequest
  - vllm.v1.engine.EngineCoreOutputs
  - vllm.v1.engine.exceptions.EngineDeadError
  - vllm.outputs.RequestOutput
invariant_namespace: ORCH-INV
last_reviewed: 2026-08-07
last_verified_commit: 3d7fc3b9ba3cac88d579d4dc35b78b0b641675fc
---

# Engine orchestration

Engine orchestration coordinates configured AR and diffusion stages while
keeping public clients independent from stage processes and transports.

## Contract status

This document is a draft description of current behavior. It also identifies
the boundary affected by the in-flight stage client/process refactor in
[#5441](https://github.com/vllm-project/vllm-omni/pull/5441). Names and
responsibilities proposed only by that PR are not current contracts.

The orchestration loop also has an opt-in event-driven mode
(`VLLM_OMNI_EVENT_DRIVEN_ORCH=1`, default off) proposed in
[#5221](https://github.com/vllm-project/vllm-omni/pull/5221). It changes poll
cadence only: the routing, ordering, and terminal-state contracts below hold
identically on both loops.

## Ownership boundary

This document owns `AsyncOmniEngine`, `Orchestrator`, request-state creation,
cross-stage routing, output ordering, companion tracking, control/RPC
correlation, cancellation propagation, and terminal-state convergence.

It does not own stage placement, replica selection, stage-process startup,
payload schemas, public protocol rendering, connector transport, or semantic
error classification. Those responsibilities belong to the stage runtime,
I/O, entrypoint, connector, and error contracts.

## Candidate invariants

These identifiers are proposals while the document is `draft`.

### ORCH-INV-001: The orchestrator owns cross-stage routing

**Rule:** Entrypoints and stage clients MUST NOT independently forward a
request to a downstream logical stage.

### ORCH-INV-002: Stage clients do not own routing policy

**Rule:** Stage clients MUST implement communication and lifecycle operations
without selecting the next logical stage.

### ORCH-INV-100: Terminal state is monotonic

**Rule:** Once a request reaches a terminal state, orchestration MUST NOT
forward new work for that request.

## Invariant namespace

`ORCH-INV` reserves `001-099` for dependency direction, `100-199` for
request state and ordering, `200-299` for failure/cancellation/cleanup, and
`300-399` for extension points, shutdown ordering, and upstream alignment.
Numbers become append-only after normative promotion.

## Safe-change guide

Test routing, output ordering, queue correlation, cancellation, failure
propagation, shutdown ordering, and representative multi-stage execution.
Changes that cross into stage runtime or public error behavior require review
from that contract's owners.

## Promotion gate

- Reconcile current names and boundaries after #5441 merges, closes, or is
  replaced.
- Demonstrate one terminal outcome per request across success, abort,
  request-scoped failure, and fatal engine failure.
- Verify queue correlation and orchestrator-before-runtime shutdown ordering
  in the cited validation paths.
- Obtain approval from a technical owner and an independent validation
  reviewer.
