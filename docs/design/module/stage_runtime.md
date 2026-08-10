---
title: Stage Runtime and Replica Lifecycle
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
  - "@Sy0307"
  - "@princepride"
  - "@chickeyton"
  - "@yenuo26"
  - "@NickCao"
primary_code_paths:
  - vllm_omni/engine/stage_runtime.py
  - vllm_omni/engine/stage_pool.py
  - vllm_omni/engine/stage_client.py
  - vllm_omni/engine/membership_controller.py
  - vllm_omni/engine/stage_engine_startup.py
  - vllm_omni/engine/stage_engine_core_client.py
  - vllm_omni/engine/stage_engine_core_proc.py
  - vllm_omni/engine/stage_engine_core_proc_manager.py
  - vllm_omni/engine/stage_init_utils.py
related_code_paths:
  - vllm_omni/distributed/omni_coordinator/**
  - vllm_omni/distributed/ray_utils/**
  - vllm_omni/diffusion/inline_stage_diffusion_client.py
  - vllm_omni/diffusion/stage_diffusion_client.py
  - vllm_omni/diffusion/stage_diffusion_proc.py
  - vllm_omni/config/**
depends_on:
  - input_output_modality_contracts.md
  - error_contracts.md
  - omni_connector.md
validation_paths:
  - tests/engine/test_async_omni_engine_stage_init.py
  - tests/engine/test_membership_controller.py
  - tests/engine/test_single_stage_mode.py
  - tests/engine/test_stage_engine_core_client.py
  - tests/engine/test_stage_engine_core_proc.py
  - tests/diffusion/test_inline_stage_diffusion_client.py
  - tests/diffusion/test_stage_diffusion_proc.py
  - tests/distributed/omni_coordinator/**
upstream_refs:
  - vllm.v1.engine.core_client.AsyncMPClient
  - vllm.v1.engine.core_client.DPLBAsyncMPClient
  - vllm.v1.engine.core.EngineCoreProc
  - vllm.v1.engine.utils.CoreEngineProcManager
  - vllm.v1.engine.coordinator.DPCoordinator
invariant_namespace: STAGE-INV
last_reviewed: 2026-08-07
last_verified_commit: 3d7fc3b9ba3cac88d579d4dc35b78b0b641675fc
---

# Stage runtime and replica lifecycle

The stage runtime turns a logical stage definition into ready local or
distributed replicas and provides the lifecycle boundary used by the
orchestrator.

## Contract status

This document describes the current `StageRuntime`, `StagePool`, stage-client,
and stage-process layout. The unified LLM/diffusion direction in
[#5441](https://github.com/vllm-project/vllm-omni/pull/5441) remains in flight
and is not presented as current behavior.

## Ownership boundary

This document owns local and distributed stage placement, startup and
readiness, one pool per logical stage, replica identity and membership,
selection and affinity, stage-client/process lifecycle, liveness, draining,
and shutdown.

It does not own cross-stage request policy, model scheduler policy, connector
transfer semantics, configuration precedence, or public error rendering.

## Candidate invariants

These identifiers are proposals while the document is `draft`.

### STAGE-INV-001: The runtime owns replica lifecycle

**Rule:** Orchestration MUST acquire stage capacity through the stage runtime
and MUST NOT construct or retire backend replicas directly.

### STAGE-INV-100: Affinity uses stable replica identity

**Rule:** Follow-up operations for a request with replica affinity MUST resolve
the same logical replica until the request terminates or the replica is
declared lost.

### STAGE-INV-200: Shutdown is idempotent

**Rule:** Repeated shutdown or cleanup signals MUST NOT leak stage clients,
process managers, or membership registrations.

## Invariant namespace

`STAGE-INV` reserves `001-099` for placement and ownership, `100-199` for
replica identity/readiness/affinity, `200-299` for liveness loss, draining and
cleanup, and `300-399` for local/distributed and upstream compatibility.
Numbers become append-only after normative promotion.

## Safe-change guide

Exercise local and distributed startup, readiness, replica selection,
affinity, membership changes, abort routing, process failure, draining, and
repeated shutdown. Cross-stage policy changes belong in
`engine_orchestration.md`.

## Promotion gate

- Reconcile terminology and ownership after #5441 reaches a final state.
- Add or identify a dedicated StageRuntime/StagePool lifecycle suite; current
  evidence is distributed across the cited tests.
- Demonstrate affinity preservation for update, interaction, and abort paths.
- Demonstrate membership loss and repeated shutdown without leaked clients or
  processes.
- Obtain approval from a technical owner and an independent validation
  reviewer.
