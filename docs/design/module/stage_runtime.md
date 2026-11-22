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
  - tests/engine/test_orchestrator.py
  - tests/engine/test_orchestrator_error_handling.py
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
last_reviewed: 2026-11-22
last_verified_commit: 5b764aea027be18e7b1723c84e1a2b7aa1bb8029
---

# Stage runtime and replica lifecycle

The stage runtime turns a logical stage definition into ready local or
distributed replicas and provides the lifecycle boundary used by the
orchestrator.

## Contract status

This document describes the current `StageRuntime`, `StagePool`, stage-client,
and stage-process layout. [Sharing stages between pipelines][shared-stages]
remains an open design proposal. The earlier [LLM and DiT stage client and
process refactor][stage-client-refactor] closed without merging.

## Ownership boundary

This document owns local and distributed stage placement, startup and
readiness, one pool per logical stage, replica identity and membership,
selection and affinity, stage-client/process lifecycle, liveness, draining,
and shutdown.

Local EngineCore initialization and multi-API serving use
`StageRuntime.launch_stage_engines()` as their common launch
entry. It resolves plans when not supplied by the runtime's device-group
scheduler, coordinates device locks and environment overlays, starts engines,
and rolls back resources on startup or attachment failure. Single-API clients
attach locally after readiness and take ownership of the engine resources;
there is no additional API subprocess. Diffusion and remote attachment retain
their backend-specific initialization.

For local multi-API serving, the parent runtime owns and launches each stage
engine once. It supplies a distinct input/output channel for every API
frontend and stage replica; frontend workers attach clients to those engines
without acquiring stage-process ownership. Readiness observation and shutdown
remain parent-owned. This topology currently excludes diffusion and
remote/headless stages, intra-stage data parallelism, and Ray backends.

It does not own cross-stage request policy, model scheduler policy, connector
transfer semantics, configuration precedence, or public error rendering.

## Request-to-replica handoff

In the [two-stage request trace](engine_orchestration.md#two-stage-request-lifecycle),
[`StageRuntime._assemble_stage_pools`][assemble-pools] provides one pool per
logical stage with its replica clients and, for LLM stages, its output
processor. The orchestrator chooses when to submit to each pool.
[`StagePool.submit_initial`][pool-submit] binds an unpinned local request
through [`select_replica_id`][select-replica], which round-robins across
available replicas for new requests. The pool registers the LLM output
processor before [the selected client sends the request][client-add] to
EngineCore. In distributed mode, [`StagePool.pick`][pick-replica] reuses
serviceable affinity or asks the load balancer to select from attached,
serviceable replicas.

The binding belongs to the pool. [`StagePool.abort_requests`][pool-abort]
groups aborts by existing affinity and sends them to the corresponding
clients. The orchestrator tells the pool to
[`release_bindings`][release-bindings] when the request is cleaned up. This
frees request routing state and leaves the replica available for later
requests. See the [round-robin and onward routing test][test-round-robin],
the [affinity test][test-affinity], and the
[multi-replica abort routing test][test-abort-routing].

### Replica loss

If polling reports `EngineDeadError` for a physical replica,
[`Orchestrator._handle_dead_replica`][dead-replica] asks the pool to
[`evict_replica`][evict-replica] before failing stranded requests. For the
ordinary non-shared-weight path, the pool retires the dead replica from
routing. If another replica survives, orchestration fails requests bound to
the dead one. With no surviving replica, it fails requests already routed
through that stage. The frontend observes a fatal error on each affected
request. See the [replica eviction test][test-eviction] and the
[last-replica failure test][test-last-replica].

For a coordinator-driven removal, [`MembershipController.handle_unregister`][unregister]
acts first. It invalidates the address, detaches and shuts down the client,
then calls the removal and request cleanup callbacks before queuing errors
for affected requests. The [unregister ordering test][test-unregister]
checks that orchestration observes the removal before request cleanup.

Source and test links in this section point to the revision in
`last_verified_commit`.

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

AR stage deployments may set `engine_extras.shutdown_timeout` to a nonnegative
grace period in seconds. The stage client uses it when shutdown has no explicit
timeout; an explicit caller timeout takes precedence, including zero. With no
stage override, a positive upstream shutdown timeout is honored, otherwise the
upstream process manager retains its default grace period. The timeout bounds
process cleanup; it does not change request draining or admission policy.

Once a local stage process enters final cleanup, repeated `SIGTERM` and
`SIGINT` signals are ignored until its resource owners finish closing. Default
signal handling is restored afterwards. The process manager can still enforce
its shutdown deadline with `SIGKILL`.

## Invariant namespace

`STAGE-INV` reserves `001-099` for placement and ownership, `100-199` for
replica identity/readiness/affinity, `200-299` for liveness loss, draining and
cleanup, and `300-399` for local/distributed and upstream compatibility.
Numbers become append-only after normative promotion.

## Safe-change guide

Exercise local and distributed startup, readiness, replica selection,
affinity, membership changes, abort routing, process failure, draining, and
repeated shutdown. Cross-stage policy changes belong in
[engine orchestration](engine_orchestration.md).

For multi-API changes, additionally exercise per-frontend channel assignment,
shared-engine startup handshakes, frontend failure observation, and parent
cleanup without duplicate stage shutdown.

## Promotion gate

- Add or identify a dedicated StageRuntime/StagePool lifecycle suite; current
  evidence is distributed across the cited tests.
- Demonstrate affinity preservation for update, interaction, and abort paths.
- Demonstrate membership loss and repeated shutdown without leaked clients or
  processes.
- Obtain approval from a technical owner and an independent validation
  reviewer.

## Platform preparation before stage construction

`StageEngineCoreProc` calls `current_omni_platform.prepare_stage_engine()`
before the upstream engine constructor creates its scheduler and cache
managers. The default hook is a no-op. The NPU implementation installs its
Mamba allocator signature compatibility here, so Ascend imports and patches
remain in the NPU platform and are not applied to other platforms.

[assemble-pools]: https://github.com/vllm-project/vllm-omni/blob/5b764aea027be18e7b1723c84e1a2b7aa1bb8029/vllm_omni/engine/stage_runtime.py#L1478
[client-add]: https://github.com/vllm-project/vllm-omni/blob/5b764aea027be18e7b1723c84e1a2b7aa1bb8029/vllm_omni/engine/stage_engine_core_client.py#L277
[dead-replica]: https://github.com/vllm-project/vllm-omni/blob/5b764aea027be18e7b1723c84e1a2b7aa1bb8029/vllm_omni/engine/orchestrator.py#L1859
[evict-replica]: https://github.com/vllm-project/vllm-omni/blob/5b764aea027be18e7b1723c84e1a2b7aa1bb8029/vllm_omni/engine/stage_pool.py#L2243
[pick-replica]: https://github.com/vllm-project/vllm-omni/blob/5b764aea027be18e7b1723c84e1a2b7aa1bb8029/vllm_omni/engine/stage_pool.py#L533
[pool-abort]: https://github.com/vllm-project/vllm-omni/blob/5b764aea027be18e7b1723c84e1a2b7aa1bb8029/vllm_omni/engine/stage_pool.py#L1997
[pool-submit]: https://github.com/vllm-project/vllm-omni/blob/5b764aea027be18e7b1723c84e1a2b7aa1bb8029/vllm_omni/engine/stage_pool.py#L1546
[release-bindings]: https://github.com/vllm-project/vllm-omni/blob/5b764aea027be18e7b1723c84e1a2b7aa1bb8029/vllm_omni/engine/stage_pool.py#L917
[select-replica]: https://github.com/vllm-project/vllm-omni/blob/5b764aea027be18e7b1723c84e1a2b7aa1bb8029/vllm_omni/engine/stage_pool.py#L1070
[shared-stages]: https://github.com/vllm-project/vllm-omni/issues/4108
[stage-client-refactor]: https://github.com/vllm-project/vllm-omni/pull/5441
[test-abort-routing]: https://github.com/vllm-project/vllm-omni/blob/5b764aea027be18e7b1723c84e1a2b7aa1bb8029/tests/engine/test_orchestrator.py#L2156
[test-affinity]: https://github.com/vllm-project/vllm-omni/blob/5b764aea027be18e7b1723c84e1a2b7aa1bb8029/tests/engine/test_orchestrator.py#L2217
[test-eviction]: https://github.com/vllm-project/vllm-omni/blob/5b764aea027be18e7b1723c84e1a2b7aa1bb8029/tests/engine/test_orchestrator_error_handling.py#L264
[test-last-replica]: https://github.com/vllm-project/vllm-omni/blob/5b764aea027be18e7b1723c84e1a2b7aa1bb8029/tests/engine/test_orchestrator_error_handling.py#L638
[test-round-robin]: https://github.com/vllm-project/vllm-omni/blob/5b764aea027be18e7b1723c84e1a2b7aa1bb8029/tests/engine/test_orchestrator.py#L2086
[test-unregister]: https://github.com/vllm-project/vllm-omni/blob/5b764aea027be18e7b1723c84e1a2b7aa1bb8029/tests/engine/test_membership_controller.py#L245
[unregister]: https://github.com/vllm-project/vllm-omni/blob/5b764aea027be18e7b1723c84e1a2b7aa1bb8029/vllm_omni/engine/membership_controller.py#L112
