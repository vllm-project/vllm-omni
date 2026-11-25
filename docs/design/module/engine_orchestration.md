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
  - vllm_omni/engine/omni_engine_base.py
  - vllm_omni/engine/async_omni_engine.py
  - vllm_omni/engine/async_engine_utils.py
  - vllm_omni/engine/orchestrator.py
  - vllm_omni/engine/cfg_companion_tracker.py
  - vllm_omni/engine/rpc_result_router.py
related_code_paths:
  - vllm_omni/entrypoints/async_omni.py
  - vllm_omni/entrypoints/async_omni_base.py
  - vllm_omni/engine/messages.py
  - vllm_omni/engine/orchestrator_monitor.py
  - vllm_omni/engine/stage_client.py
  - vllm_omni/engine/stage_pool.py
  - vllm_omni/outputs/output_processor.py
  - vllm_omni/distributed/omni_coordinator/**
  # The duplex siblings (DuplexOmniEngine / DuplexOrchestrator) and the
  # engine-resident session framework are owned by ../fullduplex.md; listed
  # here because they fill the template seams of OmniEngineBase and
  # OrchestratorBase.
  - vllm_omni/engine/duplex_omni_engine.py
  - vllm_omni/engine/duplex_orchestrator.py
  - vllm_omni/engine/duplex/**
depends_on:
  - input_output_modality_contracts.md
  - error_contracts.md
  - stage_runtime.md
  - omni_connector.md
validation_paths:
  - tests/engine/test_async_omni_engine_input.py
  - tests/engine/test_async_omni_engine_outputs.py
  - tests/engine/test_async_omni_engine_abort.py
  - tests/entrypoints/test_async_omni.py
  - tests/e2e/offline_inference/test_qwen3_omni.py
  - tests/engine/test_async_omni_engine_stage_init.py
  - tests/engine/test_orchestrator.py
  - tests/engine/test_orchestrator_error_handling.py
  - tests/engine/test_orchestrator_stage_input_bridge.py
  - tests/engine/test_cfg_companion_lifecycle.py
  - tests/engine/test_rpc_result_router.py
  - tests/engine/duplex/**
  - tests/engine/test_duplex_orchestrator.py
  - tests/engine/test_duplex_omni_engine.py
  - tests/engine/test_duplex_import_boundary.py
upstream_refs:
  - vllm.v1.engine.EngineCoreRequest
  - vllm.v1.engine.EngineCoreOutputs
  - vllm.v1.engine.exceptions.EngineDeadError
  - vllm.outputs.RequestOutput
invariant_namespace: ORCH-INV
last_reviewed: 2026-11-22
last_verified_commit: 5b764aea027be18e7b1723c84e1a2b7aa1bb8029
---

# Engine orchestration

Engine orchestration coordinates configured AR and diffusion stages while
keeping public clients independent from stage processes and transports.

## Contract status

This document is a draft description of current behavior. Open proposals
include [pipeline-configured static transitions](https://github.com/vllm-project/vllm-omni/issues/8740) and
[dynamic, contract-aware routing](https://github.com/vllm-project/vllm-omni/issues/6302). The
[design-doc and review handoff RFC](https://github.com/vllm-project/vllm-omni/issues/10923) tracks how contributors
verify and maintain these contracts.

The orchestration loop is [event-driven by default](https://github.com/vllm-project/vllm-omni/pull/8675).
`VLLM_OMNI_EVENT_DRIVEN_ORCH=0` selects the legacy 1 ms poll loop. Both
loops share the routing and terminal handling described below.

## Ownership boundary

This document owns `AsyncOmniEngine`, `Orchestrator`, request-state creation,
cross-stage routing, output ordering, companion tracking, control/RPC
correlation, cancellation propagation, and terminal-state convergence.

It does not own stage placement, replica selection, stage-process startup,
payload schemas, public protocol rendering, connector transport, or semantic
error classification. Those responsibilities belong to the [stage runtime](stage_runtime.md),
I/O, entrypoint, connector, and error contracts.

## Two-stage request lifecycle

Follow a single prompt through `AsyncOmni.generate()` with `async_chunk=False`,
two local LLM stages, and stage 1 as the only final-output stage. The
[`test_run_two_stage_llm`](https://github.com/vllm-project/vllm-omni/blob/main/tests/engine/test_orchestrator.py) uses fake stage clients and
output processors to cover the engine portion of this path. The implementation
was checked at the revision in `last_verified_commit`.

1. **Admission.** [`AsyncOmni.generate`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/entrypoints/async_omni.py) derives the final stage
   from the requested output modalities and registers frontend request state
   under an internal ID. [`AsyncOmniEngine.add_request_async`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/engine/async_omni_engine.py)
   prepares the request and enqueues a `StageSubmissionMessage`.
   [`Orchestrator._handle_add_request`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/engine/orchestrator.py) checks admission,
   records separate orchestration state, and submits to the stage 0 pool.
   The [`test_generate_submits_randomized_id_to_engine`](https://github.com/vllm-project/vllm-omni/blob/main/tests/entrypoints/test_async_omni.py) checks the internal ID
   handoff, and the two-stage test checks stage 0 dispatch.
2. **Replica selection.** The orchestrator supplies the logical stage ID.
   [`StagePool.submit_initial`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/engine/stage_pool.py) selects the replica, records
   affinity, registers output-processor state, and submits through that
   replica's client. The [replica handoff](stage_runtime.md#request-to-replica-handoff)
   describes selection and links the replica routing tests.
3. **Downstream forwarding.** The stage 0 client supplies EngineCore output.
   `StagePool.process_llm_raw_outputs` feeds the stage output
   processor. When stage 0 finishes, [`Orchestrator._route_output`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/engine/orchestrator.py)
   decides to advance. `_forward_to_next_stage_unguarded` selects
   logical stage 1 and calls
   [`StageEngineCoreClientBase.process_engine_inputs`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/engine/stage_engine_core_client.py). `_submit_to_stage`
   hands the resulting request to the stage 1 pool with the same orchestration
   ID. The two-stage test checks that ID and the new prompt
   tokens.
4. **Final output and cleanup.** When stage 1 finishes, the orchestrator
   enqueues a finished `OutputMessage`. `_cleanup_request_ids`
   releases stage bindings and request transfer resources and removes
   orchestration state. The frontend output handler `_final_output_handler`
   routes the message to the request queue, and
   [`_process_orchestrator_results`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/entrypoints/async_omni_base.py) restores the external
   ID and yields the output. The generator's `finally` block cleans up
   frontend state. See the two-stage test
   and the [`test_omni_generate_request_id`](https://github.com/vllm-project/vllm-omni/blob/main/tests/entrypoints/test_async_omni.py).

### Caller cancellation

On task cancellation or generator close, [`AsyncOmni.generate`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/entrypoints/async_omni.py)
acts first. It shields the bounded `_abort_internal_requests` call from outer
ASGI cancellation, settles any pending submit for the internal ID, and sends
an [abort RPC through `abort_async`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/engine/omni_engine_base.py). [`Orchestrator._handle_abort`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/engine/orchestrator.py)
asks the stage pools to abort work on the request's bound replicas before it
releases the bindings and removes orchestration state. It acknowledges a
successful abort after that cleanup. If cancellation arrives while only
stage 0 has been submitted, only that replica receives an abort. The
generator re-raises cancellation or generator close and cleans up its
frontend state even if the abort attempt times out. See the
[`test_generate_waits_for_abort_inside_cancelled_asgi_scope`](https://github.com/vllm-project/vllm-omni/blob/main/tests/entrypoints/test_async_omni.py) and the [`test_run_abort`](https://github.com/vllm-project/vllm-omni/blob/main/tests/engine/test_orchestrator.py).

### Downstream request failure

If stage 1's EngineCore reports `FinishReason.ERROR` for this request, the
stage reports it first. [`Orchestrator._process_llm_stage_outputs`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/engine/orchestrator.py)
recognizes the error before normal terminal routing.
`_handle_stage_error` queues a request-scoped `ErrorMessage`
with the failed stage ID, then initiates abort and cleanup. The frontend
dispatches the error to the request queue and raises it in the generator,
whose exception path attempts cleanup abort before removing frontend state.
The [`test_raw_stage_error_reaches_caller_before_normal_terminal_routing`](https://github.com/vllm-project/vllm-omni/blob/main/tests/engine/test_orchestrator.py) checks the error message, cleanup
call, and bypass of normal terminal routing. A replica process exit follows
the [replica loss path](stage_runtime.md#replica-loss). If stage 1 loses
all replicas before the handoff, `_forward_to_next_stage_unguarded`
checks the pool and fails the request before submitting. The
[`test_forward_to_dead_downstream_stage_fails_request_not_server`](https://github.com/vllm-project/vllm-omni/blob/main/tests/engine/test_orchestrator_error_handling.py) covers this case.

## Stage-0 multimodal preprocessing caches

One frontend can route requests to multiple stage-0 replicas, each with its
own receiver cache. OpenAI chat requests pass through the frontend renderer
before replica selection, so a frontend cache hit must still supply tensors
to a replica that has never received that media or has evicted it.

During stage-0 input-processor construction, replicated deployments select
the renderer's upstream processor-only cache instead of its LRU or SHM sender
cache. Distributed frontends select this policy at startup even with zero or
one initial replica, allowing later joins without switching a live cache.
Preprocessing results remain reusable; frontend tensor storage and tensor
transfer per request increase. Local single-replica deployments and disabled
or already processor-only caches retain their existing behavior. Renderer
resource cleanup remains owned by upstream vLLM.

`tests/engine/test_stage0_multimodal_cache.py` exercises startup selection,
repeated media across independent receivers, eviction, and late joins.

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
AR abort of a final-stage LLM request must deliver a terminal output with
`finish_reason="abort"` and the generated prefix so collocated training can
resume; diffusion abort remains whole-sample retry. Aborting a parallel-sampling
child must drop `parent_requests` once no children remain. EngineCore control
RPCs (`sleep` / `wake_up` / `pause_scheduler` / `resume_scheduler`) must
propagate worker exceptions rather than returning `{"supported": False}`.
When a request has multiple final output stages, only the last abort message
is request-terminal. Output-processor abort state is committed only after the
physical EngineCore abort succeeds.
Changes that cross into stage runtime or public error behavior require review
from that contract's owners.

Multimodal cache-miss feedback invalidates only the reported hashes in the
owning sender cache before request error delivery. Downstream caches are
updated under the preprocessing lock; stage 0 feedback crosses to the frontend
and is applied while draining outputs, including cancelled requests. Ordinary
request failures retain retryable classification; duplex session termination
keeps its existing semantics.

## Promotion gate

- Demonstrate one terminal outcome per request across success, abort,
  request-scoped failure, and fatal engine failure.
- Verify queue correlation and orchestrator-before-runtime shutdown ordering
  in the cited validation paths.
- Obtain approval from a technical owner and an independent validation
  reviewer.
