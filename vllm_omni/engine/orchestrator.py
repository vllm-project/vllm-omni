"""
Orchestrator for vLLM-Omni multi-stage runtime.

Runs inside a background thread with its own asyncio event loop.
Owns logical request progression across stage pools and handles
stage-to-stage transfer logic.

In distributed mode (``coordinator_pub_address`` provided), it also
owns the single :class:`OmniCoordClientForHub`, runs a
:meth:`_watch_replica_list` task that converts replica disappearances
into ``unregister_remote_replica`` control messages, and handles the
``register_remote_replica`` / ``unregister_remote_replica`` flow that
attaches / detaches head-side stage clients for headless replicas.
"""

from __future__ import annotations

import asyncio
import time as _time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import janus
import torch
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.distributed.omni_coordinator import (
    LoadBalancer,
    OmniCoordClientForHub,
    RandomBalancer,
    ReplicaStatus,
)
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.cfg_companion_tracker import CfgCompanionTracker
from vllm_omni.engine.messages import (
    AbortRequestMessage,
    AddCompanionRequestMessage,
    CollectiveRPCRequestMessage,
    CollectiveRPCResultMessage,
    EngineQueueMessage,
    ErrorMessage,
    OutputMessage,
    RegisterRemoteReplicaMessage,
    ShutdownRequestMessage,
    StageMetricsMessage,
    StageSubmissionMessage,
    UnregisterRemoteReplicaMessage,
)
from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.engine.stage_pool import StagePool
from vllm_omni.outputs import OmniRequestOutput

# Factory signature for building a head-side stage client for a
# *dynamically attached* (auto-assigned) remote replica.
#
# Receives ``(stage_id, replica_id)`` and returns an awaitable yielding the
# constructed client (any type — it must satisfy the shape expected by the
# matching :class:`StagePool`, i.e. expose ``client_addresses["input_address"]``
# or ``request_address``, plus the usual ``add_request_async`` /
# ``get_output_async`` / ``shutdown`` surface).
RemoteReplicaFactory = Callable[[int, int], Awaitable[Any]]

logger = init_logger(__name__)


def build_engine_core_request_from_tokens(
    request_id: str,
    prompt: dict[str, Any],
    params: SamplingParams | PoolingParams,
    arrival_time: float | None = None,
    model_config: ModelConfig | None = None,
    resumable: bool = False,
    mm_features: list | None = None,
) -> OmniEngineCoreRequest:
    """Build an OmniEngineCoreRequest directly from an OmniTokensPrompt."""
    if arrival_time is None:
        arrival_time = _time.time()

    prompt_token_ids = prompt["prompt_token_ids"]

    sampling_params = None
    pooling_params = None
    if isinstance(params, SamplingParams):
        sampling_params = params.clone()
        if sampling_params.max_tokens is None and model_config is not None:
            sampling_params.max_tokens = model_config.max_model_len - len(prompt_token_ids)
    else:
        pooling_params = params.clone()

    prompt_embeds: torch.Tensor | None = prompt.get("prompt_embeds")
    additional_info_payload = serialize_additional_information(
        prompt.get("additional_information"),
        log_prefix=f"build_engine_core_request_from_tokens req={request_id}",
    )

    return OmniEngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=mm_features,
        sampling_params=sampling_params,
        pooling_params=pooling_params,
        arrival_time=arrival_time,
        lora_request=getattr(params, "lora_request", None),
        cache_salt=prompt.get("cache_salt"),
        data_parallel_rank=None,
        prompt_embeds=prompt_embeds,
        resumable=resumable,
        additional_information=additional_info_payload,
    )


@dataclass
class OrchestratorRequestState:
    """Per-request bookkeeping inside the Orchestrator."""

    request_id: str
    prompt: Any = None
    sampling_params_list: list[Any] = field(default_factory=list)
    final_stage_id: int = -1

    # Metrics: timestamp when request was submitted to each stage.
    stage_submit_ts: dict[int, float] = field(default_factory=dict)
    mm_processor_kwargs: dict | None = None
    mm_features: list | None = None
    pd_prefill_multimodal_output: dict[str, Any] | None = None

    streaming: StreamingInputState = field(default_factory=lambda: StreamingInputState())

    # Per-request pipeline timing accumulator (milliseconds)
    pipeline_timings: dict[str, float] = field(default_factory=dict)


@dataclass
class StreamingInputState:
    # Flag of streaming input request
    enabled: bool = False
    # Flag of segment of streaming input finished
    segment_finished: bool = False
    # Streaming update prompt length
    new_prompt_len_snapshot: int | None = None
    # Model/bridge-specific runtime states (e.g., thinker->talker)
    bridge_states: dict[str, Any] = field(default_factory=dict)


class Orchestrator:
    """Runs inside a background thread's asyncio event loop."""

    # Cadence at which the replica-list watcher polls for disappearances.
    _WATCH_REPLICA_INTERVAL_S: float = 0.5
    _WATCH_REPLICA_IDLE_INTERVAL_S: float = 1.0

    def __init__(
        self,
        request_async_queue: janus.AsyncQueue[EngineQueueMessage],
        output_async_queue: janus.AsyncQueue[dict[str, Any]],
        rpc_async_queue: janus.AsyncQueue[dict[str, Any]],
        stage_pools: list[StagePool],
        *,
        async_chunk: bool = False,
        pd_config: dict[str, Any] | None = None,
        coordinator_pub_address: str | None = None,
        load_balancer_factory: Callable[[], LoadBalancer] | None = None,
        remote_replica_factory: RemoteReplicaFactory | None = None,
    ) -> None:
        self.request_async_queue = request_async_queue
        self.output_async_queue = output_async_queue
        self.rpc_async_queue = rpc_async_queue

        self.async_chunk = bool(async_chunk)
        self.num_stages = len(stage_pools)
        self.stage_pools: list[StagePool] = stage_pools

        # PD disaggregation state
        self._pd_pair: tuple[int, int] | None = None
        self._pd_bootstrap_addr: str | None = None
        self._pd_prefill_engine_id: str | None = None
        self._pd_kv_params: dict[str, Any] = {}
        if pd_config is not None:
            self._pd_pair = pd_config.get("pd_pair")
            self._pd_bootstrap_addr = pd_config.get("bootstrap_addr")
            self._pd_prefill_engine_id = pd_config.get("prefill_engine_id")
        self.request_states: dict[str, OrchestratorRequestState] = {}
        self._cfg_tracker = CfgCompanionTracker()

        self._shutdown_event = asyncio.Event()
        self._stages_shutdown = False
        self._fatal_error: str | None = None
        self._fatal_error_stage_id: int | None = None

        # Background tasks for fire-and-forget message handlers (currently
        # only ``register_remote_replica`` and ``unregister_remote_replica``).
        # Held as a set so each task's reference survives the loop and the
        # task can self-deregister on completion.
        self._membership_tasks: set[asyncio.Task[None]] = set()

        # Distributed-mode wiring. The hub is constructed on the
        # orchestrator's asyncio loop because it spawns a SUB background
        # thread; building it from another thread would race the
        # ``_init_done`` event.
        self._hub: OmniCoordClientForHub | None = (
            OmniCoordClientForHub(coordinator_pub_address) if coordinator_pub_address is not None else None
        )
        self._remote_replica_factory = remote_replica_factory
        # Inject hub + per-pool LB into each StagePool so they can run
        # distributed dispatch via ``StagePool.pick``.
        if self._hub is not None:
            factory = load_balancer_factory or RandomBalancer
            for pool in self.stage_pools:
                pool.attach_hub(self._hub)
                pool.attach_load_balancer(factory())

    async def run(self) -> None:
        """Main entry point for the Orchestrator event loop."""
        logger.info("[Orchestrator] Starting event loop")

        request_task = asyncio.create_task(self._request_handler(), name="orchestrator-request-handler")
        output_task = asyncio.create_task(
            self._orchestration_output_handler(),
            name="orchestrator-stage-output-handler",
        )
        # The replica watcher only runs in distributed mode. It's still
        # created in both cases so ``run()`` has a uniform task graph;
        # ``_watch_replica_list`` is a no-op poll when ``self._hub`` is None.
        watch_task = asyncio.create_task(
            self._watch_replica_list(),
            name="orchestrator-replica-watcher",
        )

        try:
            await asyncio.gather(request_task, output_task, watch_task)
        except asyncio.CancelledError:
            raise
        except EngineDeadError as e:
            # EngineDeadError from _orchestration_loop means the diffusion
            # engine died.  All pending requests were already notified and
            # _shutdown_event was already set by the loop's handler.
            # During teardown this is expected; the finally block handles
            # proper cleanup.  Do not re-raise.
            logger.info("[Orchestrator] Engine dead during shutdown: %s", e)
        except Exception:
            logger.exception("[Orchestrator] Fatal error in orchestrator tasks")
            raise
        finally:
            self._shutdown_event.set()
            for task in (request_task, output_task, watch_task):
                if not task.done():
                    task.cancel()
            try:
                await asyncio.gather(request_task, output_task, watch_task, return_exceptions=True)
            except Exception:
                pass

            # If a fatal error caused the shutdown, drain any pending
            # add_request messages that were never processed and broadcast
            # fatal error responses so callers are not left hanging.
            if self._fatal_error is not None:
                await self._drain_pending_requests_on_fatal()

            # Wait briefly for any in-flight membership handlers (register /
            # unregister remote replica) to finish so they don't leave the
            # head-side pool in a half-attached state. Cancel anything that
            # hasn't completed in time; the generic pending-task sweep below
            # will collect the cancellations.
            if self._membership_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self._membership_tasks, return_exceptions=True),
                        timeout=10.0,
                    )
                except (asyncio.TimeoutError, Exception):
                    for t in self._membership_tasks:
                        if not t.done():
                            t.cancel()

            self._shutdown_stages()

            # Close the hub last so any in-flight dispatch still has access.
            if self._hub is not None:
                try:
                    self._hub.close()
                except RuntimeError:
                    pass
                self._hub = None

            loop = asyncio.get_running_loop()
            pending = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task() and not t.done()]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    # ---- Background task helpers ----

    def _spawn_membership_task(self, coro: Awaitable[None], *, label: str) -> None:
        """Run a fire-and-forget membership-change coroutine.

        Holds a strong reference until completion (asyncio would otherwise
        garbage-collect a bare task), and logs any uncaught exception.
        """
        task = asyncio.create_task(coro, name=f"orchestrator-{label}")
        self._membership_tasks.add(task)

        def _on_done(t: asyncio.Task[None]) -> None:
            self._membership_tasks.discard(t)
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                logger.error("[Orchestrator] %s task crashed", label, exc_info=exc)

        task.add_done_callback(_on_done)

    # ---- Request handling ----

    async def _request_handler(self) -> None:
        """Read messages from the main thread via request_async_queue."""
        while True:
            msg = await self.request_async_queue.get()
            msg_type = msg.type

            if msg_type == "add_request":
                await self._handle_add_request(msg)
            elif msg_type == "streaming_update":
                await self._handle_streaming_update(msg)
            elif msg_type == "add_companion_request":
                await self._handle_add_companion(msg)
            elif msg_type == "abort":
                await self._handle_abort(msg)
            elif msg_type == "collective_rpc":
                await self._handle_collective_rpc(msg)
            elif isinstance(msg, RegisterRemoteReplicaMessage):
                # Dynamic-attach involves a ~5s blocking handshake (run in a
                # thread by ``_build_remote_replica``); ``await`` here would
                # block the queue and stall the next ``add_request`` until
                # the attach finishes. Dispatch as a background task so the
                # main message loop keeps draining.
                self._spawn_membership_task(self._handle_register_remote_replica(msg), label="register_remote_replica")
            elif isinstance(msg, UnregisterRemoteReplicaMessage):
                # Symmetric with register: keep the main queue flowing.
                self._spawn_membership_task(
                    self._handle_unregister_remote_replica(msg),
                    label="unregister_remote_replica",
                )
            elif isinstance(msg, ShutdownRequestMessage):
                logger.info("[Orchestrator] Received shutdown signal")
                self._shutdown_event.set()
                # Pre-mark stage clients as shutting down to prevent
                # proc_monitor daemon threads from flagging normal
                # process exit as EngineDeadError during teardown.
                for pool in self.stage_pools:
                    for client in pool.clients:
                        if hasattr(client, "_shutting_down"):
                            client._shutting_down = True
                self._shutdown_stages()
                break
            else:
                logger.warning("[Orchestrator] Unknown message type: %s", msg_type)

    async def _handle_add_request(self, msg: StageSubmissionMessage) -> None:
        """Handle an add_request message from the main thread."""
        stage_id = 0
        request_id = msg.request_id
        prompt = msg.prompt
        original_prompt = msg.original_prompt
        sampling_params_list = msg.sampling_params_list
        if not sampling_params_list:
            raise ValueError(f"Missing sampling params for stage 0. Got {len(sampling_params_list)} stage params.")
        final_stage_id = msg.final_stage_id

        logger.debug(
            "[Orchestrator] _handle_add_request: stage=%s req=%s "
            "prompt_type=%s original_prompt_type=%s final_stage=%s "
            "num_sampling_params=%d",
            stage_id,
            request_id,
            type(prompt).__name__,
            type(original_prompt).__name__,
            final_stage_id,
            len(sampling_params_list),
        )

        req_state = OrchestratorRequestState(
            request_id=request_id,
            prompt=original_prompt,
            sampling_params_list=sampling_params_list,
            final_stage_id=final_stage_id,
            mm_features=getattr(prompt, "mm_features", None),
        )
        self.request_states[request_id] = req_state
        req_state.streaming.enabled = bool(getattr(prompt, "resumable", False))
        req_state.stage_submit_ts[stage_id] = _time.time()
        enqueue_ts = msg.enqueue_ts
        if enqueue_ts > 0:
            req_state.pipeline_timings["queue_wait_ms"] = (_time.perf_counter() - enqueue_ts) * 1000.0
        preprocess_ms = msg.preprocess_ms
        if preprocess_ms > 0:
            req_state.pipeline_timings["preprocess_ms"] = preprocess_ms
        await self.stage_pools[stage_id].submit_initial(
            request_id,
            req_state,
            prompt,
            prompt_text=msg.output_prompt_text,
        )

        if self.async_chunk and stage_id == 0 and final_stage_id > 0:
            await self._prewarm_async_chunk_stages(request_id, prompt, req_state)

    async def _handle_streaming_update(self, msg: StageSubmissionMessage) -> None:
        """Handle a streaming_update message for an existing request."""
        stage_id = 0
        request_id = msg.request_id
        request = msg.prompt

        req_state = self.request_states.get(request_id)
        if req_state is None:
            logger.warning(
                "[Orchestrator] streaming_update for unknown req=%s, falling back to add_request",
                request_id,
            )
            fallback_msg = StageSubmissionMessage(
                type="add_request",
                request_id=msg.request_id,
                prompt=msg.prompt,
                original_prompt=msg.original_prompt,
                output_prompt_text=msg.output_prompt_text,
                sampling_params_list=msg.sampling_params_list,
                final_stage_id=msg.final_stage_id,
                preprocess_ms=msg.preprocess_ms,
                enqueue_ts=msg.enqueue_ts,
            )
            await self._handle_add_request(fallback_msg)
            return

        if msg.sampling_params_list:
            req_state.sampling_params_list = msg.sampling_params_list

        req_state.streaming.enabled = True
        req_state.stage_submit_ts[stage_id] = _time.time()
        await self.stage_pools[stage_id].submit_update(
            request_id,
            req_state,
            request,
            prompt_text=msg.output_prompt_text,
        )

    async def _handle_add_companion(self, msg: AddCompanionRequestMessage) -> None:
        """Handle an add_companion_request message: submit companion to stage 0."""
        companion_id = msg.companion_id
        parent_id = msg.parent_id
        role = msg.role
        companion_prompt = msg.prompt
        sampling_params_list = msg.sampling_params_list

        parent_state = self.request_states.get(parent_id)
        if parent_state is None:
            logger.info(
                "[Orchestrator] Dropping CFG companion %s (role=%s): parent %s is no longer active",
                companion_id,
                role,
                parent_id,
            )
            return

        self._cfg_tracker.register_companion(parent_id, role, companion_id)

        companion_state = OrchestratorRequestState(
            request_id=companion_id,
            prompt=companion_prompt,
            sampling_params_list=sampling_params_list,
            final_stage_id=0,
        )
        self.request_states[companion_id] = companion_state
        companion_state.stage_submit_ts[0] = _time.time()
        companion_replica_id = await self.stage_pools[0].submit_initial(
            companion_id,
            companion_state,
            companion_prompt,
            prompt_text=msg.companion_prompt_text,
            affinity_request_id=parent_id,
        )

        logger.info(
            "[Orchestrator] CFG companion submitted: %s (role=%s, parent=%s, stage-0 replica-%s)",
            companion_id,
            role,
            parent_id,
            companion_replica_id,
        )

    async def _handle_abort(self, msg: AbortRequestMessage) -> None:
        """Handle an abort message from the main thread."""
        request_ids = msg.request_ids
        await self._cleanup_request_ids(
            self._cfg_tracker.abort_parents(request_ids),
            abort=True,
        )
        logger.info("[Orchestrator] Aborted request(s) %s", request_ids)

    async def _abort_request_ids(self, request_ids: list[str]) -> None:
        """Forward abort requests to all stage pools."""
        if not request_ids:
            return
        for pool in self.stage_pools:
            await pool.abort_requests(request_ids)

    def _release_request_bindings(self, request_ids: list[str]) -> None:
        """Release all stage-local route bindings for the given request ids."""
        for pool in self.stage_pools:
            pool.release_bindings(request_ids)

    async def _handle_collective_rpc(self, msg: CollectiveRPCRequestMessage) -> None:
        """Handle a control-plane RPC request from the main thread."""
        rpc_id = msg.rpc_id
        method = msg.method
        timeout = msg.timeout
        args = tuple(msg.args)
        kwargs = dict(msg.kwargs or {})
        requested_stage_ids = msg.stage_ids

        target_pools: list[StagePool] = []
        if requested_stage_ids is None:
            target_pools.extend(self.stage_pools)
        else:
            for lid in requested_stage_ids:
                if not (0 <= lid < self.num_stages):
                    logger.warning("[Orchestrator] collective_rpc: ignoring invalid stage_id %s", lid)
                    continue
                target_pools.append(self.stage_pools[lid])

        results: list[Any] = []
        stage_ids: list[int] = []
        for pool in target_pools:
            for replica_id in pool.live_replica_ids():
                stage_result = await pool.collective_rpc(
                    replica_id=replica_id,
                    method=method,
                    timeout=timeout,
                    args=args,
                    kwargs=kwargs,
                )
                stage_ids.append(pool.stage_id)
                results.append(stage_result)

        await self.rpc_async_queue.put(
            CollectiveRPCResultMessage(
                rpc_id=rpc_id,
                method=method,
                stage_ids=stage_ids,
                results=results,
            )
        )

    # ---- Orchestration loop ----

    async def _orchestration_output_handler(self) -> None:
        """Poll all stages, handle transfers, send final outputs to main."""
        try:
            await self._orchestration_loop()
        except asyncio.CancelledError:
            logger.debug("[Orchestrator] _orchestration_output_handler cancelled")
            return

    async def _orchestration_loop(self) -> None:
        """Poll stage pools and route logical outputs."""
        while not self._shutdown_event.is_set():
            idle = True
            for stage_id in range(self.num_stages):
                pool = self.stage_pools[stage_id]
                for replica_id in pool.live_replica_ids():
                    if self._shutdown_event.is_set():
                        return

                    if pool.stage_type == "diffusion":
                        output = pool.poll_diffusion_output(replica_id)
                        if output is None:
                            continue

                        await self._handle_processed_outputs(stage_id, replica_id, [output])
                        idle = False
                    else:
                        try:
                            raw_outputs = await pool.poll_llm_raw_output(replica_id, timeout_s=0.001)
                            if raw_outputs is None:
                                continue

                            await self._handle_kv_ready_raw_outputs(stage_id, raw_outputs)
                            for eco in raw_outputs.outputs:
                                req_state = self.request_states.get(getattr(eco, "request_id", None))
                                if req_state is None:
                                    continue
                                req_state.streaming.segment_finished = bool(getattr(eco, "is_segment_finished", False))
                                req_state.streaming.new_prompt_len_snapshot = getattr(
                                    eco,
                                    "new_prompt_len_snapshot",
                                    None,
                                )
                            raw_output = await pool.process_llm_raw_outputs(replica_id, raw_outputs)
                        except asyncio.CancelledError:
                            raise
                        except EngineDeadError as e:
                            logger.error(
                                "[Orchestrator] Stage-%s is dead: %s",
                                stage_id,
                                e,
                            )
                            # TODO: Fault handling is intentionally fail-stop at
                            # the orchestrator level today. If one replica in a
                            # logical stage dies, we promote it to `_fatal_error`,
                            # notify requests already admitted to that stage, and
                            # re-raise so `run()` shuts down all stages. This is
                            # conservative but means a single unhealthy replica in
                            # a multi-replica deployment can take down otherwise
                            # healthy replicas in other stages. Revisit this when
                            # adding per-replica fault isolation / eviction.
                            self._fatal_error = str(e)
                            self._fatal_error_stage_id = stage_id
                            for req_id, req_state in list(self.request_states.items()):
                                if stage_id in req_state.stage_submit_ts:
                                    await self.output_async_queue.put(
                                        ErrorMessage(
                                            error=str(e),
                                            fatal=True,
                                            request_id=req_id,
                                            stage_id=stage_id,
                                        )
                                    )
                                    self.request_states.pop(req_id, None)
                            self._shutdown_event.set()
                            raise
                        except Exception:
                            if self._shutdown_event.is_set():
                                return
                            logger.exception(
                                "[Orchestrator] Stage-%s replica-%s processing failed",
                                stage_id,
                                replica_id,
                            )
                            raise

                        await self._handle_processed_outputs(stage_id, replica_id, raw_output)
                        idle = False

            if idle:
                await asyncio.sleep(0.001)
            else:
                await asyncio.sleep(0)

    async def _handle_processed_outputs(self, stage_id: int, replica_id: int, outputs: list[Any]) -> None:
        """Route processed stage outputs produced by one stage poll."""
        pool = self.stage_pools[stage_id]
        for output in outputs:
            req_state = self.request_states.get(output.request_id)
            if req_state is None:
                logger.warning(
                    "[Orchestrator] Dropping output for unknown req %s at stage-%s (known reqs: %s)",
                    output.request_id,
                    stage_id,
                    list(self.request_states.keys()),
                )
                continue

            if getattr(output, "error", None) is not None:
                await self._handle_stage_error(stage_id, output)
                continue

            stage_metrics = None
            if output.finished:
                stage_metrics = pool.build_stage_metrics(
                    [output],
                    submit_ts=req_state.stage_submit_ts.get(stage_id, _time.time()),
                    replica_id=replica_id,
                )
                stage_metrics.pipeline_timings = dict(req_state.pipeline_timings)

            await self._route_output(stage_id, output, req_state, stage_metrics)

    async def _handle_stage_error(self, stage_id: int, output: Any) -> None:
        """Emit a frontend-visible error and clean up request state."""
        if self._cfg_tracker.is_companion(output.request_id):
            parent_id = self._cfg_tracker.get_parent_id(output.request_id) or output.request_id
        else:
            parent_id = output.request_id
        await self.output_async_queue.put(
            ErrorMessage(
                request_id=parent_id,
                stage_id=stage_id,
                error=output.error,
            )
        )
        await self._cleanup_request_ids(
            [parent_id, *self._cfg_tracker.cleanup_parent(parent_id)],
            abort=True,
        )

    # ---- Shared helpers ----

    async def _cleanup_request_ids(self, request_ids: list[str], *, abort: bool = False) -> None:
        """Release pool bindings and logical request state for the given ids."""
        if not request_ids:
            return

        if abort:
            await self._abort_request_ids(request_ids)
        self._release_request_bindings(request_ids)
        for request_id in request_ids:
            self._pd_kv_params.pop(request_id, None)
            self.request_states.pop(request_id, None)

    def _maybe_clone_diffusion_params_for_cfg(self, request_id: str, params: Any) -> Any:
        """Attach CFG companion ids to diffusion sampling params when needed."""
        companion_request_ids = self._cfg_tracker.get_companion_request_ids(request_id)
        if not companion_request_ids:
            return params

        import copy

        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        if not isinstance(params, OmniDiffusionSamplingParams):
            return params

        params = copy.deepcopy(params)
        params.cfg_kv_request_ids = companion_request_ids
        return params

    async def _route_output(
        self,
        stage_id: int,
        output: Any,
        req_state: OrchestratorRequestState,
        stage_metrics: Any,
    ) -> None:
        """Route a processed output: send to frontend and/or forward."""
        req_id = output.request_id
        finished = output.finished
        submit_ts = req_state.stage_submit_ts.get(stage_id)

        # CFG companion: stash output so parent can bundle [parent, *companions]
        # into source_outputs for the bridge (e.g. thinker2imagegen).
        if finished and self._cfg_tracker.is_companion(req_id):
            self._cfg_tracker.set_companion_output(req_id, output)
            await self._handle_cfg_companion_ready(req_id)
            await self._cleanup_request_ids([req_id])
            return

        if self.stage_pools[stage_id].final_output:
            await self.output_async_queue.put(
                OutputMessage(
                    request_id=req_id,
                    stage_id=stage_id,
                    engine_outputs=output,
                    metrics=stage_metrics,
                    finished=finished and stage_id == req_state.final_stage_id,
                    stage_submit_ts=submit_ts,
                )
            )
        elif stage_metrics is not None:
            await self.output_async_queue.put(
                StageMetricsMessage(
                    request_id=req_id,
                    stage_id=stage_id,
                    metrics=stage_metrics,
                    stage_submit_ts=submit_ts,
                )
            )

        if self._pd_pair is not None and finished and stage_id == self._pd_pair[0]:
            kv_params = getattr(output, "kv_transfer_params", None)
            if kv_params is not None:
                self._pd_kv_params[req_id] = kv_params if isinstance(kv_params, dict) else dict(kv_params)
            req_state.pd_prefill_multimodal_output = getattr(output, "multimodal_output", None)

        if (
            (finished or (req_state.streaming.enabled and req_state.streaming.segment_finished))
            and stage_id < req_state.final_stage_id
            and not self.async_chunk
            and (not self._next_stage_already_submitted(stage_id, req_state) or req_state.streaming.enabled)
        ):
            if (
                finished
                and self._cfg_tracker.has_companions(req_id)
                and not self._cfg_tracker.all_companions_done(req_id)
            ):
                self._cfg_tracker.defer_parent(req_id, output, stage_id)
            else:
                await self._forward_to_next_stage(
                    req_id,
                    stage_id,
                    output,
                    req_state,
                    is_streaming_session=req_state.streaming.enabled,
                    is_final_update=False,
                )
                if req_state.streaming.enabled and finished:
                    # For streaming sessions, send the terminal (resumable=False) update only on a finish
                    await self._forward_to_next_stage(
                        req_id,
                        stage_id,
                        output,
                        req_state,
                        is_streaming_session=True,
                        is_final_update=True,
                    )

        if finished and stage_id == req_state.final_stage_id:
            await self._cleanup_request_ids([req_id, *self._cfg_tracker.cleanup_parent(req_id)])

    def _next_stage_already_submitted(self, stage_id: int, req_state: OrchestratorRequestState) -> bool:
        return (stage_id + 1) in req_state.stage_submit_ts

    async def _handle_cfg_companion_ready(self, req_id: str) -> None:
        """Mark a CFG companion as done; if all companions are done, flush deferred parent."""
        parent_id = self._cfg_tracker.on_companion_completed(req_id)
        if parent_id is None:
            return

        deferred = self._cfg_tracker.pop_pending_parent(parent_id)
        if deferred is None:
            return

        parent_state = self.request_states.get(parent_id)
        if parent_state is None:
            return

        stage_id = deferred["stage_id"]
        if (stage_id + 1) in parent_state.stage_submit_ts:
            return

        await self._forward_to_next_stage(
            parent_id,
            stage_id,
            deferred["engine_outputs"],
            parent_state,
        )

    async def _handle_kv_ready_raw_outputs(
        self,
        stage_id: int,
        raw_outputs: EngineCoreOutputs,
    ) -> None:
        """Forward split requests once stage-0 KV is ready."""
        if self.async_chunk:
            return

        for raw_output in raw_outputs.outputs:
            kv_params = getattr(raw_output, "kv_transfer_params", None)
            if not (isinstance(kv_params, dict) and kv_params.get("kv_ready")):
                continue

            req_id = raw_output.request_id
            req_state = self.request_states.get(req_id)
            if req_state is None:
                continue
            if self._cfg_tracker.is_companion(req_id):
                await self._handle_cfg_companion_ready(req_id)
                continue
            if stage_id >= req_state.final_stage_id:
                continue
            if (stage_id + 1) in req_state.stage_submit_ts:
                continue

            if self._cfg_tracker.has_companions(req_id) and not self._cfg_tracker.all_companions_done(req_id):
                self._cfg_tracker.defer_parent(req_id, raw_output, stage_id)
            else:
                await self._forward_to_next_stage(req_id, stage_id, raw_output, req_state)

    def _build_pd_decode_params(self, req_id: str, sp: Any) -> Any:
        """Build decode-side sampling params with KV transfer params for PD routing.

        Clones the sampling params and injects kv_transfer_params that tell the
        decode engine where to pull the KV cache from (prefill engine's bootstrap addr).
        """
        sp = sp.clone()
        if sp.extra_args is None:
            sp.extra_args = {}

        # Get KV params captured from the prefill output (must include remote_request_id).
        kv_prefill_params = self._pd_kv_params.pop(req_id, None)
        if not kv_prefill_params or "remote_request_id" not in kv_prefill_params:
            raise RuntimeError(
                f"[Orchestrator][PD] Missing prefill kv_transfer_params.remote_request_id for req={req_id}"
            )

        decode_kv_params: dict[str, Any] = {
            "transfer_id": f"xfer-{req_id}",
        }

        if self._pd_bootstrap_addr:
            decode_kv_params["remote_bootstrap_addr"] = self._pd_bootstrap_addr

        if self._pd_prefill_engine_id:
            decode_kv_params["remote_engine_id"] = self._pd_prefill_engine_id

        # Overlay params from prefill side (includes remote_request_id set by monkey patch).
        decode_kv_params.update(kv_prefill_params)

        # Ensure these flags are set correctly after any overlay.
        decode_kv_params["do_remote_prefill"] = True
        decode_kv_params["do_remote_decode"] = False
        if not decode_kv_params.get("transfer_id"):
            decode_kv_params["transfer_id"] = f"xfer-{req_id}"

        sp.extra_args["kv_transfer_params"] = decode_kv_params

        logger.debug(
            "[Orchestrator][PD] decode kv_transfer_params for req=%s: %s",
            req_id,
            decode_kv_params,
        )
        return sp

    async def _forward_to_next_stage(
        self,
        req_id: str,
        src_stage_id: int,
        output: Any,
        req_state: OrchestratorRequestState,
        *,
        is_streaming_session: bool = False,
        is_final_update: bool = False,
    ) -> None:
        """Forward output from the current logical stage to the next one."""
        next_logical = src_stage_id + 1
        next_pool = self.stage_pools[next_logical]
        next_client = next_pool.stage_client
        params = req_state.sampling_params_list[next_logical]
        source_outputs = [output]
        next_stage_resumable = is_streaming_session and not is_final_update
        already_submitted = self._next_stage_already_submitted(src_stage_id, req_state)
        requires_multimodal_data = getattr(next_client, "requires_multimodal_data", False)

        if next_pool.stage_type == "diffusion":
            companion_outputs = self._cfg_tracker.pop_companion_outputs(req_id)
            expected = len(self._cfg_tracker.get_companion_request_ids(req_id))
            if expected > len(companion_outputs):
                logger.warning(
                    "[Orchestrator] req=%s: only %d/%d CFG companion outputs arrived; "
                    "downstream CFG conditioning may degrade",
                    req_id,
                    len(companion_outputs),
                    expected,
                )
            diffusion_source_outputs = [output, *companion_outputs]
            if next_client.custom_process_input_func is not None:
                _t_ar2d = _time.perf_counter()
                _fn = next_client.custom_process_input_func
                _extra_kwargs: dict[str, Any] = {}
                # TODO: replace signature probe with explicit kwarg contract.
                try:
                    import inspect as _inspect

                    if "sampling_params" in _inspect.signature(_fn).parameters:
                        _extra_kwargs["sampling_params"] = params
                except (TypeError, ValueError):
                    pass
                diffusion_prompt = _fn(
                    diffusion_source_outputs,
                    req_state.prompt,
                    requires_multimodal_data,
                    **_extra_kwargs,
                )
                _dt_ar2d = (_time.perf_counter() - _t_ar2d) * 1000
                req_state.pipeline_timings["ar2diffusion_ms"] = _dt_ar2d
                logger.info(
                    "[Orchestrator] ar2diffusion req=%s wall_time=%.3fms stage=%d->%d",
                    req_id,
                    _dt_ar2d,
                    src_stage_id,
                    next_logical,
                )
                if isinstance(diffusion_prompt, list):
                    if not diffusion_prompt:
                        error_output = OmniRequestOutput.from_error(
                            req_id,
                            f"Stage-{src_stage_id} produced no valid inputs for diffusion stage-{next_logical}",
                        )
                        logger.warning(
                            "[Orchestrator] req=%s stage=%d produced empty diffusion inputs for stage=%d; "
                            "routing terminal error output",
                            req_id,
                            src_stage_id,
                            next_logical,
                        )
                        await self.output_async_queue.put(
                            OutputMessage(
                                request_id=req_id,
                                stage_id=next_logical,
                                engine_outputs=error_output,
                                metrics=None,
                                finished=True,
                            )
                        )
                        await self._cleanup_request_ids(
                            [req_id, *self._cfg_tracker.cleanup_parent(req_id)],
                        )
                        return
                    if already_submitted and len(diffusion_prompt) == 1:
                        diffusion_prompt = diffusion_prompt[0]
            else:
                diffusion_prompt = req_state.prompt

            if already_submitted:
                await next_pool.submit_update(req_id, req_state, diffusion_prompt)
            else:
                await next_pool.submit_initial(
                    req_id,
                    req_state,
                    diffusion_prompt,
                    submit_kwargs={
                        "kv_sender_info": self._build_kv_sender_info(
                            list(getattr(next_client, "engine_input_source", None) or [src_stage_id]),
                            request_id=req_id,
                        )
                    },
                    params_override=self._maybe_clone_diffusion_params_for_cfg(req_id, params),
                )
            req_state.stage_submit_ts[next_logical] = _time.time()
            return

        # PD disaggregation: prefill → decode routing uses original prompt + KV transfer params
        if self._pd_pair is not None and (src_stage_id, next_logical) == self._pd_pair:
            params = self._build_pd_decode_params(req_id, params)

            # Use the original user prompt for the decode stage (not processed embeddings)
            original_prompt = req_state.prompt
            raw_decode_inputs = [original_prompt] if not isinstance(original_prompt, list) else original_prompt

            decode_inputs: list[dict[str, Any]] = []
            for decode_input in raw_decode_inputs:
                if isinstance(decode_input, dict):
                    decode_inputs.append(decode_input)
                    continue
                prompt_token_ids = getattr(decode_input, "prompt_token_ids", None)
                if prompt_token_ids is None:
                    raise TypeError(
                        "[Orchestrator][PD] decode input must be dict or have prompt_token_ids, "
                        f"got {type(decode_input).__name__} for req={req_id}"
                    )
                decode_inputs.append({"prompt_token_ids": list(prompt_token_ids)})

            for decode_input in decode_inputs:
                request = build_engine_core_request_from_tokens(
                    request_id=req_id,
                    prompt=decode_input,
                    params=params,
                    model_config=next_pool.stage_vllm_config.model_config,
                    mm_features=req_state.mm_features,
                    resumable=next_stage_resumable,
                )
                request.external_req_id = request.request_id
                if already_submitted:
                    await next_pool.submit_update(req_id, req_state, request)
                else:
                    await next_pool.submit_initial(req_id, req_state, request, prompt_text=None)

            req_state.stage_submit_ts[next_logical] = _time.time()
            return

        if req_state.pd_prefill_multimodal_output is not None:
            req_state.streaming.bridge_states.setdefault(
                "pd_prefill_multimodal_output_by_req",
                {},
            )[req_id] = req_state.pd_prefill_multimodal_output

        try:
            next_inputs = next_client.process_engine_inputs(
                source_outputs,
                req_state.prompt,
                streaming_context=req_state.streaming,
            )
        except Exception:
            logger.exception(
                "[Orchestrator] req=%s process_engine_inputs FAILED for stage-%s",
                req_id,
                next_logical,
            )
            raise

        # Build and submit requests for each input
        for next_input in next_inputs:
            # Only AR thinker stages consume encoder mm_features; downstream
            # (talker/code2wav/…) must not see them (avoids encoder-cache misses).
            model_stage = getattr(next_client, "model_stage", None)
            mm_features = req_state.mm_features if model_stage == "thinker" else None
            request = build_engine_core_request_from_tokens(
                request_id=req_id,
                prompt=next_input,
                params=params,
                model_config=next_pool.stage_vllm_config.model_config,
                mm_features=mm_features,
                resumable=next_stage_resumable,
            )

            request.external_req_id = request.request_id
            if already_submitted:
                await next_pool.submit_update(req_id, req_state, request)
            else:
                await next_pool.submit_initial(req_id, req_state, request, prompt_text=None)

        req_state.stage_submit_ts[next_logical] = _time.time()

    async def _prewarm_async_chunk_stages(
        self,
        request_id: str,
        stage0_request: Any,
        req_state: OrchestratorRequestState,
    ) -> None:
        """Pre-submit downstream stages for async-chunk mode."""
        if req_state.final_stage_id <= 0:
            return

        prompt_token_ids = getattr(stage0_request, "prompt_token_ids", None)
        if prompt_token_ids is None:
            logger.warning(
                "[Orchestrator] async_chunk prewarm skipped for req=%s: stage0 prompt_token_ids missing",
                request_id,
            )
            return

        for next_stage_id in range(1, req_state.final_stage_id + 1):
            next_pool = self.stage_pools[next_stage_id]
            params = req_state.sampling_params_list[next_stage_id]

            req_state.stage_submit_ts[next_stage_id] = _time.time()

            if next_pool.stage_type == "diffusion":
                await next_pool.submit_initial(
                    request_id,
                    req_state,
                    req_state.prompt,
                    submit_kwargs={
                        "kv_sender_info": self._build_kv_sender_info(
                            list(getattr(next_pool.stage_client, "engine_input_source", None) or [next_stage_id - 1]),
                            request_id=request_id,
                        )
                    },
                )
            else:
                import copy

                from vllm_omni.distributed.omni_connectors.adapter import compute_talker_prompt_ids_length

                try:
                    next_prompt_len = max(1, compute_talker_prompt_ids_length(prompt_token_ids))
                except Exception:
                    next_prompt_len = max(1, len(prompt_token_ids))

                original_prompt = req_state.prompt
                if isinstance(original_prompt, dict):
                    base_input = copy.deepcopy(original_prompt)
                else:
                    base_input = {}

                base_input["prompt_token_ids"] = [0] * next_prompt_len
                base_input["multi_modal_data"] = None
                base_input["mm_processor_kwargs"] = None

                request = build_engine_core_request_from_tokens(
                    request_id=request_id,
                    prompt=base_input,
                    params=params,
                    model_config=next_pool.stage_vllm_config.model_config,
                )
                request.external_req_id = request.request_id
                await next_pool.submit_initial(request_id, req_state, request, prompt_text=None)

    def _build_kv_sender_info(
        self,
        sender_stage_ids: list[int],
        *,
        request_id: str | None = None,
    ) -> dict[int, dict[str, Any]] | None:
        """Build per-request sender info for diffusion KV-transfer receivers."""
        sender_infos: dict[int, dict[str, Any]] = {}
        for sender_stage_id in dict.fromkeys(sender_stage_ids):
            if sender_stage_id < 0 or sender_stage_id >= len(self.stage_pools):
                continue

            sender_pool = self.stage_pools[sender_stage_id]
            sender_stage = sender_pool.get_bound_client(request_id) if request_id is not None else None
            if sender_stage is None:
                sender_stage = sender_pool.stage_client
            get_sender_info = getattr(sender_stage, "get_kv_sender_info", None)
            if not callable(get_sender_info):
                continue

            sender_info = get_sender_info()
            if not sender_info:
                logger.warning(
                    "[Orchestrator] Stage-%s has no KV sender info available",
                    sender_stage_id,
                )
                continue

            sender_infos[sender_stage_id] = sender_info

        return sender_infos or None

    # ---- Shutdown / lifecycle ----

    async def _drain_pending_requests_on_fatal(self) -> None:
        """Drain the request queue and broadcast fatal errors for any
        pending add_request messages that were never processed.

        Called from the ``run()`` finally block when a fatal error
        (e.g. ``EngineDeadError``) caused the orchestrator to shut down
        before the request handler could process all queued messages.
        Also broadcasts for any already-tracked requests still in
        ``request_states`` that were not yet notified.
        """
        assert self._fatal_error is not None

        notified: set[str] = set()

        # 1) Drain pending messages from the request queue.
        while True:
            try:
                msg = self.request_async_queue.get_nowait()
            except Exception:
                break
            if msg.type == "add_request":
                req_id = msg.request_id
                await self.output_async_queue.put(
                    ErrorMessage(
                        error=self._fatal_error,
                        fatal=True,
                        request_id=req_id,
                        stage_id=self._fatal_error_stage_id,
                    )
                )
                notified.add(req_id)

        # 2) Broadcast for any tracked requests not already notified
        #    (e.g. request was registered but the EngineDeadError handler
        #    missed it because it wasn't submitted to the dead stage yet).
        for req_id in list(self.request_states):
            if req_id not in notified:
                await self.output_async_queue.put(
                    ErrorMessage(
                        error=self._fatal_error,
                        fatal=True,
                        request_id=req_id,
                        stage_id=self._fatal_error_stage_id,
                    )
                )
            self.request_states.pop(req_id, None)

    # ---- Distributed-mode replica attach / detach ----

    async def _watch_replica_list(self) -> None:
        """Convert hub replica disappearances into unregister control messages."""
        last_up: set[tuple[int, str]] = set()
        while not self._shutdown_event.is_set():
            if self._hub is None:
                # No coordinator wired up; sleep coarsely and re-check shutdown.
                try:
                    await asyncio.sleep(self._WATCH_REPLICA_IDLE_INTERVAL_S)
                except asyncio.CancelledError:
                    raise
                continue

            try:
                snap = self._hub.get_replica_list()
                current = {(rep.stage_id, rep.input_addr) for rep in snap.replicas if rep.status == ReplicaStatus.UP}
                for stage_id, addr in last_up - current:
                    await self.request_async_queue.put(
                        UnregisterRemoteReplicaMessage(
                            stage_id=stage_id,
                            input_addr=addr,
                        )
                    )
                last_up = current
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("[Orchestrator] _watch_replica_list iteration failed")

            try:
                await asyncio.sleep(self._WATCH_REPLICA_INTERVAL_S)
            except asyncio.CancelledError:
                raise

    async def _handle_register_remote_replica(self, msg: RegisterRemoteReplicaMessage) -> None:
        """Bind a head-side client for a newly registered remote replica."""
        stage_id = int(msg.stage_id)
        replica_id = int(msg.replica_id)
        if not (0 <= stage_id < self.num_stages):
            logger.warning(
                "[Orchestrator] register_remote_replica: stage_id %d out of range (num_stages=%d)",
                stage_id,
                self.num_stages,
            )
            return
        if self._remote_replica_factory is None:
            logger.warning(
                "[Orchestrator] register_remote_replica received for stage=%d replica=%d but no factory installed",
                stage_id,
                replica_id,
            )
            return

        try:
            await self._attach_remote_replica(stage_id, replica_id)
        except Exception:
            logger.exception(
                "[Orchestrator] failed to attach remote replica stage=%d replica=%d",
                stage_id,
                replica_id,
            )

    async def _handle_unregister_remote_replica(self, msg: UnregisterRemoteReplicaMessage) -> None:
        """Tear down the head-side client for a vanished remote replica."""
        stage_id = int(msg.stage_id)
        input_addr = str(msg.input_addr)
        if not (0 <= stage_id < self.num_stages):
            return
        pool = self.stage_pools[stage_id]
        affected = pool.invalidate_addr(input_addr)
        self._detach_remote_replica(stage_id, input_addr)
        if affected:
            await self._cleanup_request_ids(affected, abort=True)
            for req_id in affected:
                await self.output_async_queue.put(
                    ErrorMessage(
                        error="stage replica disappeared",
                        request_id=req_id,
                        stage_id=stage_id,
                    )
                )

    async def _attach_remote_replica(self, stage_id: int, replica_id: int) -> None:
        """Build a head-side stage client via the injected factory and register it."""
        factory = self._remote_replica_factory
        if factory is None:
            return
        pool = self.stage_pools[stage_id]
        client = await factory(stage_id, replica_id)
        input_addr = StagePool._client_input_addr(client)
        if input_addr is None:
            raise RuntimeError(
                f"remote replica factory for stage {stage_id} produced a client without a discoverable input address"
            )
        pool.add_client(input_addr, client)
        logger.info(
            "[Orchestrator] attached remote replica stage=%d replica=%d addr=%s",
            stage_id,
            replica_id,
            input_addr,
        )

    def _detach_remote_replica(self, stage_id: int, input_addr: str) -> None:
        """Shut down + remove the head-side client at ``input_addr``."""
        pool = self.stage_pools[stage_id]
        client = pool.remove_client(input_addr)
        if client is None:
            return
        try:
            client.shutdown()
        except Exception:
            logger.exception(
                "[Orchestrator] failed to shutdown client for stage=%d addr=%s",
                stage_id,
                input_addr,
            )
        logger.info(
            "[Orchestrator] detached remote replica stage=%d addr=%s",
            stage_id,
            input_addr,
        )

    def _shutdown_stages(self) -> None:
        """Shutdown all stage pools."""
        if self._stages_shutdown:
            return

        self._stages_shutdown = True
        total = sum(pool.live_num_replicas for pool in self.stage_pools)
        logger.info("[Orchestrator] Shutting down all %d client(s)", total)
        for pool in self.stage_pools:
            for replica_id in pool.live_replica_ids():
                pool.shutdown_replica(replica_id)
