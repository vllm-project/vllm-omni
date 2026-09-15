# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""AsyncOmniBase: the asyncio-side foundation shared by ``AsyncOmni`` and ``DuplexOmni``.

Owns the engine output pump (one consumer of the engine output queue that
routes messages to per-request queues, ACK resolvers, or a subclass hook),
engine-dead fan-out, health/shutdown, and the config accessors. Turn-based
request handling lives in ``AsyncOmni``; duplex sessions in ``DuplexOmni``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Iterable
from typing import Any

from vllm.logger import init_logger
from vllm.outputs import CompletionOutput
from vllm.plugins.io_processors import get_io_processor
from vllm.utils import random_uuid
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.diffusion.data import OmniACK
from vllm_omni.engine.messages import ErrorMessage, OutputMessage
from vllm_omni.entrypoints.omni_base import (
    OmniBase,
    OmniEngineDeadError,
)
from vllm_omni.metrics.stats import OrchestratorAggregator as OrchestratorMetrics
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)
_FINAL_OUTPUT_IDLE_SLEEP_S = 0.001
# Blocking-wait interval for the event-driven final-output drain
# (VLLM_OMNI_EVENT_DRIVEN_ORCH=1): a message wakes the drain immediately via
# the janus queue's condition variable; this timeout only bounds how often the
# orchestrator liveness check runs while the pipeline is idle.
_FINAL_OUTPUT_BLOCKING_WAIT_S = 1.0


class AsyncEventResolver:
    """
    A generic signal aggregator designed for synchronized handshakes in
    distributed or multi-stage environments. Supports waiting for a specified
    number (expected_count) of worker signals in both inline and multiprocess modes.
    """

    def __init__(self, orchestrator=None):
        self._pending_tasks: dict[str, dict] = {}
        self.orchestrator = orchestrator
        self._lock = asyncio.Lock()

    def watch_task(self, task_id: str, expected_count: int = 1) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._pending_tasks[task_id] = {
            "future": fut,
            "expected_count": expected_count,
            "received": [],
            "start_time": time.time(),
        }
        return fut

    async def resolve(self, ack: OmniACK):
        tid = getattr(ack, "task_id", None)

        if tid is None and isinstance(ack, dict):
            tid = ack.get("task_id")

        async with self._lock:
            task_info = self._pending_tasks.get(tid)
            if task_info is None:
                logger.warning(f"Received stray ACK for task_id {tid}. Task might have timed out.")
                return

            task_info["received"].append(ack)
            current_count = len(task_info["received"])
            expected = task_info["expected_count"]

            orchestrator = self.orchestrator
            if orchestrator and hasattr(orchestrator, "metrics") and orchestrator.metrics:
                freed = getattr(ack, "freed_bytes", 0)
                if freed == 0 and isinstance(ack, dict):
                    freed = ack.get("freed_bytes", 0)
                orchestrator.metrics.record_vram_reclaimed(freed)

            logger.info(f"[Resolver] Task {tid} progress: {current_count}/{expected} ACKs received.")

            if current_count >= expected:
                self._pending_tasks.pop(tid)
                fut = task_info["future"]
                if not fut.done():
                    elapsed = time.time() - task_info["start_time"]
                    logger.info(f"[Resolver] Task {tid} completed successfully in {elapsed:.2f}s.")
                    fut.set_result(task_info["received"])


class AsyncOmniBase(OmniBase):
    """Shared asyncio foundation of the async entrypoints (see module docstring)."""

    def __init__(self, *args: Any, model: str = "", **kwargs: Any) -> None:
        OmniBase.__init__(self, model=model, **kwargs)
        self.final_output_task: asyncio.Task | None = None
        self.event_resolver = AsyncEventResolver(orchestrator=self)
        self.config_path = self.engine.config_path
        self.input_processor = self.engine.input_processor
        self.endpoint_restrictions = self.engine.endpoint_restrictions

        stage_index = self._get_comprehension_stage_index()
        if stage_index is None:
            self.io_processor = None
        else:
            vllm_config = self.engine.stage_vllm_configs[stage_index]
            io_processor_plugin = vllm_config.model_config.io_processor_plugin
            renderer = self.renderer
            if renderer is None:
                from vllm.renderers import renderer_from_config

                renderer = renderer_from_config(vllm_config)
            self.io_processor = get_io_processor(vllm_config, renderer, io_processor_plugin)

    # ==================== Subclass seam ====================

    def _route_engine_message(self, msg: Any) -> bool:
        """Consume an engine output message the subclass owns (return True when consumed)."""
        del msg
        return False

    def _on_engine_dead(self, error: str) -> None:
        """Seam: the output pump is ending because the engine died (default: nothing).

        ``DuplexOmni`` ends every session handle here; turn-based requests are
        already failed through their per-request queues.
        """
        del error

    def _resolve_transfer_replica(self, stage_id: int, request_id: str) -> int | None:
        """Look up the sticky-routed replica for (stage_id, request_id).

        Used as the ``replica_resolver`` callback by ``OrchestratorAggregator``
        to label transfer_* metrics without plumbing replica ids through
        ``TransferEdgeStats`` / ``StageRequestStats`` / connector adapters.
        Returns None when stage_id is out of range or the request hasn't been
        bound to a replica yet — the metric emit then defensive-skips.
        """
        pools = getattr(self.engine, "stage_pools", None)
        if pools is None or not (0 <= stage_id < len(pools)):
            return None
        return pools[stage_id].get_bound_replica_id(request_id)

    def _get_comprehension_stage_index(self) -> int | None:
        fallback_idx: int | None = None
        for idx, stage_client in enumerate(self.engine.stage_clients):
            stage_vllm_config = self.engine.stage_vllm_configs[idx]
            if stage_vllm_config is None:
                continue
            if fallback_idx is None:
                fallback_idx = idx
            if stage_client.is_comprehension:
                return idx
        return fallback_idx

    @property
    def renderer(self):
        """Return the renderer from the engine input processor when available."""
        if self.input_processor is None:
            return None
        return self.input_processor.renderer

    @property
    def vllm_config(self):
        """Return the vLLM config for the comprehension stage when present."""
        stage_index = self._get_comprehension_stage_index()
        if stage_index is None:
            return None
        return self.engine.stage_vllm_configs[stage_index]

    async def get_vllm_config(self) -> Any:
        """Compatibility helper for call sites expecting async vllm config access."""
        return self.vllm_config

    def get_diffusion_od_config(self) -> Any | None:
        """Return the diffusion-stage config when the pipeline has one."""
        saw_diffusion_stage = False
        for stage_client in self.engine.stage_clients:
            if getattr(stage_client, "stage_type", None) != "diffusion":
                continue

            saw_diffusion_stage = True

            od_config = getattr(stage_client, "od_config", None)
            if od_config is not None:
                return od_config

            inner_engine = getattr(stage_client, "_engine", None)
            od_config = getattr(inner_engine, "od_config", None)
            if od_config is not None:
                return od_config

        # Out-of-process diffusion clients don't carry od_config (it lives in the
        # worker); fall back to the engine's model_class_name resolution.
        if saw_diffusion_stage:
            return self.engine.get_diffusion_od_config()

        return None

    @property
    def model_config(self):
        """Return the model config for the comprehension stage when present."""
        vllm_config = self.vllm_config
        if vllm_config is None:
            return None
        return vllm_config.model_config

    @staticmethod
    def _get_unique_request_id(external_request_id: str):
        """Get a random new request ID for this request; at the server level,
        this is usually set by the calling entrypoint, but in direct calls, we
        need to set it explicitly since we do not allow empty IDs.

        NOTE: in the upstream vLLM, this is done in the InputProcessor's
        `assign_request_id`.
        """
        uuid = random_uuid()
        prefix = "" if not external_request_id else f"{external_request_id}-"
        return f"{prefix}{uuid:.8}"

    # ==================== Processing Methods ====================

    async def _process_orchestrator_results(
        self,
        request_id: str,
        metrics: OrchestratorMetrics,
        final_stage_id_for_e2e: int,
        req_start_ts: dict[str, float],
        wall_start_ts: float,
    ) -> AsyncGenerator[OmniRequestOutput, None]:
        """Read results from the Orchestrator (via the request's asyncio.Queue)
        and yield OmniRequestOutput objects.

        The Orchestrator handles all stage-to-stage transfers. This method
        only processes final outputs that arrive on the per-request queue.
        """
        req_state = self.request_states.get(request_id)
        if req_state is None:
            return

        while True:
            result = await req_state.queue.get()

            if isinstance(result, ErrorMessage):
                logger.error(
                    "[AsyncOmni] Orchestrator error for req=%s stage-%s: %s",
                    request_id,
                    result.stage_id,
                    result.error,
                )
                if result.fatal:
                    raise OmniEngineDeadError(
                        result.error,
                        error_stage_id=result.stage_id,
                    )
                self._raise_nonfatal_error_message(result)

            if not isinstance(result, OutputMessage):
                logger.warning("[AsyncOmni] Dropping unexpected per-request message %r", result)
                continue

            stage_id = result.stage_id

            self._check_engine_output_error(result, request_id, stage_id)

            # Process the result (constructs OmniRequestOutput)
            output_to_yield = self._process_single_result(
                result,
                stage_id,
                metrics,
                req_start_ts,
                wall_start_ts,
                final_stage_id_for_e2e,
            )

            if output_to_yield:
                # Set the external request ID back to the user yielded input
                output_to_yield.request_id = req_state.external_request_id or output_to_yield.request_id
                logger.debug(
                    "[AsyncOmni] req=%s stage-%s yielding final_output_type=%s",
                    request_id,
                    stage_id,
                    getattr(output_to_yield, "final_output_type", None),
                )
                yield output_to_yield

            # The Orchestrator sets "finished" when the final stage is done
            if result.finished:
                break

    # ==================== Output Handler ====================

    def _final_output_handler(self) -> None:
        """Start the final output handler if not already running.

        This handler reads messages from the Orchestrator output queue and
        routes them to per-request asyncio.Queues.
        """
        if self.final_output_task is not None:
            return

        engine = self.engine

        # Event-driven drain (VLLM_OMNI_EVENT_DRIVEN_ORCH=1): block on the
        # queue's condition variable in a dedicated thread instead of the
        # get_nowait + 1 ms sleep cadence. Same flag as the orchestrator-side
        # event-driven loop (vllm_omni/engine/orchestrator.py).
        from vllm_omni.engine.orchestrator import _event_driven_orch_enabled

        event_driven_drain = _event_driven_orch_enabled() and hasattr(engine, "get_output_blocking_async")

        async def _final_output_loop():
            """Background coroutine that dispatches final outputs to request queues."""
            try:
                while True:
                    if event_driven_drain:
                        msg = await engine.get_output_blocking_async(timeout=_FINAL_OUTPUT_BLOCKING_WAIT_S)
                        if msg is None:
                            # Timed out with the orchestrator alive; loop for
                            # the periodic liveness check.
                            continue
                    else:
                        msg = await engine.try_get_output_async()
                        if msg is None:
                            await asyncio.sleep(_FINAL_OUTPUT_IDLE_SLEEP_S)
                            continue

                    if self._route_engine_message(msg):
                        continue

                    if isinstance(msg, dict) and msg.get("type") == "ack":
                        ack_data = msg.get("ack")
                        tid = getattr(ack_data, "task_id", "unknown")
                        logger.info(f"[{self._name}] Intercepted wrapped ACK for task {tid}")
                        await self.event_resolver.resolve(ack_data)
                        continue
                    if isinstance(msg, OmniACK):
                        logger.info(f"[{self._name}] Intercepted raw ACK object: {msg.task_id}")
                        await self.event_resolver.resolve(msg)
                        continue
                    if hasattr(msg, "task_id"):
                        tid = getattr(msg, "task_id")
                        logger.info(f"[{self._name}] Intercepted task-ID object: {tid}")
                        await self.event_resolver.resolve(msg)
                        continue

                    if isinstance(msg, ErrorMessage):
                        # Route request-scoped errors to that request's queue and
                        # keep the loop alive. A request whose stage replica died
                        # and was evicted gets a fatal error delivered here; only
                        # that request fails (its consumer raises), the server
                        # stays up for other stages/requests (#4285). A fatal
                        # error without a request_id is a genuine engine-wide
                        # death and falls through to the except handler below.
                        if msg.request_id is not None:
                            req_state = self.request_states.get(msg.request_id)
                            if req_state is not None:
                                await req_state.queue.put(msg)
                            else:
                                logger.warning(
                                    "[%s] dropping error for unknown req %s",
                                    self._name,
                                    msg.request_id,
                                )
                            continue
                        if not msg.fatal:
                            continue

                    should_continue, _, stage_id, req_state = self._handle_output_message(msg)
                    if should_continue:
                        continue

                    req_state.stage_id = stage_id

                    # Route to the per-request queue
                    await req_state.queue.put(msg)

            except asyncio.CancelledError:
                raise
            except OmniEngineDeadError as e:
                logger.error("[%s] Engine dead: %s", self._name, e)
                for req_state in list(self.request_states.values()):
                    error_msg = ErrorMessage(
                        error=str(e),
                        fatal=True,
                        request_id=req_state.request_id,
                        stage_id=e.error_stage_id,
                    )
                    await req_state.queue.put(error_msg)
                self._on_engine_dead(str(e))
            except EngineDeadError as e:
                logger.error("[%s] Engine dead: %s", self._name, e)
                for req_state in list(self.request_states.values()):
                    error_msg = ErrorMessage(
                        error=str(e),
                        fatal=True,
                        request_id=req_state.request_id,
                    )
                    await req_state.queue.put(error_msg)
                self._on_engine_dead(str(e))
            except Exception as e:
                logger.exception("[%s] final_output_loop failed.", self._name)
                for req_state in list(self.request_states.values()):
                    error_msg = ErrorMessage(
                        request_id=req_state.request_id,
                        error=str(e),
                    )
                    await req_state.queue.put(error_msg)
                self.final_output_task = None
                self._on_engine_dead(str(e))

        self.final_output_task = asyncio.create_task(_final_output_loop())
        logger.debug("[AsyncOmni] Final output handler started")

    async def _abort_internal_requests(self, request_id: str | Iterable[str]):
        """Abort request(s) via the Orchestrator given internal request IDs,
        which take the format <external_request_id>-<UUID>.
        """
        request_ids = [request_id] if isinstance(request_id, str) else list(request_id)
        # Request IDs are already internal, so we just need to get the matching states.
        internal_req_ids = [rid for rid in request_ids if rid in self.request_states]
        await self._abort(internal_req_ids)

    async def _abort(self, request_ids: list[str]) -> None:
        """Abort request IDs via the engine and enqueue terminal abort outputs.

        Waits for orchestrator abort acknowledgment, enqueues any AR terminal
        abort outputs (partial tokens) into each request's asyncio queue, then
        cancels the input pump. Frontend ``request_states`` stay registered so
        ``generate()`` can consume the terminal message in
        ``_process_orchestrator_results`` and run normal cleanup.

        When ``abort_async`` returns no output for an active request (OP not
        registered yet, unbound replica, or orchestrator id drop), enqueue a
        synthetic finished abort so ``generate()`` cannot hang on ``queue.get``.
        """
        abort_outputs = await self.engine.abort_async(request_ids) or []
        delivered: set[str] = set()
        for output_msg in abort_outputs:
            req_id = getattr(output_msg, "request_id", None)
            if req_id is None:
                continue
            state = self.request_states.get(req_id)
            if state is None:
                logger.debug("[AsyncOmni] Dropping abort output for unknown req %s", req_id)
                continue
            await state.queue.put(output_msg)
            delivered.add(req_id)
        for rid in request_ids:
            state = self.request_states.get(rid)
            if state is not None and rid not in delivered:
                queue = getattr(state, "queue", None)
                if queue is not None:
                    await state.queue.put(self._synthetic_abort_output_message(rid))
                    delivered.add(rid)
        for rid in request_ids:
            self._record_request_failure_once(rid, reason="client_abort")
            state = self.request_states.get(rid)
            input_stream_task = getattr(state, "input_stream_task", None)
            if input_stream_task is not None and not input_stream_task.done():
                input_stream_task.cancel()
        if self.log_stats:
            logger.info("[AsyncOmni] Aborted request(s) %s", ",".join(request_ids))

    @staticmethod
    def _synthetic_abort_output_message(request_id: str) -> OutputMessage:
        """Terminal abort OutputMessage used when the engine returned none."""
        engine_output = OmniRequestOutput(
            request_id=request_id,
            finished=True,
            stage_id=0,
            final_output_type="text",
            outputs=[
                CompletionOutput(
                    index=0,
                    text="",
                    token_ids=[],
                    cumulative_logprob=None,
                    logprobs=None,
                    finish_reason="abort",
                    stop_reason=None,
                )
            ],
        )
        return OutputMessage(
            request_id=request_id,
            stage_id=0,
            replica_id=None,
            engine_outputs=engine_output,
            metrics=None,
            finished=True,
            stage_submit_ts=None,
        )

    @property
    def is_running(self) -> bool:
        """Check if the engine is running."""
        orchestrator_alive = self.engine.is_alive()
        task_alive = self.final_output_task is not None and not self.final_output_task.done()
        return orchestrator_alive and task_alive

    @property
    def _name(self) -> str:
        return type(self).__name__

    @property
    def is_stopped(self) -> bool:
        """EngineClient abstract property implementation."""
        return self.errored

    @property
    def dead_error(self) -> BaseException:
        """EngineClient abstract property implementation."""
        return OmniEngineDeadError()

    # ==================== EngineClient Interface ====================

    async def check_health(self) -> None:
        """Check engine health by verifying the Orchestrator process is alive."""
        OmniBase.check_health(self)

    # ==================== Shutdown ====================

    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown the engine."""
        if self.final_output_task is not None:
            self.final_output_task.cancel()
            self.final_output_task = None
        OmniBase.shutdown(self)
