# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Stage Core Process for vLLM-Omni V1 architecture.

StageEngineCoreProc inherits from vLLM's EngineCoreProc and runs the engine core
busy loop in a subprocess, communicating with StageEngineCoreClient via ZMQ.
"""

from __future__ import annotations

import contextlib
import os
import queue
import signal
from typing import Any

import vllm.v1.engine.core as _vllm_engine_core_module
from vllm.logger import init_logger
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value,
)
from vllm.utils.system_utils import (
    decorate_logs,
    set_process_title,
)
from vllm.v1.engine import EngineCoreRequestType
from vllm.v1.engine.core import EngineCoreProc, EngineShutdownState
from vllm.v1.engine.utils import (
    EngineZmqAddresses,
    SignalCallback,
)
from vllm.v1.executor.uniproc_executor import UniProcExecutor

from vllm_omni.distributed.omni_coordinator import create_stage_coord_client
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.stage_init_utils import (
    maybe_apply_cfg_scheduler_patches,
    set_death_signal,
)

logger = init_logger(__name__)


_SIGNAL_EXIT_BASE = 128
_OMNI_CHUNK_READY = object()
_OMNI_CHUNK_MAINTENANCE_S = 0.1


def _install_phase_locks(kwargs: dict[str, Any], local_dp_rank: int) -> None:
    """Wrap ``kwargs["executor_class"]`` with the SH/EX phase-lock guard.

    Fail-closed: ``parallel_stage_init`` promises that every memory-mutating
    init phase in this child runs under the per-device locks, so a missing
    ``vllm_config`` or ``executor_class`` must abort the launch rather than
    silently proceed with an unguarded parallel initialization.
    """
    from vllm_omni.engine.stage_phase_lock import (
        DevicePhaseLock,
        wrap_executor_with_phase_locks,
    )

    missing = [key for key in ("vllm_config", "executor_class") if kwargs.get(key) is None]
    if missing:
        raise RuntimeError(
            f"parallel_stage_init is enabled but EngineCore kwargs are missing {missing}, "
            "so the SH/EX phase-lock guard cannot be installed. Refusing to run an "
            "unguarded parallel initialization; fix the launch plumbing or disable "
            "parallel_stage_init."
        )
    locker = DevicePhaseLock.from_child(kwargs["vllm_config"], local_dp_rank)
    kwargs["executor_class"] = wrap_executor_with_phase_locks(kwargs["executor_class"], locker)
    logger.info(
        "[StageEngineCoreProc] parallel_stage_init: SH/EX phase locks on devices %s",
        locker.device_ids,
    )


def _signal_exit_code(signum: int) -> int:
    """Return the conventional process exit code for signal-driven exits."""
    return _SIGNAL_EXIT_BASE + signum


def _bind_native_data_plane_ready_sink(model_executor: Any, scheduler: Any, wakeup: Any = None) -> bool:
    """Bind the TP1 in-process runner control plane directly to its scheduler."""
    if not isinstance(model_executor, UniProcExecutor):
        return False
    parallel_config = model_executor.vllm_config.parallel_config
    if parallel_config.tensor_parallel_size != 1 or parallel_config.pipeline_parallel_size != 1:
        return False
    driver_worker = getattr(model_executor, "driver_worker", None)
    worker = getattr(driver_worker, "worker", None)
    model_runner = getattr(worker, "model_runner", None)
    data_plane = getattr(model_runner, "_omni_data_plane", None)
    sink = getattr(scheduler, "enqueue_omni_connector_output", None)
    if data_plane is None or not callable(sink):
        return False
    if (
        os.getenv("VLLM_OMNI_GENERATION_PAYLOAD_NATIVE", "0") == "1"
        and callable(getattr(scheduler, "has_runnable_omni_chunks", None))
        and getattr(getattr(model_runner, "model", None), "supports_native_payload_input", False)
    ):
        data_plane.payload_native = True
        scheduler.input_coordinator.payload_native = True
        logger.info("Native MRv2 generation uses payload-only inputs and control-token scheduling")
    if wakeup is not None:
        original_sink = sink

        def sink_and_wake(output: Any) -> None:
            original_sink(output)
            wakeup()

        sink = sink_and_wake
    data_plane.set_omni_connector_output_sink(sink)
    return True


class StageEngineCoreProc(EngineCoreProc):
    """Stage-specific engine core process for vLLM-Omni.

    Inherits from EngineCoreProc and provides its own ``run_stage_core``
    entry point for launching in a subprocess.  Does **not** delegate to
    ``EngineCoreProc.run_engine_core()``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._omni_completion_observer = None
        self._omni_chunk_wakeup = False
        super().__init__(*args, **kwargs)
        requested = (
            os.getenv("VLLM_OMNI_CHUNK_ENGINE_WAKEUP", "0") == "1"
            and getattr(self.scheduler, "_native_data_plane", False)
            and callable(getattr(self.scheduler, "has_runnable_omni_chunks", None))
            and self.model_executor.vllm_config.parallel_config.data_parallel_size == 1
            and self.scheduler.connector is None
            and self.scheduler.ec_connector is None
        )
        wakeup = (lambda: self.input_queue.put((_OMNI_CHUNK_READY, None))) if requested else None
        if _bind_native_data_plane_ready_sink(self.model_executor, self.scheduler, wakeup):
            logger.info("Bound native MRv2 connector readiness directly to the scheduler inbox.")
            self._omni_chunk_wakeup = requested
            if requested:
                self.scheduler._omni_chunk_wakeup_bound = True
                logger.info("Native MRv2 generation EngineCore chunk-ready wakeup enabled")
                if os.getenv("VLLM_OMNI_GENERATION_COMPLETION_EVENTS", "0") == "1" and self.batch_queue is not None:
                    from vllm_omni.engine.generation_completion import GenerationCompletionObserver

                    self._omni_completion_observer = GenerationCompletionObserver(wakeup)

    def has_work(self) -> bool:
        if not getattr(self, "_omni_chunk_wakeup", False):
            return super().has_work()
        if self.shutdown_state != EngineShutdownState.RUNNING:
            # Restore upstream lifetime-based stepping during shutdown. Apart
            # from keeping parked streams alive, this continues scheduling
            # their timeout/cleanup work while graceful drain is in progress.
            self.scheduler._omni_chunk_wakeup_bound = False
        observer = getattr(self, "_omni_completion_observer", None)
        if observer is not None and self.shutdown_state == EngineShutdownState.RUNNING:
            if self.batch_queue:
                if observer.ready(self.batch_queue[-1][0]):
                    return True
                return len(self.batch_queue) < self.batch_queue_size and self.scheduler.has_requests()
        return super().has_work()

    def step_with_batch_queue(self):
        observer = getattr(self, "_omni_completion_observer", None)
        if observer is None:
            return super().step_with_batch_queue()
        # Generation still uses sample_tokens to construct its async output,
        # even though it does not sample autoregressive tokens.
        batch_queue = self.batch_queue
        model_executed = False
        if len(batch_queue) < self.batch_queue_size and self.scheduler.has_requests():
            scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())
            with self.log_error_detail(scheduler_output):
                exec_future = self.model_executor.execute_model(scheduler_output, non_block=True)
            model_executed = self.is_ec_consumer and scheduler_output.total_num_scheduled_tokens > 0
            future = exec_future
            if not self.is_pooling_model and model_executed:
                grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
                future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft((future, scheduler_output, exec_future))
        if not batch_queue:
            return None, model_executed
        future, scheduler_output, exec_future = batch_queue[-1]
        if not observer.ready(future):
            return None, model_executed
        batch_queue.pop()
        with (
            self.capture_iteration_details(scheduler_output) as iteration_details,
            self.log_error_detail(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                exec_future.result()
                raise RuntimeError("unexpected error")
        observer.consumed(future)
        self._process_aborts_queue()
        outputs = self.scheduler.update_from_output(scheduler_output, model_output)
        self._attach_iteration_details(outputs, iteration_details)
        return outputs, model_executed

    def shutdown(self):
        observer = getattr(self, "_omni_completion_observer", None)
        if observer is not None:
            observer.close()
            self._omni_completion_observer = None
        return super().shutdown()

    def _handle_client_request(self, request_type: Any, request: Any) -> None:
        if request_type is _OMNI_CHUNK_READY:
            return
        return super()._handle_client_request(request_type, request)

    def _process_input_queue(self) -> None:
        if not getattr(self, "_omni_chunk_wakeup", False):
            return super()._process_input_queue()
        # The native receiver writes to this same queue after publishing its
        # scheduler inbox entry. queue.get makes arrival-before-wait safe;
        # merely blocking on the scheduler's separate inbox would miss ADD,
        # ABORT and utility messages. Periodic control ticks retain connector
        # deadlines and deferred cleanup without issuing continuous empty work.
        while not self.has_work() and self.is_running():
            # Preserve upstream idle-callback semantics: waiting for a chunk
            # is a scheduling pause, not the end of a live request lifetime.
            if not self.scheduler.requests and not self.batch_queue:
                self._notify_idle_state_callbacks()
            if self.input_queue.empty():
                with self.aborts_queue.mutex:
                    self.aborts_queue.queue.clear()
            try:
                timeout = _OMNI_CHUNK_MAINTENANCE_S if (self.scheduler.requests or self.batch_queue) else None
                block = self.process_input_queue_block
                item = self.input_queue.get(block=block, timeout=timeout if block else None)
            except queue.Empty:
                self.scheduler._omni_maintenance_due = True
                break
            self._handle_client_request(*item)
            if not block:
                break
        while not self.input_queue.empty():
            self._handle_client_request(*self.input_queue.get_nowait())

    def preprocess_add_request(self, request: OmniEngineCoreRequest) -> tuple[Any, int]:
        """Preserve omni payloads when vLLM builds its scheduler request."""
        scheduler_request, current_wave = super().preprocess_add_request(request)
        scheduler_request.additional_information = request.additional_information
        scheduler_request.external_req_id = getattr(request, "external_req_id", request.request_id)
        return scheduler_request, current_wave

    @staticmethod
    def run_stage_core(
        *args: Any,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
        omni_coordinator_address: str | None = None,
        omni_stage_id: int | None = None,
        omni_replica_id: int = 0,
        omni_parallel_stage_init: bool = False,
        **kwargs: Any,
    ) -> None:
        """Launch StageEngineCoreProc busy loop in background process.

        Omni-specific kwargs:
          - ``omni_coordinator_address``: ROUTER address of the head-side
            :class:`OmniCoordinator`. When provided, this subprocess
            instantiates an :class:`OmniCoordClientForStage` after the
            HELLO/INIT/READY handshake completes and reports its status +
            queue length via heartbeats. The hook is wired so each
            heartbeat refreshes ``queue_length`` from the live scheduler.
          - ``omni_stage_id``: logical stage id this replica belongs to.
            Required when ``omni_coordinator_address`` is provided.
          - ``omni_replica_id``: cluster-unique replica id within the
            stage (assigned by :class:`OmniMasterServer`). Used for
            logging / metrics only.
        """
        signal_callback: SignalCallback | None = None
        maybe_register_config_serialize_by_value()

        # Register vllm-omni reasoning parsers (e.g. step_audio) in this
        # subprocess so they are available when the engine core resolves
        # ``--reasoning-parser``.  The main process already registered them
        # at import time, but the forked subprocess starts with a fresh
        # ReasoningParserManager.
        try:
            import vllm_omni.reasoning  # noqa: F401
        except ImportError:
            logger.warning(
                "Failed to import vllm_omni.reasoning in subprocess; "
                "custom reasoning parsers (e.g. step_audio) will not be "
                "available."
            )

        engine_core: StageEngineCoreProc | None = None
        coord_client = None
        try:
            # NOTE: previous revisions hardcoded data_parallel_size=1 here
            # (TODO referencing issue #984). The hardcoding has been removed
            # so the DP fields propagate through from the caller exactly
            # like upstream vLLM.

            stage_label = f"stage{omni_stage_id}" if omni_stage_id is not None else "noid"
            set_death_signal(signal.SIGTERM)
            set_process_title(f"StageEngineCoreProc_{stage_label}_replica{omni_replica_id}_DP{dp_rank}")
            decorate_logs()
            # Workaround for flashinfer/jit-cache version mismatch in CI.
            # The parent process handles this gracefully via ring_globals.py,
            # but the subprocess hits an unprotected import in TopKTopPSampler.
            # Setting this env var allows the same graceful fallback to work.
            os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
            os.environ["VLLM_OMNI_REPLICA_ID"] = str(max(int(omni_replica_id), 0))

            # Patch the decoder type so process_input_sockets (started
            # during __init__) decodes OmniEngineCoreRequest (which
            # carries additional_information) instead of the base
            # EngineCoreRequest.  Must happen BEFORE __init__ because
            # the IO thread creates MsgpackDecoder(EngineCoreRequest)
            # during __init__.
            _vllm_engine_core_module.EngineCoreRequest = OmniEngineCoreRequest
            logger.debug(
                "[StageEngineCoreProc] Patched EngineCoreRequest -> OmniEngineCoreRequest: %s",
                _vllm_engine_core_module.EngineCoreRequest,
            )

            # CFG pairing scheduler patches must land before EngineCore builds
            # its Scheduler; gated on the stage's logits_processors and its
            # default sampling extra_args.
            maybe_apply_cfg_scheduler_patches(kwargs.get("vllm_config"))

            # When parallel stage init is enabled, wrap this driver's executor
            # so its memory-mutating phases (load / KV alloc / capture) hold a
            # per-device LOCK_SH and its profiling measurement holds LOCK_EX.
            # The wrapper must be installed here (in the engine-core child):
            # phase boundaries live inside EngineCore.__init__, invisible to the
            # orchestrator. See vllm_omni.engine.stage_phase_lock.
            if omni_parallel_stage_init:
                _install_phase_locks(kwargs, local_dp_rank)

            engine_core = StageEngineCoreProc(
                *args,
                engine_index=dp_rank,
                **kwargs,
            )

            # Each subprocess corresponds to exactly one omni replica with
            # its own OmniMasterServer allocation, so the heartbeat client
            # runs unconditionally — there is no dp_rank-based gating.
            if omni_coordinator_address is not None:
                if omni_stage_id is None:
                    raise ValueError("omni_stage_id must be provided when omni_coordinator_address is set")
                addresses: EngineZmqAddresses = engine_core.addresses
                if not addresses.inputs or not addresses.outputs:
                    raise RuntimeError(
                        "EngineCore handshake did not populate input/output addresses; "
                        "cannot start OmniCoordClientForStage"
                    )
                scheduler = getattr(engine_core, "scheduler", None)
                if scheduler is None:
                    raise RuntimeError("EngineCore scheduler is not initialized")
                coord_client = create_stage_coord_client(
                    coord_zmq_addr=omni_coordinator_address,
                    input_addr=addresses.inputs[0],
                    output_addr=addresses.outputs[0],
                    stage_id=int(omni_stage_id),
                    queue_length_getter=scheduler.get_num_unfinished_requests,
                )

            def wakeup_engine() -> None:
                engine_core.input_queue.put_nowait((EngineCoreRequestType.WAKEUP, None))

            signal_callback = SignalCallback(wakeup_engine)

            def signal_handler(signum: int, frame: Any) -> None:
                engine_core.shutdown_state = EngineShutdownState.REQUESTED
                signal_callback.trigger()
                raise SystemExit(_signal_exit_code(signum))

            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)

            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("StageEngineCoreProc exiting.")
            raise
        except Exception:
            if engine_core is None:
                logger.exception("StageEngineCoreProc failed to start.")
            else:
                logger.exception("StageEngineCoreProc encountered a fatal error.")
                engine_core._send_engine_dead()
            raise
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            if signal_callback is not None:
                signal_callback.stop()
            if coord_client is not None:
                with contextlib.suppress(RuntimeError):
                    coord_client.close()
            if engine_core is not None:
                engine_core.shutdown()
