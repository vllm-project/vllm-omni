# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import inspect
import os
import queue
import threading
import time
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import PIL.Image
import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.engine.exceptions import EngineDeadError
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_omni.diffusion.data import (
    DiffusionOutput,
    DiffusionRequestAbortedError,
    OmniDiffusionConfig,
    uses_diffusers_adapter,
)
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode, is_scheduler_paged_kv_mode
from vllm_omni.diffusion.diffusion_kv.initialization import initialize_diffusion_kv_control_plane
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.io_support import (
    get_dummy_run_num_frames,
    get_dummy_run_num_image_inputs,
    image_color_format,
    supports_audio_output,
    supports_multimodal_input,
)
from vllm_omni.diffusion.output_formatter import (
    format_diffusion_outputs,
    format_empty_diffusion_outputs,
    normalize_diffusion_postprocess_output,
)
from vllm_omni.diffusion.registry import (
    DiffusionModelRegistry,
    get_diffusion_post_process_func,
    get_diffusion_pre_process_func,
)
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
from vllm_omni.diffusion.sched import BaseScheduler, RequestScheduler, StepScheduler
from vllm_omni.diffusion.sched.interface import DiffusionRequestStatus
from vllm_omni.diffusion.worker.utils import BaseRunnerOutput, BatchRunnerOutput, RunnerOutput
from vllm_omni.errors import client_error_from_metadata, is_client_error_status
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt
from vllm_omni.metrics.utils import (
    diffusion_scheduler_waiting_metrics,
    extract_diffusion_denoise_ms,
    extract_diffusion_vae_decode_ms,
)

if TYPE_CHECKING:
    from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)

_ASYNC_OUTPUT_TIMEOUT_ENV = "VLLM_OMNI_ASYNC_OUTPUT_TIMEOUT"
_ASYNC_OUTPUT_TIMEOUT_DEFAULT = 600.0  # seconds


def _async_output_timeout() -> float:
    """Seconds to wait for one step's background D2H/SHM copy.

    The copy itself finishes in milliseconds, but it is queued behind the GPU
    work for that step, so the wall-clock wait tracks step time — a single-GPU
    box legitimately runs tens of seconds per step on large shapes. A tight
    bound therefore does not catch a hung engine (worker death and a dead
    result pump are surfaced by the worker monitor and ``check_health``); it
    only aborts renders that are still making progress, throwing away the
    denoise that already completed. The default matches
    ``_DLO_DP_WAVE_TIMEOUT_S`` in the same subsystem.

    Resolved here rather than at import so a malformed value degrades to the
    default instead of raising on the request path: this runs inside
    ``step_streaming``/``add_req_and_wait_for_response``, where a typo in the
    environment must not start failing generations.
    """
    raw = os.environ.get(_ASYNC_OUTPUT_TIMEOUT_ENV)
    if raw is None:
        return _ASYNC_OUTPUT_TIMEOUT_DEFAULT
    try:
        timeout = float(raw)
    except ValueError:
        logger.warning_once(
            "Ignoring %s=%r: not a number. Using the default %.1fs.",
            _ASYNC_OUTPUT_TIMEOUT_ENV,
            raw,
            _ASYNC_OUTPUT_TIMEOUT_DEFAULT,
        )
        return _ASYNC_OUTPUT_TIMEOUT_DEFAULT
    if timeout <= 0:
        logger.warning_once(
            "Ignoring %s=%r: must be positive. Using the default %.1fs.",
            _ASYNC_OUTPUT_TIMEOUT_ENV,
            raw,
            _ASYNC_OUTPUT_TIMEOUT_DEFAULT,
        )
        return _ASYNC_OUTPUT_TIMEOUT_DEFAULT
    return timeout


__all__ = [
    "DiffusionEngine",
    "DiffusionExecutionMode",
    "_RpcTask",
    "_move_tensor_tree_to_cpu",
    "get_dummy_run_num_frames",
    "image_color_format",
    "supports_audio_output",
    "supports_multimodal_input",
]


def _func_accepts_parameter(func: object | None, parameter_name: str) -> bool:
    if func is None:
        return False
    parameters = inspect.signature(func).parameters
    return parameter_name in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def _resolve_custom_pipeline_cls(custom_pipeline_args: dict[str, Any] | None) -> type | None:
    if custom_pipeline_args is None:
        return None

    try:
        pipeline_cls = custom_pipeline_args["pipeline_class"]
    except KeyError as exc:
        raise ValueError("custom_pipeline_args must include 'pipeline_class'.") from exc

    if isinstance(pipeline_cls, type):
        return pipeline_cls
    if isinstance(pipeline_cls, str):
        try:
            return resolve_obj_by_qualname(pipeline_cls)
        except (AttributeError, ImportError, ValueError) as exc:
            raise ValueError(f"Failed to resolve custom diffusion pipeline class {pipeline_cls!r}.") from exc
    raise TypeError(
        f"custom_pipeline_args['pipeline_class'] must be a qualified name string or a class, "
        f"got {type(pipeline_cls).__name__}"
    )


def supports_request_batch(od_config: OmniDiffusionConfig) -> bool:
    model_cls = _resolve_custom_pipeline_cls(getattr(od_config, "custom_pipeline_args", None))
    if model_cls is not None:
        return bool(getattr(model_cls, "supports_request_batch", False))

    if uses_diffusers_adapter(od_config):
        model_cls = DiffusionModelRegistry._try_load_model_cls("DiffusersAdapterPipeline")
        return bool(getattr(model_cls, "supports_request_batch", False))

    model_cls = DiffusionModelRegistry._try_load_model_cls(getattr(od_config, "model_class_name", None))
    if model_cls is None:
        return False
    return bool(getattr(model_cls, "supports_request_batch", False))


def _max_num_seqs(od_config: OmniDiffusionConfig) -> int:
    try:
        return max(1, int(getattr(od_config, "max_num_seqs", 1)))
    except (TypeError, ValueError):
        return 1


def _uses_dlo_dp_concurrency(od_config: OmniDiffusionConfig) -> bool:
    parallel_config = getattr(od_config, "parallel_config", None)
    dp_size = getattr(parallel_config, "data_parallel_size", 1)
    return (
        dp_size > 1
        and getattr(od_config, "enable_distributed_layerwise_offload", False)
        and getattr(od_config, "dlo_use_allgather", True)
    )


def _move_tensor_tree_to_cpu(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.cpu() if value.device.type != "cpu" else value
    if isinstance(value, dict):
        return {key: _move_tensor_tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tensor_tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensor_tree_to_cpu(item) for item in value)
    return value


@dataclass
class _RpcTask:
    """A pending collective_rpc invocation queued for the busy loop."""

    method: str
    args: tuple
    kwargs: dict | None
    deadline: float | None
    unique_reply_rank: int | None
    future: concurrent.futures.Future = field(default_factory=concurrent.futures.Future)


class DiffusionExecutionMode(str, Enum):
    REQUEST_BATCH = "request_batch"
    STEP_BATCH = "step_batch"


class DiffusionEngine:
    """The diffusion engine for vLLM-Omni diffusion models."""

    #: Import path of the model runner this engine's workers should build, or
    #: ``None`` for the platform default. Subclasses declare their runner here
    #: (e.g. the AR-Diffusion engine), the worker resolves it through
    #: :meth:`resolve_engine_class` — so engine->runner routing lives on the
    #: engine class itself, and ``od_config.diffusion_model_runner_cls``
    #: remains a pure explicit user override (never mutated by engines).
    default_diffusion_model_runner_cls: str | None = None

    # Class-level default so tests using object.__new__ (without __init__)
    # don't hit AttributeError when _busy_loop accesses self.dp_concurrent.
    dp_concurrent: bool = False

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        scheduler: BaseScheduler | None = None,
    ):
        """Initialize the diffusion engine.

        Args:
            od_config: The configuration for the diffusion engine.
            scheduler: Optional scheduler override for tests or custom engine
                integrations. When omitted, the engine selects a scheduler
                from the resolved execution mode.
        """
        self.od_config = od_config
        # Set after the paged-KV profile request has gone through model-owned
        # preprocessing. Real requests are admitted only within this measured
        # activation envelope: (max execution sequences, max seq_len,
        # max target_len).
        self._diffusion_kv_profile_limits: tuple[int, int, int] | None = None

        self._init_process_hooks(od_config)
        self.execution_mode = self._resolve_execution_mode(od_config)
        self._init_executor(od_config)
        try:
            profile_requests = self._prepare_diffusion_kv_profile_requests()
            kv_control_plane = initialize_diffusion_kv_control_plane(
                self.executor,
                od_config,
                profile_requests=profile_requests,
            )
            if kv_control_plane is None:
                self._init_scheduler(od_config, scheduler)
            else:
                (
                    kv_cache_config,
                    scheduler_block_size,
                    hash_block_size,
                    kv_vllm_config,
                ) = kv_control_plane
                self._init_scheduler(
                    od_config,
                    scheduler,
                    kv_cache_config,
                    scheduler_block_size=scheduler_block_size,
                    hash_block_size=hash_block_size,
                    kv_vllm_config=kv_vllm_config,
                )
            self._init_runtime_state()
            self._init_execute_fn()
            self._log_execution_mode(od_config)
        except Exception:
            # close() cannot be used because runtime synchronization state may not exist yet.
            scheduler_to_close = getattr(self, "scheduler", None)
            if scheduler_to_close is not None:
                try:
                    scheduler_to_close.close()
                except Exception:
                    logger.exception("Failed to close Scheduler after DiffusionEngine initialization failed")
            try:
                self.executor.shutdown()
            except Exception:
                logger.exception("Failed to shut down Executor after DiffusionEngine initialization failed")
            raise

    def _init_process_hooks(self, od_config: OmniDiffusionConfig) -> None:
        self.post_process_func = get_diffusion_post_process_func(od_config)
        self.pre_process_func = get_diffusion_pre_process_func(od_config)
        # Cache whether the model-specific postprocess accepts request-level
        # sampling params so step() can support both legacy and extended hooks.
        self._post_process_accepts_sampling_params = _func_accepts_parameter(self.post_process_func, "sampling_params")

    def _resolve_execution_mode(self, od_config: OmniDiffusionConfig) -> DiffusionExecutionMode:
        self.step_execution = bool(getattr(od_config, "step_execution", False))
        if od_config.streaming_output and not self.step_execution:
            logger.warning("streaming_output=True requires step_execution=True; enabling step execution.")
            od_config.step_execution = True
            self.step_execution = True

        if self.step_execution:
            self.supports_request_batch = False
            return DiffusionExecutionMode.STEP_BATCH

        self.supports_request_batch = supports_request_batch(od_config)
        if not self.supports_request_batch and _max_num_seqs(od_config) > 1 and not _uses_dlo_dp_concurrency(od_config):
            raise ValueError(
                f"{getattr(od_config, 'model_class_name', None)!r} does not support request-level batching. "
                "Use max_num_seqs=1 for serial request execution, or choose a pipeline with "
                "supports_request_batch=True."
            )
        return DiffusionExecutionMode.REQUEST_BATCH

    def _init_executor(self, od_config: OmniDiffusionConfig) -> None:
        executor_class = DiffusionExecutor.get_class(od_config)
        self.executor = executor_class(od_config)

    def _init_scheduler(
        self,
        od_config: OmniDiffusionConfig,
        scheduler: BaseScheduler | None = None,
        kv_cache_config: KVCacheConfig | None = None,
        *,
        scheduler_block_size: int | None = None,
        hash_block_size: int | None = None,
        kv_vllm_config: VllmConfig | None = None,
    ) -> None:
        if scheduler is not None:
            self.scheduler = scheduler
        elif self.execution_mode == DiffusionExecutionMode.STEP_BATCH:
            self.scheduler = StepScheduler()
        else:
            self.scheduler = RequestScheduler()
        if kv_cache_config is None:
            self.scheduler.initialize(od_config)
        else:
            self.scheduler.initialize(
                od_config,
                kv_cache_config=kv_cache_config,
                scheduler_block_size=scheduler_block_size,
                hash_block_size=hash_block_size,
                kv_vllm_config=kv_vllm_config,
            )

    def _init_runtime_state(self) -> None:
        # DP multi-concurrency: allow batching dp_size requests so each
        # worker processes a different request in parallel.  Only enabled
        # for distributed layerwise offload (which shards weights and
        # needs all ranks active simultaneously).  Ordinary DP with a
        # non-batch pipeline should not schedule multiple requests.
        dp_size = getattr(getattr(self.od_config, "parallel_config", None), "data_parallel_size", 1)
        if _uses_dlo_dp_concurrency(self.od_config):
            self.scheduler.max_num_running_reqs = dp_size
            self.dp_concurrent = True
            logger.info(
                "dp_concurrent: max_num_running_reqs=%d, batch_wait=%sms",
                dp_size,
                self.od_config.request_batch_max_wait_ms,
            )
        else:
            self.dp_concurrent = False
        self.main_loop: asyncio.AbstractEventLoop | None = None
        self.stop_event: threading.Event | None = None
        self.worker_thread: threading.Thread | None = None
        self._loop_started = False
        self._init_lock = asyncio.Lock()
        # _rpc_lock is retained solely as the underlying lock for self._cv,
        # which is used to signal the busy loop. Worker-call serialization is
        # now handled structurally by routing all executor calls through the
        # busy loop rather than via mutual exclusion.
        self._rpc_lock = threading.RLock()
        self._cv = threading.Condition(self._rpc_lock)
        self._out_streams: dict[str, asyncio.Queue[DiffusionOutput]] = {}
        self._closed = False
        self._shutdown_complete = False
        self.abort_queue: queue.Queue[str] = queue.Queue()
        self._rpc_queue: queue.Queue[_RpcTask] = queue.Queue()
        # Copied onto the existing output metrics payload so queue monitoring
        # reuses the normal diffusion result path without additional IPC.
        self._scheduler_num_waiting_reqs = 0

    def _init_execute_fn(self) -> None:
        if self.execution_mode == DiffusionExecutionMode.STEP_BATCH:
            self.execute_fn = self.executor.execute_step
        else:
            self.execute_fn = self.executor.execute_batch

    def _log_execution_mode(self, od_config: OmniDiffusionConfig) -> None:
        if self.execution_mode == DiffusionExecutionMode.REQUEST_BATCH:
            logger.info(
                "[RequestBatch] engine init max_num_seqs=%s max_wait_ms=%s",
                getattr(od_config, "max_num_seqs", None),
                getattr(od_config, "request_batch_max_wait_ms", None),
            )

    async def _check_and_start_background_loop(self):
        if self._closed:
            raise RuntimeError("DiffusionEngine is closed.")
        if self._loop_started:
            return

        async with self._init_lock:
            # double check, in case of lock queue issue
            if self._closed:
                raise RuntimeError("DiffusionEngine is closed.")
            if self._loop_started:
                return

            self.main_loop = asyncio.get_running_loop()
            self.stop_event = threading.Event()
            self.worker_thread = threading.Thread(target=self._busy_loop)
            self.worker_thread.start()
            self._loop_started = True

    async def step_streaming(self, request: OmniDiffusionRequest) -> AsyncGenerator[list[OmniRequestOutput], None]:
        await self._check_and_start_background_loop()

        diffusion_engine_start_time = time.perf_counter()

        preprocess_time = 0.0
        has_preprocessor = getattr(self, "pre_process_func", None) is not None
        preprocess_start_time = time.perf_counter() if has_preprocessor else None
        request = self._prepare_request_for_admission(request)
        if preprocess_start_time is not None:
            preprocess_time = time.perf_counter() - preprocess_start_time
            logger.debug("Pre-processing completed in %.4f seconds", preprocess_time)

        exec_start_time = time.perf_counter()
        request_id = self._add_prepared_request(request)
        generator = self.get_output_stream(request_id)
        async for output in generator:
            exec_total_time = time.perf_counter() - exec_start_time
            # Async mode: wait for background D2H/SHM to complete.
            if output.async_output_id:
                fut = self.executor.wait_output_ready(output.async_output_id)
                timeout = _async_output_timeout()
                try:
                    output = await asyncio.wait_for(asyncio.wrap_future(fut), timeout=timeout)
                except (TimeoutError, asyncio.TimeoutError):
                    describe = getattr(self.executor, "describe_pending_state", None)
                    logger.error(
                        "Timed out after %.1fs waiting for async output; set %s to a larger value "
                        "to allow slower steps. Executor state: %s",
                        timeout,
                        _ASYNC_OUTPUT_TIMEOUT_ENV,
                        describe(output.async_output_id) if describe else "unavailable",
                    )
                    raise
            postprocess_start_time = time.perf_counter()
            scheduler_metrics = diffusion_scheduler_waiting_metrics(getattr(self, "_scheduler_num_waiting_reqs", 0))
            try:
                formatted_outputs = self.postprocess_output(request, output)
            except Exception as exc:
                # Preserve the latest scheduler snapshot across terminal
                # abort/error paths, which do not produce formatted outputs.
                setattr(exc, "diffusion_metrics", scheduler_metrics)
                raise
            postprocess_time = time.perf_counter() - postprocess_start_time
            step_total_ms = (time.perf_counter() - diffusion_engine_start_time) * 1000
            logger.debug(
                "DiffusionEngine.step_streaming breakdown: preprocess=%.2f ms, "
                "add_req_and_wait=%.2f ms, postprocess=%.2f ms, total=%.2f ms",
                preprocess_time * 1000,
                exec_total_time * 1000,
                postprocess_time * 1000,
                step_total_ms,
            )
            for request_output in formatted_outputs:
                metrics_update = {
                    "preprocess_time_ms": preprocess_time * 1000,
                    "diffusion_engine_exec_time_ms": exec_total_time * 1000,
                    "postprocess_time_ms": postprocess_time * 1000,
                    **scheduler_metrics,
                }
                if request.scheduler_queue_wait_ms is not None:
                    metrics_update["scheduler_queue_wait_ms"] = request.scheduler_queue_wait_ms
                vae_decode_ms = extract_diffusion_vae_decode_ms(output)
                if vae_decode_ms is not None:
                    metrics_update["vae_decode_time_ms"] = vae_decode_ms
                forward_ms = extract_diffusion_denoise_ms(output)
                if forward_ms is not None:
                    metrics_update["forward_time_ms"] = forward_ms
                kv_recv_ms = getattr(output, "kv_recv_ms", 0.0)
                if kv_recv_ms > 0:
                    metrics_update["kv_recv_time_ms"] = kv_recv_ms
                request_output.metrics.update(metrics_update)
            yield formatted_outputs

    async def step(self, request: OmniDiffusionRequest) -> list[OmniRequestOutput]:
        """Deprecated compatibility wrapper over ``step_streaming()``.

        Use ``step_streaming()`` for new callers. This method drains the
        unified output stream and returns only the final output batch, matching
        the historical non-streaming ``step()`` behavior.
        """
        logger.warning_once(
            "DiffusionEngine.step() is deprecated; use step_streaming() and consume the final output batch instead."
        )
        final_output: list[OmniRequestOutput] | None = None
        async for output in self.step_streaming(request):
            final_output = output
        return final_output or []

    def postprocess_output(
        self,
        request: OmniDiffusionRequest,
        output: DiffusionOutput,
    ) -> list[OmniRequestOutput]:
        """Convert a DiffusionOutput to a list of OmniRequestOutput."""
        if output.aborted:
            raise DiffusionRequestAbortedError(output.abort_message or "Diffusion request aborted.")
        if output.error:
            if is_client_error_status(output.error_status_code):
                raise client_error_from_metadata(
                    output.error,
                    status_code=output.error_status_code,
                    error_type=output.error_type,
                )
            raise RuntimeError(output.error)
        logger.debug("Generation completed successfully.")

        if output.output is None:
            logger.warning("Output is None, returning empty OmniRequestOutput")
            return format_empty_diffusion_outputs(request, finished=output.finished)

        # When CPU offload is enabled, move output to CPU before
        # post-processing to avoid device OOM — model weights may still
        # reside on the device and leave no headroom for intermediates.
        output_data = output.output
        if self.od_config.enable_cpu_offload:
            output_data = _move_tensor_tree_to_cpu(output_data)

        if self.post_process_func is not None:
            # Some video pipelines need request-level controls during
            # postprocess (for example worker-side frame interpolation).
            postprocess_kwargs: dict[str, object] = {}
            if self._post_process_accepts_sampling_params:
                postprocess_kwargs["sampling_params"] = request.sampling_params
            outputs = self.post_process_func(output_data, **postprocess_kwargs)
        else:
            outputs = output_data

        postprocess_output = normalize_diffusion_postprocess_output(outputs)

        return format_diffusion_outputs(
            request=request,
            od_config=self.od_config,
            diffusion_output=output,
            output_data=output_data,
            postprocess_output=postprocess_output,
        )

    def _busy_loop(self):
        while not self.stop_event.is_set():
            self._process_aborts_queue()
            self._process_rpc_queue()

            with self._cv:
                while (
                    not self.scheduler.has_requests()
                    and self._rpc_queue.empty()
                    and self.abort_queue.empty()
                    and not self.stop_event.is_set()
                ):
                    self._cv.wait(timeout=1.0)

                if self.stop_event.is_set():
                    break

                if not self.scheduler.has_requests():
                    # Only RPC / abort work pending; loop back to drain it.
                    continue

                self._wait_for_admission_if_needed_locked()

                sched_output = self.scheduler.schedule()
                self._scheduler_num_waiting_reqs = max(int(sched_output.num_waiting_reqs), 0)

            if sched_output.is_empty:
                self._emit_finished_outputs(sched_output.finished_req_ids, None)
                continue

            try:
                runner_output: BaseRunnerOutput = self.execute_fn(sched_output)  # pyright: ignore[reportAssignmentType]
            except Exception as exc:
                logger.error(
                    "Execution failed for diffusion requests %s", sched_output.scheduled_request_ids, exc_info=True
                )
                runner_output = BatchRunnerOutput.from_list(
                    [
                        RunnerOutput(
                            request_id=request_id,
                            step_index=None,
                            finished=True,
                            result=DiffusionOutput.from_exception(exc),
                        )
                        for request_id in sched_output.scheduled_request_ids
                    ]
                )

            self._process_aborts_queue()
            self._process_rpc_queue()
            finished_req_ids = self.scheduler.update_from_output(sched_output, runner_output)
            self._emit_outputs(finished_req_ids, sched_output.scheduled_request_ids, runner_output)

        # Engine is stopping: fail any RPCs still queued so callers don't hang.
        self._fail_pending_rpcs(RuntimeError("DiffusionEngine is shutting down."))

    def _wait_for_admission_if_needed_locked(self) -> None:
        """Apply scheduler admission policy while holding the engine condition.

        Caller must hold ``self._cv``.
        """
        start = time.monotonic()
        decision = self.scheduler.get_admission_wait_decision(
            now=start,
            dp_concurrent=self.dp_concurrent,
        )
        if not decision.should_wait:
            return

        last_waiting = -1
        stable_since = start

        while not self.stop_event.is_set():
            waiting = self.scheduler.num_waiting_requests()
            now = time.monotonic()

            if waiting > last_waiting:
                stable_since = now
                last_waiting = waiting

            if self.scheduler.should_end_admission_wait(
                decision,
                now=now,
                stable_since=stable_since,
            ):
                break

            remaining = decision.deadline - now if decision.deadline is not None else 0.002
            self._cv.wait(timeout=min(max(remaining, 0.0), 0.002))

        waited_ms = (time.monotonic() - start) * 1000.0
        final_waiting = self.scheduler.num_waiting_requests()
        if final_waiting > 0:
            logger.info(
                "[RequestBatch] admission wait done waiting=%d max_batch=%d waited_ms=%.1f",
                final_waiting,
                decision.max_batch,
                waited_ms,
            )

    def _process_rpc_queue(self) -> None:
        """Execute pending collective_rpc tasks from the busy-loop thread.

        Running these here means executor calls are naturally serialized
        against execute_fn() without any mutual-exclusion locking.
        """
        while True:
            try:
                task = self._rpc_queue.get_nowait()
            except queue.Empty:
                return

            fut = task.future
            if fut.cancelled() or fut.done():
                continue

            remaining: float | None = None
            if task.deadline is not None:
                remaining = task.deadline - time.monotonic()
                if remaining <= 0:
                    if not fut.done():
                        fut.set_exception(TimeoutError(f"RPC call to {task.method} timed out before execution."))
                    continue

            try:
                result = self.executor.collective_rpc(
                    method=task.method,
                    timeout=remaining,
                    args=task.args,
                    kwargs=task.kwargs,
                    unique_reply_rank=task.unique_reply_rank,
                )
            except BaseException as exc:  # noqa: BLE001 - propagate to caller
                # The future may have been cancelled (e.g. by a sync timeout
                # or asyncio cancellation) while the executor call was
                # running. Setting state on a cancelled/done future raises
                # InvalidStateError, which would kill the busy loop.
                if not fut.done():
                    fut.set_exception(exc)
            else:
                if not fut.done():
                    fut.set_result(result)

    def _fail_pending_rpcs(self, exc: BaseException) -> None:
        while True:
            try:
                task = self._rpc_queue.get_nowait()
            except queue.Empty:
                return
            if not task.future.done():
                task.future.set_exception(exc)

    def _remove_diffusion_kv_requests(self, request_ids: Iterable[str]) -> None:
        """Clear terminal Worker rows while Scheduler owns the allocations."""

        od_config = getattr(self, "od_config", None)
        if od_config is None or not is_scheduler_paged_kv_mode(
            getattr(od_config, "diffusion_kv_mode", DiffusionKVCacheMode.DENSE_LEGACY)
        ):
            return
        unique_request_ids = list(dict.fromkeys(request_ids))
        if unique_request_ids:
            self.executor.remove_diffusion_kv_requests(unique_request_ids)

    def _emit_finished_outputs(
        self,
        finished_ids: set[str],
        runner_output: BaseRunnerOutput | None = None,
        missing_result_error: str = "Diffusion execution finished without a final output",
    ) -> None:
        self._remove_diffusion_kv_requests(finished_ids)
        for rid in finished_ids:
            if runner_output is not None:
                _output = runner_output.get_request_output(rid)
            else:
                _output = None
            out = self._finalize_finished_request(rid, _output, missing_result_error)
            self._put_output(rid, out)

    def _emit_outputs(
        self,
        finished_ids: set[str],
        scheduled_request_ids: list[str],
        runner_output: BaseRunnerOutput,
    ) -> None:
        """Emit output chunks for every request through the unified output stream."""
        if self.execution_mode != DiffusionExecutionMode.STEP_BATCH:
            self._emit_finished_outputs(finished_ids, runner_output)
            return

        self._remove_diffusion_kv_requests(finished_ids)

        delivered_finished_req_ids: set[str] = set()

        # finished_ids may have some requests that are not scheduler in this round.
        # First handle this-round requests.
        for request_id in scheduled_request_ids:
            req_output = runner_output.get_request_output(request_id)
            if request_id in finished_ids:
                # This entire request is finished (this is the last chunk)
                out = self._finalize_finished_request(
                    request_id,
                    req_output,
                    missing_result_error="Diffusion step execution finished without a final output.",
                )
                self._put_output(request_id, out)
                delivered_finished_req_ids.add(request_id)
            elif req_output is not None and req_output.result is not None:
                # This is a non-terminal chunk. It is not in scheduler's
                # finished_ids, but it still belongs on the request stream.
                self._put_output(request_id, req_output.result)

        # Then handle other requests that are finished in this round.
        for request_id in finished_ids - delivered_finished_req_ids:
            out = self._finalize_finished_request(
                request_id,
                missing_result_error="Diffusion step request finished without execution output.",
            )
            self._put_output(request_id, out)

    def _has_output_stream(self, request_id: str) -> bool:
        with self._cv:
            return request_id in self._out_streams

    @staticmethod
    def resolve_engine_class(config: OmniDiffusionConfig) -> type[DiffusionEngine]:
        """Resolve the engine class selected by ``config.engine_backend``.

        Mirrors ``DiffusionExecutor.get_class``: accepts ``"default"``, a
        ``DiffusionEngine`` subclass, or an import-path string (e.g. a deploy
        config's ``engine_backend``). Kept separate from :meth:`make_engine` so the
        selection is testable without constructing an engine (which runs a dummy forward).

        Args:
            config: The configuration for the diffusion engine.

        Returns:
            The ``DiffusionEngine`` (sub)class to instantiate.
        """
        backend = getattr(config, "engine_backend", "default") or "default"

        if isinstance(backend, type):
            if not issubclass(backend, DiffusionEngine):
                raise TypeError(f"engine_backend must be a DiffusionEngine subclass. Got {backend}.")
            return backend
        if backend == "default":
            return DiffusionEngine
        if isinstance(backend, str):
            try:
                engine_class = resolve_obj_by_qualname(backend)
            except (ImportError, ValueError) as e:
                raise ValueError(
                    f"Failed to load engine_backend '{backend}'. Ensure it is a valid python path. Error: {e}"
                ) from e
            if not issubclass(engine_class, DiffusionEngine):
                raise TypeError(f"engine_backend must resolve to a DiffusionEngine subclass. Got {engine_class}.")
            return engine_class
        raise ValueError(f"Unknown engine_backend: {backend!r}")

    @staticmethod
    def make_engine(
        config: OmniDiffusionConfig,
        scheduler: BaseScheduler | None = None,
    ) -> DiffusionEngine:
        """Factory method to create the engine selected by ``config.engine_backend``.

        Args:
            config: The configuration for the diffusion engine.
            scheduler: Optional scheduler override. When omitted, the selected
                engine chooses the scheduler from its execution mode.

        Returns:
            An instance of the resolved ``DiffusionEngine`` (sub)class.
        """
        engine_class = DiffusionEngine.resolve_engine_class(config)
        engine = engine_class(config, scheduler=scheduler)
        engine.run_startup_warmup()
        return engine

    def _prepare_request_for_admission(self, request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        """Run model-owned preprocessing once, before entering Engine locks."""

        pre_process_func = getattr(self, "pre_process_func", None)
        if pre_process_func is not None:
            request = pre_process_func(request)
        self._validate_diffusion_kv_profile_limits(request)
        return request

    def _validate_diffusion_kv_profile_limits(self, request: OmniDiffusionRequest) -> None:
        """Keep admitted paged-KV requests within the profiled activation shape."""

        profile_limits = getattr(self, "_diffusion_kv_profile_limits", None)
        if profile_limits is None:
            return
        kv_requests = request.diffusion_kv_requests
        if not kv_requests:
            return

        request_limits = (
            len(kv_requests),
            max(kv_request.seq_len for kv_request in kv_requests),
            max(kv_request.target_len for kv_request in kv_requests),
        )
        if any(actual > profiled for actual, profiled in zip(request_limits, profile_limits, strict=True)):
            request_sequences, request_seq_len, request_target_len = request_limits
            profile_sequences, profile_seq_len, profile_target_len = profile_limits
            raise ValueError(
                f"Diffusion KV request {request.request_id!r} exceeds the startup memory-profile envelope: "
                f"sequences={request_sequences} (profiled={profile_sequences}), "
                f"max_seq_len={request_seq_len} (profiled={profile_seq_len}), "
                f"max_target_len={request_target_len} (profiled={profile_target_len}). "
                "Reduce the request shape or extend the model's paged-KV profile recipe."
            )

    def _add_prepared_request(self, request: OmniDiffusionRequest) -> str:
        """Admit a request whose model-owned preprocessing is complete."""

        with self._cv:
            if self._closed:
                raise RuntimeError("DiffusionEngine is closed.")
            queue: asyncio.Queue[DiffusionOutput] = asyncio.Queue()
            request_id = self.scheduler.add_request(request)
            self._out_streams[request_id] = queue
            self._cv.notify_all()

        return request_id

    def add_request(self, request: OmniDiffusionRequest) -> str:
        request = self._prepare_request_for_admission(request)
        return self._add_prepared_request(request)

    async def get_output_stream(self, request_id: str) -> AsyncGenerator[DiffusionOutput, None]:
        with self._cv:
            queue = self._out_streams.get(request_id)
        if queue is None:
            raise RuntimeError(f"Request {request_id} not found in output queue.")
        try:
            while True:
                output: DiffusionOutput = await queue.get()
                yield output
                if output.finished:
                    break
        except Exception as e:
            logger.error(f"Wait for response failed: {e}")
            raise
        finally:
            with self._cv:
                if self._out_streams.get(request_id) is queue:
                    self._out_streams.pop(request_id, None)

    def async_add_req_and_stream_response(self, request: OmniDiffusionRequest) -> AsyncGenerator[DiffusionOutput, None]:
        request_id = self.add_request(request)
        return self.get_output_stream(request_id)

    async def async_add_req_and_wait_for_response(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        """Deprecated compatibility wrapper over ``async_add_req_and_stream_response()``.

        Use ``async_add_req_and_stream_response()`` for new callers. This
        method drains the unified output stream and returns only the final
        ``DiffusionOutput``, matching the historical non-streaming behavior.
        """
        logger.warning_once(
            "DiffusionEngine.async_add_req_and_wait_for_response() is deprecated; "
            "use async_add_req_and_stream_response() and consume the final output instead."
        )
        final_output: DiffusionOutput | None = None
        async for output in self.async_add_req_and_stream_response(request):
            final_output = output
        if final_output is None:
            raise RuntimeError("Diffusion execution completed without an output.")
        return final_output

    def add_req_and_wait_for_response(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        request = self._prepare_request_for_admission(request)
        with self._rpc_lock:
            if self._closed:
                raise RuntimeError("DiffusionEngine is closed.")
            target_request_id = self.scheduler.add_request(request)

            # keep scheduling and executing until the target request is finished
            while True:
                self._process_aborts_queue()
                sched_output = self.scheduler.schedule()
                if sched_output.is_empty:
                    if target_request_id in sched_output.finished_req_ids:
                        self._remove_diffusion_kv_requests([target_request_id])
                        return self._finalize_finished_request(target_request_id)
                    if not self.scheduler.has_requests():
                        raise RuntimeError("Diffusion scheduler has no runnable requests.")
                    continue

                # NOTE: add_req_and_wait_for_response() is synchronous, will be only called
                # within _dummy_run, only one request will be scheduled
                request_id = sched_output.scheduled_request_ids[0]
                try:
                    runner_output: BaseRunnerOutput = self.execute_fn(sched_output)  # pyright: ignore[reportAssignmentType]
                except EngineDeadError:
                    raise
                except Exception as exc:
                    logger.error("Execution failed for diffusion request %s", request_id, exc_info=True)
                    runner_output = RunnerOutput(
                        request_id=request_id,
                        step_index=None,
                        finished=True,
                        result=DiffusionOutput.from_exception(exc),
                    )

                self._process_aborts_queue()

                finished_req_ids = self.scheduler.update_from_output(sched_output, runner_output)

                # sync func should receive one result
                if not isinstance(runner_output, RunnerOutput) and not len(runner_output) == 1:
                    raise ValueError("Sync func should receive one result at one time")
                if target_request_id in finished_req_ids:
                    self._remove_diffusion_kv_requests([target_request_id])
                    req_output = runner_output.get_request_output(target_request_id)
                    output = self._finalize_finished_request(
                        target_request_id,
                        runner_output=req_output,
                        missing_result_error="Diffusion execution finished without a final output.",
                    )
                    if output.async_output_id:
                        fut = self.executor.wait_output_ready(output.async_output_id)
                        output = fut.result(timeout=_async_output_timeout())
                    return output

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        """Start or stop profiling on all diffusion workers.

        Args:
            is_start: True to start profiling, False to stop.
            profile_prefix: Optional prefix for trace filename.
        """
        if is_start:
            if profile_prefix is None:
                profile_prefix = f"diffusion_{int(time.time())}"
            logger.info(f"Starting diffusion profiling with prefix: {profile_prefix}")
        else:
            logger.info("Stopping diffusion profiling...")

        try:
            self.collective_rpc(method="profile", args=(is_start, profile_prefix))
        except Exception as e:
            action = "start" if is_start else "stop"
            logger.error(f"Failed to {action} profiling on workers", exc_info=True)
            if is_start:
                raise RuntimeError(f"Could not {action} profiler: {e}") from e

    def run_startup_warmup(self) -> None:
        dlo_use_allgather = getattr(self.od_config, "dlo_use_allgather", True)
        # Skip dummy run when AllGather is used with more than 1 rank,
        # because the dummy run sends only 1 request but AllGather requires
        # all ranks to participate simultaneously.  This covers both DP > 1
        # and SP > 1 (where dp_size is derived from sp_size in OffloadConfig).
        pc = getattr(self.od_config, "parallel_config", None)
        dp_size = getattr(pc, "data_parallel_size", 1) if pc else 1
        sp_size = getattr(pc, "sequence_parallel_size", 1) if pc else 1
        effective_shard_size = max(dp_size, sp_size)
        skip_dummy = (
            getattr(self.od_config, "enable_distributed_layerwise_offload", False)
            and dlo_use_allgather
            and effective_shard_size > 1
        )
        if skip_dummy:
            logger.info(
                "Skipping dummy run (dist_offload with AllGather, dp_size=%d, sp_size=%d)",
                dp_size,
                sp_size,
            )
            return
        try:
            self._dummy_run()
        except Exception as e:
            logger.error(f"Dummy run failed: {e}")
            self.close()
            raise e

    def _make_dummy_request(
        self,
        *,
        height: int,
        width: int,
        guidance_scale: float,
        num_image_inputs: int = 1,
    ) -> OmniDiffusionRequest | None:
        """Build a one-step model request for startup profiling or warmup."""

        prompt: OmniTextPrompt = {"prompt": "dummy run"}
        supports_image_input, supports_audio_input = supports_multimodal_input(self.od_config)
        if supports_image_input:
            color_format = image_color_format(self.od_config.model_class_name)
            images = [PIL.Image.new(color_format, (width, height)) for _ in range(num_image_inputs)]
            prompt.setdefault("multi_modal_data", {})["image"] = images[0] if len(images) == 1 else images

        if supports_audio_input:
            audio_sr = 16000
            prompt.setdefault("multi_modal_data", {})["audio"] = np.random.randn(audio_sr * 2).astype(np.float32)

        num_frames = get_dummy_run_num_frames(self.od_config.model_class_name, supports_audio_input)
        if num_frames <= 0:
            return None
        return OmniDiffusionRequest(
            prompt=prompt,
            request_id=DUMMY_DIFFUSION_REQUEST_ID,
            sampling_params=OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_inference_steps=1,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_outputs_per_prompt=1,
                extra_args={"cfg_text_scale": 1.0, "cfg_img_scale": 1.0},
            ),
        )

    def _prepare_diffusion_kv_profile_requests(self) -> list[OmniDiffusionRequest] | None:
        """Prepare the per-rank request batch used to profile paged-KV headroom.

        The profile executes directly on each Worker before the Scheduler and
        its KV manager exist. It uses the maximum number of requests that one
        rank can execute together. DLO+DP is the exception because each rank
        executes one request from the collective wave.

        Hunyuan is currently the only model integrated with
        ``paged_scheduler``; 1024x1024, enabled CFG, and the maximum advertised
        reference-image count exercise its first-step activation peak.
        Admission compares each preprocessed request's CFG count and tokenized
        sequence/target shape with the resulting per-request profile envelope.
        Future paged model integrations must extend this recipe for their
        serving limits (for example, video frame count) rather than reusing it
        silently.
        """

        if (
            getattr(self.od_config, "diffusion_kv_mode", DiffusionKVCacheMode.DENSE_LEGACY)
            is not DiffusionKVCacheMode.PAGED_SCHEDULER
        ):
            return None
        request = self._make_dummy_request(
            height=1024,
            width=1024,
            guidance_scale=5.0,
            num_image_inputs=get_dummy_run_num_image_inputs(self.od_config.model_class_name),
        )
        if request is None:
            raise RuntimeError("paged_scheduler requires a runnable Diffusion KV memory profile request")
        request = self._prepare_request_for_admission(request)
        kv_requests = request.diffusion_kv_requests
        if not kv_requests:
            raise RuntimeError("paged_scheduler profile preprocessing must produce Diffusion KV requests")
        self._diffusion_kv_profile_limits = (
            len(kv_requests),
            max(kv_request.seq_len for kv_request in kv_requests),
            max(kv_request.target_len for kv_request in kv_requests),
        )
        dlo_dp_request_mode = self.execution_mode is DiffusionExecutionMode.REQUEST_BATCH and _uses_dlo_dp_concurrency(
            self.od_config
        )
        profile_batch_size = 1 if dlo_dp_request_mode else _max_num_seqs(self.od_config)
        profile_requests: list[OmniDiffusionRequest] = []
        for index in range(profile_batch_size):
            profile_request = copy.copy(request)
            profile_request.request_id = f"{DUMMY_DIFFUSION_REQUEST_ID}/kv-profile-{index}"
            profile_request.sampling_params = copy.deepcopy(request.sampling_params)
            # Preprocessing owns request geometry and prepared model inputs,
            # but native KV requests remain Scheduler-only mutable state. The
            # profile bypasses Scheduler admission and must not send them to
            # the Worker.
            profile_request.diffusion_kv_requests = None
            profile_requests.append(profile_request)
        return profile_requests

    def _dummy_run(self):
        """A dummy run to warm up the model."""
        req = self._make_dummy_request(
            height=512,
            width=512,
            guidance_scale=0.0,
        )
        if req is None:
            logger.info("Skipping dummy warmup run (num_frames=0)")
            return
        logger.info("dummy run to warm up the model")
        output = self.add_req_and_wait_for_response(req)
        if output.error:
            raise RuntimeError(f"Dummy run failed: {output.error}")

    def _submit_rpc(
        self,
        method: str,
        timeout: float | None,
        args: tuple,
        kwargs: dict | None,
        unique_reply_rank: int | None,
    ) -> _RpcTask:
        assert isinstance(method, str), "Only string method names are supported for now"
        deadline = None if timeout is None else time.monotonic() + timeout
        task = _RpcTask(
            method=method,
            args=args,
            kwargs=kwargs,
            deadline=deadline,
            unique_reply_rank=unique_reply_rank,
        )
        with self._cv:
            self._rpc_queue.put(task)
            self._cv.notify_all()
        return task

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ) -> Any:
        """Call a method on worker processes and get results immediately.

        The call is enqueued and executed by the engine's busy loop between
        scheduler steps, so it is naturally serialized against per-request
        execute_fn() invocations without any explicit mutual-exclusion lock.

        Args:
            method: The method name (str) to execute on workers
            timeout: Optional timeout in seconds
            args: Positional arguments for the method
            kwargs: Keyword arguments for the method
            unique_reply_rank: If set, only get reply from this rank

        Returns:
            Single result if unique_reply_rank is provided, otherwise list of results
        """
        assert isinstance(method, str), "Only string method names are supported for now"

        # If the busy loop hasn't started yet (e.g. during _dummy_run in
        # __init__, or before the first async request after construction),
        # there is no busy-loop thread to drain the RPC queue. Fall back to
        # calling the executor directly, but serialize concurrent callers
        # via self._cv's underlying lock so multiple threads in this window
        # cannot race on the shared broadcast_mq / result_mq transport.
        if not self._loop_started:
            with self._cv:
                # Re-check under the lock: the busy loop may have started
                # between the outer check and acquiring the lock, in which
                # case we should use the queued path for proper ordering.
                if not self._loop_started:
                    return self.executor.collective_rpc(
                        method=method,
                        timeout=timeout,
                        args=args,
                        kwargs=kwargs,
                        unique_reply_rank=unique_reply_rank,
                    )

        task = self._submit_rpc(method, timeout, args, kwargs, unique_reply_rank)
        try:
            return task.future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            task.future.cancel()
            raise TimeoutError(f"RPC call to {method} timed out.") from exc

    async def async_collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ) -> Any:
        """Async variant of :meth:`collective_rpc` for event-loop callers.

        Enqueue a task keyed by a future and ``await`` the result without
        blocking the loop.
        """
        await self._check_and_start_background_loop()
        task = self._submit_rpc(method, timeout, args, kwargs, unique_reply_rank)
        aio_fut = asyncio.wrap_future(task.future)
        try:
            if timeout is None:
                return await aio_fut
            return await asyncio.wait_for(aio_fut, timeout=timeout)
        except asyncio.TimeoutError as exc:
            task.future.cancel()
            raise TimeoutError(f"RPC call to {method} timed out.") from exc

    def _put_queue_output(
        self,
        queue: asyncio.Queue[DiffusionOutput],
        output: DiffusionOutput,
    ) -> None:
        loop = self.main_loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(queue.put_nowait, output)
        else:
            queue.put_nowait(output)

    def _put_output(self, request_id: str, output: DiffusionOutput) -> None:
        with self._cv:
            queue = self._out_streams.get(request_id)
        if queue is None:
            return
        self._put_queue_output(queue, output)

    def close(self) -> None:
        pending_streams: list[asyncio.Queue[DiffusionOutput]] = []
        with self._cv:
            if self._closed and self._shutdown_complete:
                return
            if not self._closed:
                self._closed = True
                if self.stop_event is not None:
                    self.stop_event.set()
                pending_streams = list(self._out_streams.values())
                self._out_streams.clear()
                self._cv.notify_all()

        closed_output = DiffusionOutput(error="DiffusionEngine is closed.")
        for stream in pending_streams:
            self._put_queue_output(stream, closed_output)

        worker_thread = self.worker_thread
        if worker_thread is not None:
            if worker_thread.is_alive():
                worker_thread.join(timeout=10)
            if worker_thread.is_alive():
                logger.warning(
                    "Worker thread did not terminate within 10s; scheduler and executor shutdown will be deferred."
                )
                return
            else:
                self._loop_started = False
        else:
            self._loop_started = False

        self.scheduler.close()
        self.executor.shutdown()
        self._shutdown_complete = True

    def abort(self, request_id: str | Iterable[str]) -> None:
        request_ids = [request_id] if isinstance(request_id, str) else list(request_id)

        with self._cv:
            if self._closed:
                return
            for req_id in request_ids:
                self.abort_queue.put(req_id)
            self._cv.notify_all()

    def _process_aborts_queue(self) -> None:
        with self._cv:
            self._drain_abort_queue()

    def _drain_abort_queue(self) -> None:
        if self.abort_queue.empty():
            return

        request_ids: list[str] = []
        while not self.abort_queue.empty():
            ids = self.abort_queue.get_nowait()
            request_ids.extend((ids,) if isinstance(ids, str) else ids)

        self._abort_requests(request_ids)

    def _abort_requests(self, request_ids: str | Iterable[str]) -> None:
        request_ids = [request_ids] if isinstance(request_ids, str) else list(request_ids)
        request_ids = list(dict.fromkeys(request_ids))

        self._remove_diffusion_kv_requests(request_ids)

        for request_id in request_ids:
            if self.scheduler.get_request_state(request_id) is not None:
                self.scheduler.finish_requests(request_id, DiffusionRequestStatus.FINISHED_ABORTED)

    def _finalize_finished_request(
        self,
        request_id: str,
        runner_output: RunnerOutput | None = None,
        missing_result_error: str = "Diffusion scheduler finished target request without execution output.",
    ) -> DiffusionOutput:
        state = self.scheduler.get_request_state(request_id)
        popped_state = self.scheduler.pop_request_state(request_id)
        state = state or popped_state

        if state is None:
            raise RuntimeError(f"Diffusion scheduler lost state for request {request_id}.")

        if state.status == DiffusionRequestStatus.FINISHED_ABORTED:
            # Preserve runner-provided abort details when available.
            if runner_output is not None and runner_output.result is not None and runner_output.result.aborted:
                return runner_output.result
            return DiffusionOutput(
                aborted=True,
                abort_message=f"Request {request_id} aborted.",
            )

        if runner_output is not None and runner_output.result is not None:
            return runner_output.result

        if runner_output is not None and runner_output.async_output_id is not None:
            return DiffusionOutput(async_output_id=runner_output.async_output_id)

        if state.status == DiffusionRequestStatus.FINISHED_ERROR and state.error:
            return DiffusionOutput(error=state.error)

        return DiffusionOutput(error=missing_result_error)
