# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import queue
import threading
import time
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import PIL.Image
import torch
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.diffusion.data import (
    DiffusionOutput,
    DiffusionRequestAbortedError,
    OmniDiffusionConfig,
)
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.io_support import (
    get_dummy_run_num_frames,
    image_color_format,
    supports_audio_output,
    supports_multimodal_input,
)
from vllm_omni.diffusion.output_formatter import (
    DiffusionStepTimings,
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
from vllm_omni.diffusion.sched import RequestScheduler, SchedulerInterface, StepScheduler
from vllm_omni.diffusion.sched.interface import DiffusionRequestStatus
from vllm_omni.diffusion.worker.utils import BaseRunnerOutput, BatchRunnerOutput, RunnerOutput
from vllm_omni.errors import client_error_from_metadata, is_client_error_status
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt

if TYPE_CHECKING:
    from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)

__all__ = [
    "DiffusionEngine",
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
    if model_cls is None:
        model_cls = DiffusionModelRegistry._try_load_model_cls(getattr(od_config, "model_class_name", None))
    if model_cls is None:
        return False
    return bool(getattr(model_cls, "supports_request_batch", False))


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


class DiffusionEngine:
    """The diffusion engine for vLLM-Omni diffusion models."""

    #: Import path of the model runner this engine's workers should build, or
    #: ``None`` for the platform default. Subclasses declare their runner here
    #: (e.g. the AR-Diffusion engine), the worker resolves it through
    #: :meth:`resolve_engine_class` — so engine->runner routing lives on the
    #: engine class itself, and ``od_config.diffusion_model_runner_cls``
    #: remains a pure explicit user override (never mutated by engines).
    default_diffusion_model_runner_cls: str | None = None

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        scheduler: SchedulerInterface | None = None,
    ):
        """Initialize the diffusion engine.

        Args:
            config: The configuration for the diffusion engine.
        """
        self.od_config = od_config

        self.post_process_func = get_diffusion_post_process_func(od_config)
        self.pre_process_func = get_diffusion_pre_process_func(od_config)
        # Cache whether the model-specific postprocess accepts request-level
        # sampling params so step() can support both legacy and extended hooks.
        self._post_process_accepts_sampling_params = _func_accepts_parameter(self.post_process_func, "sampling_params")

        self.step_execution = bool(getattr(od_config, "step_execution", False))
        if self.od_config.streaming_output and not self.step_execution:
            logger.warning("streaming_output=True requires step_execution=True; enabling step execution.")
            self.od_config.step_execution = True
            self.step_execution = True

        executor_class = DiffusionExecutor.get_class(od_config)
        self.executor = executor_class(od_config)
        self.scheduler: SchedulerInterface = scheduler or (
            StepScheduler() if self.step_execution else RequestScheduler()
        )
        self.scheduler.initialize(od_config)
        self.supports_request_batch = False if self.step_execution else supports_request_batch(od_config)
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
        self._out_queue: dict[str, asyncio.Future] = {}
        self._out_queue_streaming: dict[str, asyncio.Queue[DiffusionOutput]] = {}
        self._closed = False
        self._shutdown_complete = False
        self.abort_queue: queue.Queue[str] = queue.Queue()
        self._rpc_queue: queue.Queue[_RpcTask] = queue.Queue()
        if self.step_execution:
            self.execute_fn = self.executor.execute_step
        elif self.supports_request_batch:
            self.execute_fn = self.executor.execute_batch
        else:
            self.execute_fn = self.executor.execute_request

        if self.supports_request_batch:
            logger.info(
                "[RequestBatch] engine init max_num_seqs=%s max_wait_ms=%s",
                getattr(od_config, "max_num_seqs", None),
                getattr(od_config, "request_batch_max_wait_ms", None),
            )

        try:
            self._dummy_run()
        except Exception as e:
            logger.error(f"Dummy run failed: {e}")
            self.close()
            raise e

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

    async def step(self, request: OmniDiffusionRequest) -> list[OmniRequestOutput]:
        await self._check_and_start_background_loop()

        diffusion_engine_start_time = time.perf_counter()

        # Apply pre-processing if available
        preprocess_time = 0.0
        if self.pre_process_func is not None:
            preprocess_start_time = time.perf_counter()
            request = self.pre_process_func(request)
            preprocess_time = time.perf_counter() - preprocess_start_time
            logger.debug("Pre-processing completed in %.4f seconds", preprocess_time)

        exec_start_time = time.perf_counter()
        output = await self.async_add_req_and_wait_for_response(request)
        exec_total_time = time.perf_counter() - exec_start_time
        return self.postprocess_output(request, output, diffusion_engine_start_time, preprocess_time, exec_total_time)

    async def step_streaming(self, request: OmniDiffusionRequest) -> AsyncGenerator[list[OmniRequestOutput], None]:
        await self._check_and_start_background_loop()

        diffusion_engine_start_time = time.perf_counter()

        preprocess_time = 0.0
        if self.pre_process_func is not None:
            preprocess_start_time = time.perf_counter()
            request = self.pre_process_func(request)
            preprocess_time = time.perf_counter() - preprocess_start_time
            logger.debug("Pre-processing completed in %.4f seconds", preprocess_time)

        exec_start_time = time.perf_counter()
        generator = self.async_add_req_and_stream_response(request)
        async for output in generator:
            exec_total_time = time.perf_counter() - exec_start_time
            yield self.postprocess_output(
                request, output, diffusion_engine_start_time, preprocess_time, exec_total_time
            )

    def postprocess_output(
        self,
        request: OmniDiffusionRequest,
        output: DiffusionOutput,
        diffusion_engine_start_time: float,
        preprocess_time: float,
        exec_total_time: float,
    ) -> list[OmniRequestOutput]:
        """Convert a DiffusionOutput to a list of OmniRequestOutput, attaching profiling metrics."""
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

        postprocess_start_time = time.perf_counter()
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
        postprocess_time = time.perf_counter() - postprocess_start_time
        logger.debug("Post-processing completed in %.4f seconds", postprocess_time)

        step_total_ms = (time.perf_counter() - diffusion_engine_start_time) * 1000
        logger.debug(
            "DiffusionEngine.step breakdown: preprocess=%.2f ms, "
            "add_req_and_wait=%.2f ms, postprocess=%.2f ms, total=%.2f ms",
            preprocess_time * 1000,
            exec_total_time * 1000,
            postprocess_time * 1000,
            step_total_ms,
        )

        return format_diffusion_outputs(
            request=request,
            od_config=self.od_config,
            diffusion_output=output,
            output_data=output_data,
            postprocess_output=postprocess_output,
            timings=DiffusionStepTimings(
                preprocess_time_s=preprocess_time,
                exec_time_s=exec_total_time,
                postprocess_time_s=postprocess_time,
                total_time_ms=step_total_ms,
            ),
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

                if self.supports_request_batch:
                    self._wait_for_request_batch_admission_locked()

                sched_output = self.scheduler.schedule()

            if sched_output.is_empty:
                if self.od_config.streaming_output:
                    self._handle_empty_streaming_requests(sched_output.finished_req_ids)
                else:
                    self._handle_finished_requests(sched_output.finished_req_ids, None)
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
            if self.od_config.streaming_output:
                self._handle_step_streaming_runner_output(
                    finished_req_ids,
                    sched_output.scheduled_request_ids,
                    runner_output,
                )
            else:
                self._handle_finished_requests(finished_req_ids, runner_output)

        # Engine is stopping: fail any RPCs still queued so callers don't hang.
        self._fail_pending_rpcs(RuntimeError("DiffusionEngine is shutting down."))

    def _wait_for_request_batch_admission_locked(self) -> None:
        """Wait for compatible requests to accumulate before scheduling a wave.

        Caller must hold ``self._cv``.
        """
        if self.step_execution or not self.supports_request_batch:
            return

        max_wait_s = self.od_config.request_batch_max_wait_ms / 1000.0
        if max_wait_s == 0:
            return

        max_batch = self.scheduler.max_num_running_reqs
        waiting = self.scheduler.num_waiting_requests()
        running = self.scheduler.num_running_requests()

        if running > 0:
            return

        start = time.monotonic()
        deadline = start + max_wait_s
        last_waiting = -1
        stable_since = start
        # Require a short idle period with no queue growth so bursty HTTP
        # ingress can land before the first schedule() of a wave.
        stable_window_s = min(0.05, max_wait_s / 5.0)

        while not self.stop_event.is_set():
            waiting = self.scheduler.num_waiting_requests()
            now = time.monotonic()

            if waiting >= max_batch:
                break
            if waiting > 0 and (now - stable_since) >= stable_window_s:
                break
            if now >= deadline:
                break

            if waiting > last_waiting:
                stable_since = now
                last_waiting = waiting

            remaining = deadline - now
            self._cv.wait(timeout=min(remaining, 0.002))

        waited_ms = (time.monotonic() - start) * 1000.0
        final_waiting = self.scheduler.num_waiting_requests()
        if final_waiting > 0:
            logger.info(
                "[RequestBatch] admission wait done waiting=%d max_batch=%d waited_ms=%.1f",
                final_waiting,
                max_batch,
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

    def _handle_finished_requests(
        self,
        finished_ids: set[str],
        runner_output: BaseRunnerOutput | None = None,
        missing_result_error: str = "Diffusion execution finished without a final output",
    ):
        for rid in finished_ids:
            with self._cv:
                fut = self._out_queue.pop(rid, None)
            if fut is None:
                continue
            if runner_output is not None:
                _output = runner_output.get_request_output(rid)
            else:
                _output = None
            out = self._finalize_finished_request(rid, _output, missing_result_error)
            self._complete_future(fut, out)

    def _handle_empty_streaming_requests(
        self,
        finished_ids: set[str],
        missing_result_error: str = "Diffusion streaming request finished without execution output.",
    ) -> None:
        """Mirrors `_handle_finished_requests()` in non-streaming mode when used for empty scheduler output."""
        for rid in finished_ids:
            out = self._finalize_finished_request(rid, None, missing_result_error=missing_result_error)
            self._put_streaming_output_with_cv(rid, out)

    def _handle_step_streaming_runner_output(
        self,
        finished_req_ids: set[str],
        scheduled_request_ids: list[str],
        runner_output: BaseRunnerOutput,
    ) -> None:
        """
        Deliver partial step-execution outputs in streaming mode.

        Step execution returns one ``RunnerOutput`` per scheduled request per
        engine tick. Most denoise steps have ``result=None``; chunk boundaries
        return a ``DiffusionOutput`` that must be delivered even before the
        scheduler marks the request finished.
        """
        delivered_finished_req_ids: set[str] = set()

        # finished_ids may have some requests that are not scheduler in this round.
        # First handle this-round requests.
        for request_id in scheduled_request_ids:
            req_output = runner_output.get_request_output(request_id)
            if request_id in finished_req_ids:
                # This entire request is finished (this is the last chunk)
                out = self._finalize_finished_request(
                    request_id,
                    req_output,
                    missing_result_error="Diffusion streaming execution finished without a final output.",
                )
                self._put_streaming_output_with_cv(request_id, out)
                delivered_finished_req_ids.add(request_id)
            elif req_output is not None and req_output.result is not None:
                # This is a non-terminal chunk. So it is not in scheduler's finished_req_ids, but still need delivering.
                self._put_streaming_output_with_cv(request_id, req_output.result)

        # Then handle other requests that are finished in this round.
        for request_id in finished_req_ids - delivered_finished_req_ids:
            out = self._finalize_finished_request(
                request_id,
                missing_result_error="Diffusion streaming request finished without execution output.",
            )
            self._put_streaming_output_with_cv(request_id, out)

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
        scheduler: SchedulerInterface | None = None,
    ) -> DiffusionEngine:
        """Factory method to create the engine selected by ``config.engine_backend``.

        Args:
            config: The configuration for the diffusion engine.

        Returns:
            An instance of the resolved ``DiffusionEngine`` (sub)class.
        """
        engine_class = DiffusionEngine.resolve_engine_class(config)
        return engine_class(config, scheduler=scheduler)

    def add_request(self, request: OmniDiffusionRequest) -> str:
        with self._cv:
            if self._closed:
                raise RuntimeError("DiffusionEngine is closed.")
            if not self.od_config.streaming_output:
                fut = self.main_loop.create_future()
                request_id = self.scheduler.add_request(request)
                self._out_queue[request_id] = fut
            else:
                queue: asyncio.Queue[DiffusionOutput] = asyncio.Queue()
                request_id = self.scheduler.add_request(request)
                self._out_queue_streaming[request_id] = queue
            self._cv.notify_all()

        return request_id

    async def get_result(self, request_id: str) -> DiffusionOutput:
        fut = self._out_queue.get(request_id)

        if fut is None:
            raise RuntimeError(f"Request {request_id} not found in output queue.")
        try:
            return await fut
        except Exception as e:
            logger.error(f"Wait for response failed: {e}")
            raise

    async def get_streaming_result(self, request_id: str) -> AsyncGenerator[DiffusionOutput, None]:
        """Mirrors `get_result()` in non-streaming mode."""
        with self._cv:
            queue = self._out_queue_streaming.get(request_id)
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
            # In streaming mode, an output queue is maintained until the terminal chunk is met.
            # So unlike the non-streaming mode where output Future is popped in `_handle_finished_requests` (immediately
            # after the request is returned), the streaming mode needs to pop the output queue here (one layer above).
            with self._cv:
                if self._out_queue_streaming.get(request_id) is queue:
                    self._out_queue_streaming.pop(request_id, None)

    async def async_add_req_and_wait_for_response(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        # No lock needed: add_request is already protected by self._cv, and
        # all executor calls are serialized inside the busy loop.
        request_id = self.add_request(request)
        return await self.get_result(request_id)

    def async_add_req_and_stream_response(self, request: OmniDiffusionRequest) -> AsyncGenerator[DiffusionOutput, None]:
        request_id = self.add_request(request)
        return self.get_streaming_result(request_id)

    def add_req_and_wait_for_response(self, request: OmniDiffusionRequest) -> DiffusionOutput:
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
                    req_output = runner_output.get_request_output(target_request_id)
                    return self._finalize_finished_request(
                        target_request_id,
                        runner_output=req_output,
                        missing_result_error="Diffusion execution finished without a final output.",
                    )

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

    def _dummy_run(self):
        """A dummy run to warm up the model."""
        num_inference_steps = 1
        height = 512
        width = 512
        prompt: OmniTextPrompt = {"prompt": "dummy run"}

        supports_image_input, supports_audio_input = supports_multimodal_input(self.od_config)
        if supports_image_input:
            # Provide a dummy image input if the model supports it
            color_format = image_color_format(self.od_config.model_class_name)
            dummy_image = PIL.Image.new(color_format, (width, height))
            prompt.setdefault("multi_modal_data", {})["image"] = dummy_image

        if supports_audio_input:
            audio_sr = 16000
            dummy_audio = np.random.randn(audio_sr * 2).astype(np.float32)
            prompt.setdefault("multi_modal_data", {})["audio"] = dummy_audio

        num_frames = get_dummy_run_num_frames(self.od_config.model_class_name, supports_audio_input)
        if num_frames <= 0:
            logger.info("Skipping dummy warmup run (num_frames=0)")
            return
        req = OmniDiffusionRequest(
            prompt=prompt,
            request_id=DUMMY_DIFFUSION_REQUEST_ID,
            sampling_params=OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
                # Keep warmup path minimal and robust across text encoders.
                # Some models may fail when warmup implicitly triggers
                # classifier-free guidance with an empty negative prompt.
                guidance_scale=0.0,
                num_outputs_per_prompt=1,
                # Disable CFG for warmup to avoid triggering CFG parallel
                # validation when cfg_parallel_size > 1.
                extra_args={"cfg_text_scale": 1.0, "cfg_img_scale": 1.0},
            ),
        )
        logger.info("dummy run to warm up the model")
        request = self.pre_process_func(req) if self.pre_process_func is not None else req
        output = self.add_req_and_wait_for_response(request)
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

        Mirrors :meth:`async_add_req_and_wait_for_response`: enqueue a task
        keyed by a future and ``await`` the result without blocking the loop.
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

    def _complete_future(self, fut: asyncio.Future, output: DiffusionOutput) -> None:
        if fut.done():
            return

        def _set_result() -> None:
            if not fut.done():
                fut.set_result(output)

        try:
            loop = fut.get_loop()
        except AttributeError:
            loop = self.main_loop

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if loop is not None and loop.is_running() and loop is not running_loop:
            loop.call_soon_threadsafe(_set_result)
        else:
            _set_result()

    def _put_streaming_queue_output(
        self,
        queue: asyncio.Queue[DiffusionOutput],
        output: DiffusionOutput,
    ) -> None:
        """Append to streaming output queue in a safe event loop. Mirrors `_complete_future()` in non-streaming mode."""
        loop = self.main_loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(queue.put_nowait, output)
        else:
            queue.put_nowait(output)

    def _put_streaming_output_with_cv(self, request_id: str, output: DiffusionOutput) -> None:
        with self._cv:
            queue = self._out_queue_streaming.get(request_id)
        if queue is None:
            return
        self._put_streaming_queue_output(queue, output)

    def close(self) -> None:
        pending_futures: list[asyncio.Future] = []
        pending_streaming_queues: list[asyncio.Queue[DiffusionOutput]] = []
        with self._cv:
            if self._closed and self._shutdown_complete:
                return
            if not self._closed:
                self._closed = True
                if self.stop_event is not None:
                    self.stop_event.set()
                pending_futures = list(self._out_queue.values())
                pending_streaming_queues = list(self._out_queue_streaming.values())
                self._out_queue.clear()
                self._out_queue_streaming.clear()
                self._cv.notify_all()

        closed_output = DiffusionOutput(error="DiffusionEngine is closed.")
        for fut in pending_futures:
            self._complete_future(fut, closed_output)
        for streaming_queue in pending_streaming_queues:
            self._put_streaming_queue_output(streaming_queue, closed_output)

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

        for request_id in dict.fromkeys(request_ids):
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

        return DiffusionOutput(error=missing_result_error)
