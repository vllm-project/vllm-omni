# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterable
from concurrent.futures import Future, InvalidStateError
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any

import numpy as np
import PIL.Image
import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import (
    DiffusionOutput,
    DiffusionRequestAbortedError,
    OmniDiffusionConfig,
)
from vllm_omni.diffusion.executor.abstract import DiffusionExecutor
from vllm_omni.diffusion.registry import (
    DiffusionModelRegistry,
    get_diffusion_post_process_func,
    get_diffusion_pre_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import RequestScheduler, SchedulerInterface, StepScheduler
from vllm_omni.diffusion.sched.interface import DiffusionRequestStatus
from vllm_omni.diffusion.worker.utils import RunnerOutput
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


def supports_image_input(model_class_name: str) -> bool:
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    if model_cls is None:
        return False
    return bool(getattr(model_cls, "support_image_input", False))


def supports_audio_input(model_class_name: str) -> bool:
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    if model_cls is None:
        return False
    return bool(getattr(model_cls, "support_audio_input", False))


def image_color_format(model_class_name: str) -> str:
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    return getattr(model_cls, "color_format", "RGB")


def supports_audio_output(model_class_name: str) -> bool:
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    if model_cls is None:
        return False
    return bool(getattr(model_cls, "support_audio_output", False))


@dataclass(slots=True)
class _AddRequestCmd:
    """Command payload for a newly submitted diffusion request.

    External threads never mutate scheduler state directly. They submit the
    request plus a reply future to the queue, and the dedicated core loop
    thread becomes the sole owner of request lifecycle transitions.
    """

    request: OmniDiffusionRequest
    future: Future[DiffusionOutput]


@dataclass(slots=True)
class _AbortCmd:
    """Command payload for aborting one or more public request ids."""

    request_ids: list[str]


@dataclass(slots=True)
class _RpcCmd:
    """Command payload for an executor collective RPC."""

    method: str
    timeout: float | None
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    unique_reply_rank: int | None
    future: Future[Any]


@dataclass(slots=True)
class _ShutdownCmd:
    """Command telling the core loop to reject new work and exit quickly."""


_DiffusionCmd = _AddRequestCmd | _AbortCmd | _RpcCmd | _ShutdownCmd


class DiffusionEngine:
    """Diffusion engine for vLLM-Omni diffusion models.

    Scheduler coordination is owned by a dedicated in-process core loop thread.
    Caller threads only enqueue commands and wait on futures, which removes the
    old caller-driven busy loop and the global RPC lock.
    """

    CORE_READY_TIMEOUT_S = 30.0
    CORE_THREAD_JOIN_TIMEOUT_S = 10.0

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        scheduler: SchedulerInterface | None = None,
    ):
        """Initialize the diffusion engine.

        Args:
            od_config: The configuration for the diffusion engine.
            scheduler: Optional scheduler implementation to install.
        """
        self.od_config = od_config

        self.post_process_func = get_diffusion_post_process_func(od_config)
        self.pre_process_func = get_diffusion_pre_process_func(od_config)

        executor_class = DiffusionExecutor.get_class(od_config)
        self.executor = executor_class(od_config)
        self.step_execution = bool(getattr(od_config, "step_execution", False))
        self.scheduler: SchedulerInterface = scheduler or (
            StepScheduler() if self.step_execution else RequestScheduler()
        )
        self.scheduler.initialize(od_config)
        self.execute_fn = self.executor.execute_step if self.step_execution else self.executor.execute_request

        self._start_core_thread()

        try:
            self._wait_for_core_ready()
            self._dummy_run()
        except Exception as e:
            logger.error(f"Dummy run failed: {e}")
            self.close()
            raise e

    def step(self, request: OmniDiffusionRequest) -> list[OmniRequestOutput]:
        """Run one synchronous diffusion request end to end."""
        diffusion_engine_start_time = time.perf_counter()
        request, preprocess_time = self._prepare_step_request(request)

        exec_start_time = time.perf_counter()
        output = self.add_req_and_wait_for_response(request)
        exec_total_time = time.perf_counter() - exec_start_time

        return self._materialize_step_outputs(
            request=request,
            output=output,
            preprocess_time=preprocess_time,
            exec_total_time=exec_total_time,
            diffusion_engine_start_time=diffusion_engine_start_time,
        )

    @staticmethod
    def make_engine(
        config: OmniDiffusionConfig,
        scheduler: SchedulerInterface | None = None,
    ) -> DiffusionEngine:
        """Factory method to create a DiffusionEngine instance.

        Args:
            config: The configuration for the diffusion engine.

        Returns:
            An instance of DiffusionEngine.
        """
        return DiffusionEngine(config, scheduler=scheduler)

    def submit_request(self, request: OmniDiffusionRequest) -> Future[DiffusionOutput]:
        """Submit a diffusion request to the queue-backed core loop.

        Args:
            request: The fully prepared diffusion request to schedule.

        Returns:
            A future that resolves to the terminal ``DiffusionOutput``.

        Notes:
            The latched core-loop error is checked before the shutdown flag so
            new callers observe the original crash reason instead of a generic
            closed error whenever the owner thread exits unexpectedly.
        """
        future: Future[DiffusionOutput] = Future()
        if self._core_loop_error is not None:
            future.set_exception(self._clone_exception(self._core_loop_error))
            return future
        if self._shutdown_requested.is_set():
            future.set_exception(RuntimeError("DiffusionEngine is closed."))
            return future

        self._cmd_queue.put(_AddRequestCmd(request=request, future=future))
        self._fail_submitted_future_if_owner_gone(future)
        return future

    def add_req_and_wait_for_response(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        """Synchronously submit a request and wait for its final result."""
        return self.submit_request(request).result()

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
        if supports_image_input(self.od_config.model_class_name):
            # Provide a dummy image input if the model supports it
            color_format = image_color_format(self.od_config.model_class_name)
            dummy_image = PIL.Image.new(color_format, (width, height))
        else:
            dummy_image = None

        if supports_audio_input(self.od_config.model_class_name):
            audio_sr = 16000
            dummy_audio_duration_sec = 2
            dummy_audio = np.random.randn(audio_sr * dummy_audio_duration_sec).astype(np.float32)
        else:
            dummy_audio = None

        prompt: OmniTextPrompt = {
            "prompt": "dummy run",
            "multi_modal_data": {"image": dummy_image, "audio": dummy_audio},
        }
        req = OmniDiffusionRequest(
            prompts=[prompt],
            request_ids=["dummy_req_id"],
            sampling_params=OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
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
        request, _ = self._prepare_step_request(req)
        output = self.add_req_and_wait_for_response(request)
        if output.error:
            raise RuntimeError(f"Dummy run failed: {output.error}")

    def submit_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        unique_reply_rank: int | None = None,
    ) -> Future[Any]:
        """Submit an executor collective RPC to the core loop."""
        assert isinstance(method, str), "Only string method names are supported for now"

        future: Future[Any] = Future()
        if self._core_loop_error is not None:
            future.set_exception(self._clone_exception(self._core_loop_error))
            return future
        if self._shutdown_requested.is_set():
            future.set_exception(RuntimeError("DiffusionEngine is closed."))
            return future

        self._cmd_queue.put(
            _RpcCmd(
                method=method,
                timeout=timeout,
                args=args,
                kwargs=kwargs or {},
                unique_reply_rank=unique_reply_rank,
                future=future,
            )
        )
        self._fail_submitted_future_if_owner_gone(future)
        return future

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ) -> Any:
        """Call a method on worker processes and wait for the result."""
        future = self.submit_rpc(
            method=method,
            timeout=timeout,
            args=args,
            kwargs=kwargs,
            unique_reply_rank=unique_reply_rank,
        )
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"RPC call to {method} timed out.") from exc

    def close(self) -> None:
        """Stop the core loop and best-effort release engine resources.

        Shutdown is intentionally fail-fast. The owner thread is asked to exit,
        then joined for a bounded amount of time. If it cannot make progress
        quickly enough, ``close()`` forcefully fails still-pending futures so
        blocked callers do not hang forever behind a stuck executor call.
        """
        failure = RuntimeError("DiffusionEngine is closed.")

        shutdown_requested = getattr(self, "_shutdown_requested", None)
        if shutdown_requested is not None:
            shutdown_requested.set()

        cmd_queue = getattr(self, "_cmd_queue", None)
        if cmd_queue is not None:
            try:
                cmd_queue.put_nowait(_ShutdownCmd())
            except Exception:
                pass

        core_thread = getattr(self, "_core_thread", None)
        if core_thread is not None and core_thread.is_alive():
            core_thread.join(timeout=self.CORE_THREAD_JOIN_TIMEOUT_S)
            if core_thread.is_alive():
                logger.warning(
                    "DiffusionEngine core thread did not exit within %.1f seconds.",
                    self.CORE_THREAD_JOIN_TIMEOUT_S,
                )
                # The owner thread is still blocked, so perform a last-resort
                # cleanup from the caller thread to wake blocked submitters.
                self._fail_pending_futures(failure)
                self._fail_queued_command_futures(failure)
            else:
                # The core thread already ran its own finally block, but a new
                # submitter may still have raced with shutdown after that
                # cleanup completed. Sweep the queue one last time.
                self._fail_queued_command_futures(self._owner_unavailable_error())
        elif core_thread is not None:
            self._fail_queued_command_futures(self._owner_unavailable_error())
        elif core_thread is None:
            # Tests may construct a partially initialized engine without the core loop.
            self._best_effort_shutdown_components()

    def abort(self, request_id: str | Iterable[str]) -> None:
        """Request abortion of one or more public request ids."""
        request_ids = [request_id] if isinstance(request_id, str) else list(request_id)
        if not request_ids:
            return
        self._cmd_queue.put(_AbortCmd(request_ids=request_ids))

    def _start_core_thread(self) -> None:
        """Allocate core-loop state and start the owner thread.

        ``_pending_futures`` belongs to the core loop logically, but
        ``close()`` may perform a last-resort cleanup if the owner thread is
        stuck and cannot service shutdown in time.
        """
        self._cmd_queue: queue.Queue[_DiffusionCmd] = queue.Queue()
        self._pending_futures: dict[str, Future[DiffusionOutput]] = {}
        self._shutdown_requested = threading.Event()
        self._core_ready = threading.Event()
        self._core_loop_error: RuntimeError | None = None
        self._core_stopping = False
        self._in_step_execution = False
        self._core_thread = threading.Thread(
            target=self._run_core_loop,
            name="DiffusionEngineCore",
            daemon=True,
        )
        self._core_thread.start()

    def _wait_for_core_ready(self) -> None:
        """Wait until the owner thread is ready to consume commands.

        Raises:
            RuntimeError: If the core thread does not reach its ready point
                before the initialization timeout expires.
        """
        if self._core_ready.wait(timeout=self.CORE_READY_TIMEOUT_S):
            return
        raise RuntimeError(f"DiffusionEngine core loop did not become ready within {self.CORE_READY_TIMEOUT_S:.1f}s.")

    def _prepare_step_request(
        self,
        request: OmniDiffusionRequest,
    ) -> tuple[OmniDiffusionRequest, float]:
        """Apply optional preprocessing before the request enters scheduling."""
        preprocess_time = 0.0
        if self.pre_process_func is not None:
            preprocess_start_time = time.perf_counter()
            request = self.pre_process_func(request)
            preprocess_time = time.perf_counter() - preprocess_start_time
            logger.info(f"Pre-processing completed in {preprocess_time:.4f} seconds")
        return request, preprocess_time

    def _build_step_metrics(
        self,
        request: OmniDiffusionRequest,
        *,
        preprocess_ms: float,
        exec_ms: float,
        postprocess_ms: float,
        total_ms: float,
    ) -> dict[str, float | int]:
        """Build the public diffusion metrics emitted by ``step()``.

        Args:
            request: Logical diffusion request whose sampling parameters are
                mirrored into the telemetry payload.
            preprocess_ms: Time spent in the optional preprocessing hook.
            exec_ms: Time spent waiting for the scheduled diffusion execution.
            postprocess_ms: Time spent converting the terminal diffusion output
                into public response objects.
            total_ms: End-to-end wall-clock time for the entire step call.

        Returns:
            A metrics dictionary with grouped key names for timings and request
            parameters.

        Notes:
            All timing values are passed in from the caller after it has frozen
            the measurement boundaries. This helper must not call
            ``time.perf_counter()`` on its own, or the metrics payload would
            drift from the logged step breakdown.
        """
        return {
            "time_preprocess_ms": preprocess_ms,
            "time_exec_ms": exec_ms,
            "time_postprocess_ms": postprocess_ms,
            "time_total_ms": total_ms,
            "param_num_outputs_per_prompt": int(request.sampling_params.num_outputs_per_prompt),
            "param_resolution": int(request.sampling_params.resolution),
        }

    def _move_output_to_cpu(self, output_data: Any) -> Any:
        """Recursively move diffusion outputs to CPU for postprocessing.

        Args:
            output_data: Terminal diffusion output payload. Some models return
                a single tensor, while others may return nested ``tuple``,
                ``list``, or ``dict`` structures that contain tensors.

        Returns:
            A payload with every tensor moved to CPU before postprocessing
            consumes it.
        """
        if isinstance(output_data, torch.Tensor):
            if output_data.device.type == "cpu":
                return output_data
            return output_data.to("cpu")
        if isinstance(output_data, tuple):
            return tuple(self._move_output_to_cpu(item) for item in output_data)
        if isinstance(output_data, list):
            return [self._move_output_to_cpu(item) for item in output_data]
        if isinstance(output_data, dict):
            return {key: self._move_output_to_cpu(value) for key, value in output_data.items()}
        return output_data

    def _extract_audio_slice(
        self,
        audio_payload: Any,
        *,
        start_idx: int,
        end_idx: int,
        num_outputs: int,
    ) -> Any:
        """Slice the shared audio payload down to one logical request.

        Args:
            audio_payload: Combined audio payload returned by postprocessing.
                Depending on the backend, this may be a Python sequence or an
                array-like object with a leading batch dimension.
            start_idx: Inclusive start offset for the current request.
            end_idx: Exclusive end offset for the current request.
            num_outputs: Number of outputs requested for the current prompt.

        Returns:
            The audio payload belonging to the current logical request.
        """
        sliced_audio = audio_payload

        if isinstance(audio_payload, (list, tuple)):
            sliced_audio = audio_payload[start_idx:end_idx]
            if len(sliced_audio) == 1:
                sliced_audio = sliced_audio[0]
        elif hasattr(audio_payload, "shape") and getattr(audio_payload, "shape", None) is not None:
            if len(audio_payload.shape) > 0 and audio_payload.shape[0] >= end_idx:
                sliced_audio = audio_payload[start_idx:end_idx]
                if num_outputs == 1:
                    sliced_audio = sliced_audio[0]
        else:
            logger.warning(
                "Audio payload of type %s does not support per-request slicing; "
                "reusing the original payload for request range [%d:%d).",
                type(audio_payload).__name__,
                start_idx,
                end_idx,
            )

        return sliced_audio

    def _build_request_output(
        self,
        *,
        request_id: str,
        prompt: OmniTextPrompt,
        request_outputs: Any,
        metrics: dict[str, float | int],
        diffusion_output: DiffusionOutput,
        model_outputs_audio: bool,
        request_multimodal_output: dict[str, Any] | None = None,
    ) -> OmniRequestOutput:
        """Build one public response object from per-request payloads.

        Args:
            request_id: Public request identifier for the logical request being
                materialized.
            prompt: Prompt corresponding to the logical request.
            request_outputs: Outputs already narrowed to the current request.
            metrics: Step-level telemetry payload shared by all prompts in the
                same diffusion step.
            diffusion_output: Raw terminal diffusion output that carries
                latents and auxiliary metadata.
            model_outputs_audio: Whether the primary model output modality is
                audio instead of images.
            request_multimodal_output: Optional per-request multimodal payload
                attached under ``multimodal_output``.

        Returns:
            The final ``OmniRequestOutput`` for one logical request.

        Notes:
            Batch-scoped metadata such as ``trajectory_latents``,
            ``custom_output``, ``stage_durations``, and ``peak_memory_mb`` is
            intentionally copied into every per-request output. This preserves
            the historical diffusion response contract for callers that already
            expect those fields to be available on each logical result.
        """
        request_multimodal_output = request_multimodal_output or {}

        if model_outputs_audio:
            request_audio_payload = request_outputs
            if isinstance(request_outputs, (list, tuple)) and len(request_outputs) == 1:
                request_audio_payload = request_outputs[0]
            multimodal_output = {"audio": request_audio_payload}
            multimodal_output.update(request_multimodal_output)
            return OmniRequestOutput.from_diffusion(
                request_id=request_id,
                images=[],
                prompt=prompt,
                metrics=metrics,
                latents=diffusion_output.trajectory_latents,
                trajectory_latents=diffusion_output.trajectory_latents,
                trajectory_timesteps=diffusion_output.trajectory_timesteps,
                trajectory_log_probs=diffusion_output.trajectory_log_probs,
                trajectory_decoded=diffusion_output.trajectory_decoded,
                multimodal_output=multimodal_output,
                final_output_type="audio",
                stage_durations=diffusion_output.stage_durations,
                peak_memory_mb=diffusion_output.peak_memory_mb,
            )

        return OmniRequestOutput.from_diffusion(
            request_id=request_id,
            images=request_outputs,
            prompt=prompt,
            metrics=metrics,
            latents=diffusion_output.trajectory_latents,
            trajectory_latents=diffusion_output.trajectory_latents,
            trajectory_timesteps=diffusion_output.trajectory_timesteps,
            trajectory_log_probs=diffusion_output.trajectory_log_probs,
            trajectory_decoded=diffusion_output.trajectory_decoded,
            custom_output=diffusion_output.custom_output or {},
            multimodal_output=request_multimodal_output,
            stage_durations=diffusion_output.stage_durations,
            peak_memory_mb=diffusion_output.peak_memory_mb,
        )

    def _materialize_step_outputs(
        self,
        request: OmniDiffusionRequest,
        output: DiffusionOutput,
        preprocess_time: float,
        exec_total_time: float,
        diffusion_engine_start_time: float,
    ) -> list[OmniRequestOutput]:
        """Convert a terminal output into ``OmniRequestOutput`` objects.

        Notes:
            This helper keeps ``step()`` readable by separating terminal output
            materialization from request scheduling. It preserves the existing
            response semantics while applying the remaining telemetry and
            readability cleanups tracked in Issue #2335.
        """
        if output.aborted:
            raise DiffusionRequestAbortedError(output.abort_message or "Diffusion request aborted.")
        if output.error:
            raise Exception(f"{output.error}")
        logger.info("Generation completed successfully.")

        if output.output is None:
            logger.warning("Output is None, returning empty OmniRequestOutput")
            return [
                OmniRequestOutput.from_diffusion(
                    request_id=request.request_ids[i] if i < len(request.request_ids) else "",
                    images=[],
                    prompt=prompt,
                    metrics={},
                    latents=None,
                )
                for i, prompt in enumerate(request.prompts)
            ]

        # When CPU offload is enabled, move output to CPU before
        # post-processing to avoid device OOM — model weights may still
        # reside on the device and leave no headroom for intermediates.
        output_data = output.output
        if self.od_config.enable_cpu_offload:
            output_data = self._move_output_to_cpu(output_data)

        postprocess_start_time = time.perf_counter()
        outputs = self.post_process_func(output_data) if self.post_process_func is not None else output_data
        audio_payload = None
        model_audio_sample_rate = None
        model_fps = None
        if isinstance(outputs, dict):
            audio_payload = outputs.get("audio")
            model_audio_sample_rate = outputs.get("audio_sample_rate")
            model_fps = outputs.get("fps")
            outputs = outputs.get("video", outputs)
        postprocess_time = time.perf_counter() - postprocess_start_time
        logger.info(f"Post-processing completed in {postprocess_time:.4f} seconds")

        step_total_ms = (time.perf_counter() - diffusion_engine_start_time) * 1000
        logger.info(
            "DiffusionEngine.step breakdown: preprocess=%.2f ms, "
            "add_req_and_wait=%.2f ms, postprocess=%.2f ms, total=%.2f ms",
            preprocess_time * 1000,
            exec_total_time * 1000,
            postprocess_time * 1000,
            step_total_ms,
        )

        # Convert to OmniRequestOutput format
        model_outputs_audio = supports_audio_output(self.od_config.model_class_name)
        if not model_outputs_audio and not isinstance(outputs, list):
            outputs = [outputs] if outputs is not None else []

        preprocess_ms = preprocess_time * 1000
        exec_ms = exec_total_time * 1000
        postprocess_ms = postprocess_time * 1000
        total_ms = step_total_ms
        metrics = self._build_step_metrics(
            request,
            preprocess_ms=preprocess_ms,
            exec_ms=exec_ms,
            postprocess_ms=postprocess_ms,
            total_ms=total_ms,
        )

        single_request = len(request.prompts) == 1
        results = []
        output_idx = 0

        for i, prompt in enumerate(request.prompts):
            request_id = request.request_ids[i] if i < len(request.request_ids) else ""
            num_outputs = request.sampling_params.num_outputs_per_prompt

            if single_request:
                request_outputs = outputs
            else:
                start_idx = output_idx
                end_idx = start_idx + num_outputs
                if model_outputs_audio:
                    request_outputs = self._extract_audio_slice(
                        outputs,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        num_outputs=num_outputs,
                    )
                else:
                    request_outputs = outputs[start_idx:end_idx] if output_idx < len(outputs) else []
                output_idx = end_idx

            request_multimodal_output: dict[str, Any] = {}
            if not model_outputs_audio and audio_payload is not None:
                request_multimodal_output["audio"] = (
                    audio_payload
                    if single_request
                    else self._extract_audio_slice(
                        audio_payload,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        num_outputs=num_outputs,
                    )
                )
            if model_audio_sample_rate is not None:
                request_multimodal_output["audio_sample_rate"] = model_audio_sample_rate
            if model_fps is not None:
                request_multimodal_output["fps"] = model_fps

            results.append(
                self._build_request_output(
                    request_id=request_id,
                    prompt=prompt,
                    request_outputs=request_outputs,
                    metrics=metrics,
                    diffusion_output=output,
                    model_outputs_audio=model_outputs_audio,
                    request_multimodal_output=request_multimodal_output,
                )
            )

        return results

    def _run_core_loop(self) -> None:
        """Run the dedicated owner loop for scheduler and executor coordination.

        The loop alternates between two modes:
        - idle: block on the command queue until new work arrives
        - active: drain commands, run one scheduler/execute/update cycle, repeat

        Any unexpected exception is latched and converted into failures for all
        pending futures so no caller can remain blocked forever.
        """
        self._core_ready.set()
        try:
            while not self._core_stopping:
                if self.scheduler.has_requests():
                    # Keep ingesting external commands while there is runnable
                    # scheduler state. This allows new requests, aborts and RPCs
                    # to arrive concurrently with long-running worker execution.
                    self._drain_commands(block=False)
                    if not self._core_stopping and self.scheduler.has_requests():
                        self._step_engine_once()
                    continue

                # When there is no unfinished request we park on the queue,
                # which removes the old busy-wait behavior entirely.
                self._drain_commands(block=True)
        except BaseException as exc:
            logger.error("DiffusionEngine core loop crashed.", exc_info=True)
            self._core_loop_error = RuntimeError(f"DiffusionEngine core loop exited unexpectedly: {exc}")
            self._shutdown_requested.set()
        finally:
            # The thread may crash before a submitter gets past the constructor's
            # wait. Setting the event again here guarantees the waiter is
            # released even on a startup failure path.
            self._core_ready.set()
            if self._core_loop_error is not None:
                failure = self._clone_exception(self._core_loop_error)
            elif self._shutdown_requested.is_set():
                failure = RuntimeError("DiffusionEngine is closed.")
            else:
                failure = RuntimeError("DiffusionEngine core loop exited unexpectedly.")
            self._fail_pending_futures(failure)
            self._fail_queued_command_futures(failure)
            self._best_effort_shutdown_components()

    def _drain_commands(self, block: bool) -> None:
        """Drain queued commands on the owner thread.

        Args:
            block: If ``True``, wait for at least one command before returning.
                Otherwise only process commands that are already queued.
        """
        should_block = block
        while not self._core_stopping:
            try:
                cmd = self._cmd_queue.get() if should_block else self._cmd_queue.get_nowait()
            except queue.Empty:
                return

            should_block = False
            self._handle_command(cmd)

    def _handle_command(self, cmd: _DiffusionCmd) -> None:
        """Dispatch one command on the owner thread."""
        if isinstance(cmd, _AddRequestCmd):
            self._handle_add_request_command(cmd)
            return
        if isinstance(cmd, _AbortCmd):
            self._handle_abort_command(cmd)
            return
        if isinstance(cmd, _RpcCmd):
            self._handle_rpc_command(cmd)
            return
        if isinstance(cmd, _ShutdownCmd):
            self._handle_shutdown_command()
            return
        raise TypeError(f"Unsupported diffusion command: {type(cmd)!r}")

    def _handle_add_request_command(self, cmd: _AddRequestCmd) -> None:
        """Register a newly submitted request with the scheduler."""
        if cmd.future.cancelled():
            return
        if self._core_loop_error is not None:
            self._try_set_future_exception(cmd.future, self._clone_exception(self._core_loop_error))
            return
        if self._shutdown_requested.is_set():
            self._try_set_future_exception(cmd.future, RuntimeError("DiffusionEngine is closed."))
            return

        try:
            sched_req_id = self.scheduler.add_request(cmd.request)
        except Exception as exc:
            self._try_set_future_exception(cmd.future, exc)
            return
        except BaseException as exc:
            self._try_set_future_exception(
                cmd.future,
                RuntimeError(f"Diffusion scheduler add_request failed: {exc}"),
            )
            raise

        # Future ownership starts here and stays on the core thread until the
        # request reaches a terminal state, unless close() must force cleanup.
        self._pending_futures[sched_req_id] = cmd.future

    def _handle_abort_command(self, cmd: _AbortCmd) -> None:
        """Abort queued or running requests by public request id.

        Waiting requests can be resolved immediately because they have not been
        handed to the executor yet. Running requests rely on the second command
        drain inside ``_step_engine_once()`` so the aborted state is visible
        before ``scheduler.update_from_output()`` finalizes the runner output.
        """
        waiting_req_ids: list[str] = []

        sched_req_ids: list[str] = []
        for request_id in dict.fromkeys(cmd.request_ids):
            sched_req_id = self.scheduler.get_sched_req_id(request_id)
            if sched_req_id is not None:
                sched_req_ids.append(sched_req_id)

        for sched_req_id in dict.fromkeys(sched_req_ids):
            state = self.scheduler.get_request_state(sched_req_id)
            if state is None or state.is_finished():
                continue

            was_waiting = state.status != DiffusionRequestStatus.RUNNING
            self.scheduler.finish_requests(sched_req_id, DiffusionRequestStatus.FINISHED_ABORTED)
            if was_waiting:
                waiting_req_ids.append(sched_req_id)

        for sched_req_id in waiting_req_ids:
            self._resolve_finished_request(sched_req_id, runner_output=None)

    def _handle_rpc_command(self, cmd: _RpcCmd) -> None:
        """Execute a worker RPC if the caller is still waiting for it.

        Notes:
            ``Future.set_running_or_notify_cancel()`` is the key queue-based
            replacement for the old engine-lock timeout semantics. If a
            timed-out caller already cancelled the future, the RPC is skipped
            entirely.
        """
        if cmd.future.cancelled():
            return
        if self._core_loop_error is not None:
            self._try_set_future_exception(cmd.future, self._clone_exception(self._core_loop_error))
            return
        if self._shutdown_requested.is_set():
            self._try_set_future_exception(cmd.future, RuntimeError("DiffusionEngine is closed."))
            return
        if not cmd.future.set_running_or_notify_cancel():
            return

        try:
            result = self.executor.collective_rpc(
                method=cmd.method,
                timeout=cmd.timeout,
                args=cmd.args,
                kwargs=cmd.kwargs,
                unique_reply_rank=cmd.unique_reply_rank,
            )
        except Exception as exc:
            self._try_set_future_exception(cmd.future, exc)
            return
        except BaseException as exc:
            self._try_set_future_exception(
                cmd.future,
                RuntimeError(f"DiffusionEngine RPC {cmd.method!r} failed: {exc}"),
            )
            raise

        self._try_set_future_result(cmd.future, result)

    def _handle_shutdown_command(self) -> None:
        """Mark the engine closed and begin fast shutdown.

        Shutdown remains fail-fast, but if the command is observed from the
        second command drain inside ``_step_engine_once()``, we defer failing
        request futures until the current runner output has been folded back
        into scheduler state. This avoids discarding a result that is already
        available on the CPU while still rejecting all remaining work.
        """
        self._shutdown_requested.set()
        self._core_stopping = True

        failure = RuntimeError("DiffusionEngine is closed.")
        # If shutdown arrives from the second drain inside _step_engine_once(),
        # keep pending request futures intact for a moment so the already
        # returned runner output can still resolve whatever finished work is
        # recoverable. Any leftover futures are then failed by
        # _run_core_loop()'s finally block during teardown.
        if not self._in_step_execution:
            self._fail_pending_futures(failure)
        self._fail_queued_command_futures(failure)

    def _step_engine_once(self) -> None:
        """Run one scheduler/execute/update cycle on the owner thread.

        The order here is intentional:
        1. schedule runnable work
        2. execute one runner call
        3. drain commands that arrived during execution
        4. update scheduler state from the runner output
        5. resolve any finished request futures

        The second drain between execute and update is what lets a running
        request observe an abort before ``update_from_output()`` commits the
        runner result as successfully completed.
        """
        sched_output = self.scheduler.schedule()
        if sched_output.is_empty:
            return

        self._in_step_execution = True
        try:
            sched_req_id = sched_output.scheduled_req_ids[0]
            try:
                runner_output = self.execute_fn(sched_output)
            except Exception as exc:
                # Convert unexpected execution failures into a synthetic terminal
                # runner output so the scheduler can still drive the request to a
                # FINISHED_ERROR state instead of leaking lifecycle ownership.
                logger.error("Execution failed for diffusion request %s", sched_req_id, exc_info=True)
                runner_output = RunnerOutput(
                    req_id=sched_req_id,
                    step_index=None,
                    finished=True,
                    result=DiffusionOutput(error=str(exc)),
                )

            # Process abort / RPC / shutdown commands that arrived while
            # execute_fn() was blocked on worker-side execution before we fold
            # the runner output back into scheduler state.
            self._drain_commands(block=False)

            finished_req_ids = self.scheduler.update_from_output(sched_output, runner_output)
            for finished_req_id in finished_req_ids:
                self._resolve_finished_request(finished_req_id, runner_output=runner_output)
        finally:
            self._in_step_execution = False

    def _resolve_finished_request(
        self,
        sched_req_id: str,
        runner_output: RunnerOutput | None,
    ) -> None:
        """Finalize scheduler state and complete the matching caller future."""
        future = self._pending_futures.pop(sched_req_id, None)
        try:
            output = self._finalize_finished_request(
                sched_req_id,
                runner_output=runner_output,
                missing_result_error="Diffusion execution finished without a final output.",
            )
        except BaseException as exc:
            # A buggy third-party scheduler could lose the request state between
            # update_from_output() and finalization. If that happens, fail the
            # waiting caller future before re-raising so the core-loop crash
            # path does not orphan a blocked submitter forever.
            if future is not None:
                self._try_set_future_exception(
                    future,
                    RuntimeError(f"Failed to finalize diffusion request {sched_req_id}: {exc}"),
                )
            raise

        if future is not None:
            self._try_set_future_result(future, output)

    def _fail_pending_futures(self, error: Exception) -> None:
        """Fail every request future that is still pending."""
        pending_futures = list(self._pending_futures.values())
        self._pending_futures.clear()
        for future in pending_futures:
            self._try_set_future_exception(future, self._clone_exception(error))

    def _fail_queued_command_futures(self, error: Exception) -> None:
        """Fail command futures still sitting in the queue."""
        while True:
            try:
                cmd = self._cmd_queue.get_nowait()
            except queue.Empty:
                return

            if isinstance(cmd, (_AddRequestCmd, _RpcCmd)):
                self._try_set_future_exception(cmd.future, self._clone_exception(error))

    def _best_effort_shutdown_components(self) -> None:
        """Close scheduler and executor without masking the original failure."""
        if hasattr(self, "scheduler"):
            try:
                self.scheduler.close()
            except Exception:
                logger.warning("Failed to close diffusion scheduler cleanly.", exc_info=True)
        if hasattr(self, "executor"):
            try:
                self.executor.shutdown()
            except Exception:
                logger.warning("Failed to shut down diffusion executor cleanly.", exc_info=True)

    def _try_set_future_result(self, future: Future[Any], result: Any) -> None:
        """Best-effort future completion that tolerates benign cross-thread races.

        ``close()`` may force cleanup from a caller thread while the core loop
        is simultaneously unwinding the same request. If another thread wins
        the completion race after our ``done()`` check but before
        ``set_result()`` runs, ``InvalidStateError`` simply means the future
        was already resolved.
        """
        if future.done():
            return
        try:
            future.set_result(result)
        except InvalidStateError:
            return

    def _try_set_future_exception(self, future: Future[Any], error: BaseException) -> None:
        """Best-effort exception completion that ignores benign state races."""
        if future.done():
            return
        try:
            future.set_exception(error)
        except InvalidStateError:
            return

    def _fail_submitted_future_if_owner_gone(self, future: Future[Any]) -> None:
        """Fail a just-enqueued command if the core thread already exited.

        This defends against a small check-then-enqueue race: the caller can
        observe an open engine, enqueue a command, and only then discover that
        the core loop has already exited and will never consume the queue
        entry. A second shutdown or crash check is required even while the
        core thread still reports itself alive, because the owner may already
        be in its teardown path and no longer willing to accept new work.
        Turning that race into an immediate future failure avoids a
        permanently blocked caller.
        """
        core_thread = getattr(self, "_core_thread", None)
        shutdown_requested = getattr(self, "_shutdown_requested", None)
        is_shutting_down = shutdown_requested.is_set() if shutdown_requested is not None else False

        if (
            core_thread is not None
            and core_thread.is_alive()
            and self._core_loop_error is None
            and not is_shutting_down
        ):
            return

        error = self._owner_unavailable_error()
        self._try_set_future_exception(future, self._clone_exception(error))
        self._fail_queued_command_futures(error)

    def _owner_unavailable_error(self) -> Exception:
        """Return the most specific error that explains why the owner is gone."""
        if self._core_loop_error is not None:
            return self._core_loop_error
        if self._shutdown_requested.is_set():
            return RuntimeError("DiffusionEngine is closed.")
        return RuntimeError("DiffusionEngine core loop is not running.")

    @staticmethod
    def _clone_exception(error: Exception) -> Exception:
        """Create a fresh exception instance before attaching it to a future."""
        try:
            return error.__class__(*error.args)
        except Exception:
            return RuntimeError(str(error))

    def _finalize_finished_request(
        self,
        sched_req_id: str,
        runner_output: RunnerOutput | None = None,
        missing_result_error: str = "Diffusion scheduler finished target request without execution output.",
    ) -> DiffusionOutput:
        state = self.scheduler.get_request_state(sched_req_id)
        popped_state = self.scheduler.pop_request_state(sched_req_id)
        state = state or popped_state

        if state is None:
            raise RuntimeError(f"Diffusion scheduler lost state for request {sched_req_id}.")

        if state.status == DiffusionRequestStatus.FINISHED_ABORTED:
            request_id = state.req.request_ids[0] if state.req.request_ids else sched_req_id
            return DiffusionOutput(
                aborted=True,
                abort_message=f"Request {request_id} aborted.",
            )

        if runner_output is not None and runner_output.result is not None:
            return runner_output.result

        return DiffusionOutput(error=missing_result_error)
