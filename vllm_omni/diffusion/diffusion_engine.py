# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import inspect
import queue
import threading
import time
from collections.abc import Iterable
from typing import Any

import numpy as np
import PIL.Image
import torch
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError

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
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput, RunnerOutput
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniTextPrompt
from vllm_omni.outputs import OmniRequestOutput

logger = init_logger(__name__)


def supports_multimodal_input(od_config: OmniDiffusionConfig) -> tuple[bool, bool]:
    if od_config.diffusion_load_format == "diffusers" and (pipe_cls := od_config.diffusers_pipeline_cls) is not None:
        signature = inspect.signature(pipe_cls.__call__)
        support_image_input = "image" in signature.parameters
        support_audio_input = (
            "audio" in signature.parameters or "audio_latents" in signature.parameters
        )  # ref. LTX-2 format
        return support_image_input, support_audio_input

    supports_image_input = False
    supports_audio_input = False

    model_cls = DiffusionModelRegistry._try_load_model_cls(od_config.model_class_name)
    if model_cls is not None:
        supports_image_input = bool(getattr(model_cls, "support_image_input", False))
        supports_audio_input = bool(getattr(model_cls, "support_audio_input", False))
    return supports_image_input, supports_audio_input


def image_color_format(model_class_name: str) -> str:
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    return getattr(model_cls, "color_format", "RGB")


def supports_audio_output(model_class_name: str) -> bool:
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    if model_cls is None:
        return False
    return bool(getattr(model_cls, "support_audio_output", False))


class DiffusionEngine:
    """The diffusion engine for vLLM-Omni diffusion models."""

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
        self._post_process_accepts_sampling_params = bool(
            self.post_process_func is not None
            and "sampling_params" in inspect.signature(self.post_process_func).parameters
        )

        executor_class = DiffusionExecutor.get_class(od_config)
        self.executor = executor_class(od_config)
        self.step_execution = bool(getattr(od_config, "step_execution", False))
        self.scheduler: SchedulerInterface = scheduler or (
            StepScheduler() if self.step_execution else RequestScheduler()
        )
        self.scheduler.initialize(od_config)
        if self.scheduler.max_num_running_reqs > 1 and not self.step_execution:
            max_num_seqs = self.scheduler.max_num_running_reqs
            self.scheduler.max_num_running_reqs = 1
            logger.warning(f"Non-stepwise-execution does not support max-num-seqs={max_num_seqs}, set it to 1.")
        self.main_loop: asyncio.AbstractEventLoop | None = None
        self.stop_event: threading.Event | None = None
        self._loop_started = False
        self._init_lock = asyncio.Lock()
        self._rpc_lock = threading.RLock()
        self._cv = threading.Condition(self._rpc_lock)
        self._out_queue: dict[str, asyncio.Future] = {}
        self.abort_queue: queue.Queue[str] = queue.Queue()
        self.execute_fn = self.executor.execute_step if self.step_execution else self.executor.execute_request

        try:
            self._dummy_run()
        except Exception as e:
            logger.error(f"Dummy run failed: {e}")
            self.close()
            raise e

    async def _check_and_start_background_loop(self):
        if self._loop_started:
            return

        async with self._init_lock:
            # double check, in case of lock queue issue
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
            logger.info(f"Pre-processing completed in {preprocess_time:.4f} seconds")

        exec_start_time = time.perf_counter()
        output = await self.async_add_req_and_wait_for_response(request)
        exec_total_time = time.perf_counter() - exec_start_time

        if output.aborted:
            raise DiffusionRequestAbortedError(output.abort_message or "Diffusion request aborted.")
        if output.error:
            raise RuntimeError(output.error)
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
        if (
            self.od_config.enable_cpu_offload
            and isinstance(output_data, torch.Tensor)
            and output_data.device.type != "cpu"
        ):
            output_data = output_data.cpu()

        postprocess_start_time = time.perf_counter()
        if self.post_process_func is not None:
            # Some video pipelines need request-level controls during
            # postprocess (for example worker-side frame interpolation).
            if self._post_process_accepts_sampling_params:
                outputs = self.post_process_func(output_data, sampling_params=request.sampling_params)
            else:
                outputs = self.post_process_func(output_data)
        else:
            outputs = output_data
        audio_payload = None
        custom_output = output.custom_output or {}
        model_audio_sample_rate = None
        model_fps = None
        if isinstance(outputs, dict):
            audio_payload = outputs.get("audio")
            custom_output.update(outputs.get("custom_output") or {})
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
        # Ensure outputs is a list
        if not isinstance(outputs, list):
            outputs = [outputs] if outputs is not None else []

        metrics = {
            "preprocess_time_ms": preprocess_time * 1000,
            "diffusion_engine_exec_time_ms": exec_total_time * 1000,
            "diffusion_engine_total_time_ms": step_total_ms,
            "image_num": int(request.sampling_params.num_outputs_per_prompt),
            "resolution": int(request.sampling_params.resolution),
            "postprocess_time_ms": postprocess_time * 1000,
        }

        # Handle single request or multiple requests
        is_audio_output = supports_audio_output(self.od_config.model_class_name)
        if is_audio_output and model_audio_sample_rate is None:
            model_cls = DiffusionModelRegistry._try_load_model_cls(self.od_config.model_class_name)
            model_audio_sample_rate = getattr(model_cls, "audio_sample_rate", None)

        def _audio_mm(payload: Any) -> dict[str, Any]:
            mm: dict[str, Any] = {"audio": payload}
            if model_audio_sample_rate is not None:
                mm["audio_sample_rate"] = model_audio_sample_rate
            return mm

        if len(request.prompts) == 1:
            # Single request: return single OmniRequestOutput
            prompt = request.prompts[0]
            request_id = request.request_ids[0] if request.request_ids else ""

            if is_audio_output:
                request_audio_payload = outputs[0] if len(outputs) == 1 else outputs
                return [
                    OmniRequestOutput.from_diffusion(
                        request_id=request_id,
                        images=[],
                        prompt=prompt,
                        metrics=metrics,
                        latents=output.trajectory_latents,
                        trajectory_latents=output.trajectory_latents,
                        trajectory_timesteps=output.trajectory_timesteps,
                        trajectory_log_probs=output.trajectory_log_probs,
                        trajectory_decoded=output.trajectory_decoded,
                        multimodal_output=_audio_mm(request_audio_payload),
                        final_output_type="audio",
                        stage_durations=output.stage_durations,
                        peak_memory_mb=output.peak_memory_mb,
                    ),
                ]
            else:
                mm_output = {}
                if audio_payload is not None:
                    mm_output["audio"] = audio_payload
                if model_audio_sample_rate is not None:
                    mm_output["audio_sample_rate"] = model_audio_sample_rate
                if model_fps is not None:
                    mm_output["fps"] = model_fps
                return [
                    OmniRequestOutput.from_diffusion(
                        request_id=request_id,
                        images=outputs,
                        prompt=prompt,
                        metrics=metrics,
                        latents=output.trajectory_latents,
                        trajectory_latents=output.trajectory_latents,
                        trajectory_timesteps=output.trajectory_timesteps,
                        trajectory_log_probs=output.trajectory_log_probs,
                        trajectory_decoded=output.trajectory_decoded,
                        custom_output=custom_output,
                        multimodal_output=mm_output,
                        stage_durations=output.stage_durations,
                        peak_memory_mb=output.peak_memory_mb,
                    ),
                ]
        else:
            # Multiple requests: return list of OmniRequestOutput
            # Split images based on num_outputs_per_prompt for each request
            results = []
            output_idx = 0

            for i, prompt in enumerate(request.prompts):
                request_id = request.request_ids[i] if i < len(request.request_ids) else ""

                # Get images for this request
                num_outputs = request.sampling_params.num_outputs_per_prompt
                start_idx = output_idx
                end_idx = start_idx + num_outputs
                request_outputs = outputs[start_idx:end_idx] if output_idx < len(outputs) else []
                output_idx = end_idx

                if is_audio_output:
                    request_audio_payload = request_outputs[0] if len(request_outputs) == 1 else request_outputs
                    results.append(
                        OmniRequestOutput.from_diffusion(
                            request_id=request_id,
                            images=[],
                            prompt=prompt,
                            metrics=metrics,
                            latents=output.trajectory_latents,
                            trajectory_latents=output.trajectory_latents,
                            trajectory_timesteps=output.trajectory_timesteps,
                            trajectory_log_probs=output.trajectory_log_probs,
                            trajectory_decoded=output.trajectory_decoded,
                            multimodal_output=_audio_mm(request_audio_payload),
                            final_output_type="audio",
                            stage_durations=output.stage_durations,
                            peak_memory_mb=output.peak_memory_mb,
                        ),
                    )
                else:
                    mm_output = {}
                    if audio_payload is not None:
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
                        mm_output["audio"] = sliced_audio
                    if model_audio_sample_rate is not None:
                        mm_output["audio_sample_rate"] = model_audio_sample_rate
                    if model_fps is not None:
                        mm_output["fps"] = model_fps
                    results.append(
                        OmniRequestOutput.from_diffusion(
                            request_id=request_id,
                            images=request_outputs,
                            prompt=prompt,
                            metrics=metrics,
                            latents=output.trajectory_latents,
                            trajectory_latents=output.trajectory_latents,
                            trajectory_timesteps=output.trajectory_timesteps,
                            trajectory_log_probs=output.trajectory_log_probs,
                            trajectory_decoded=output.trajectory_decoded,
                            custom_output=custom_output,
                            multimodal_output=mm_output,
                            stage_durations=output.stage_durations,
                            peak_memory_mb=output.peak_memory_mb,
                        ),
                    )

            return results

    def _busy_loop(self):
        while not self.stop_event.is_set():
            self._process_aborts_queue()

            with self._cv:
                while not self.scheduler.has_requests() and not self.stop_event.is_set():
                    self._cv.wait(timeout=1.0)

                if self.stop_event.is_set():
                    break

                sched_output = self.scheduler.schedule()

            if sched_output.is_empty:
                self._handle_finished_requests(sched_output.finished_req_ids, None)
                continue

            try:
                runner_output = self.execute_fn(sched_output)
            except Exception as exc:
                logger.error(
                    "Execution failed for diffusion requests %s", sched_output.scheduled_req_ids, exc_info=True
                )
                runner_output = BatchRunnerOutput.from_list(
                    [
                        RunnerOutput(
                            req_id=req_id,
                            step_index=None,
                            finished=True,
                            result=DiffusionOutput(error=str(exc)),
                        )
                        for req_id in sched_output.scheduled_req_ids
                    ]
                )

            self._process_aborts_queue()
            finished_req_ids = self.scheduler.update_from_output(sched_output, runner_output)
            self._handle_finished_requests(finished_req_ids, runner_output)

    def _handle_finished_requests(
        self,
        finished_ids: set[str],
        runner_output: RunnerOutput | None = None,
        missing_result_error: str = "Diffusion execution finished without a final output",
    ):
        for rid in finished_ids:
            if rid in self._out_queue:
                fut = self._out_queue.pop(rid)
                if runner_output is not None:
                    _output = runner_output.get_req_output(rid)
                else:
                    _output = None
                out = self._finalize_finished_request(rid, _output, missing_result_error)
                if not fut.done():
                    self.main_loop.call_soon_threadsafe(fut.set_result, out)

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

    def add_request(self, request: OmniDiffusionRequest) -> str:
        fut = self.main_loop.create_future()
        with self._cv:
            sched_req_id = self.scheduler.add_request(request)
            self._out_queue[sched_req_id] = fut
            self._cv.notify_all()

        return sched_req_id

    async def get_result(self, request_id: str) -> DiffusionOutput:
        fut = self._out_queue.get(request_id)

        if fut is None:
            raise RuntimeError(f"Request {request_id} not found in output queue.")
        try:
            return await fut
        except Exception as e:
            logger.error(f"Wait for response failed: {e}")
            raise

    async def async_add_req_and_wait_for_response(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        sched_req_id = self.add_request(request)
        return await self.get_result(sched_req_id)

    def add_req_and_wait_for_response(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        with self._rpc_lock:
            target_sched_req_id = self.scheduler.add_request(request)

            # keep scheduling and executing until the target request is finished
            while True:
                self._process_aborts_queue()
                sched_output = self.scheduler.schedule()
                if sched_output.is_empty:
                    if target_sched_req_id in sched_output.finished_req_ids:
                        return self._finalize_finished_request(target_sched_req_id)
                    if not self.scheduler.has_requests():
                        raise RuntimeError("Diffusion scheduler has no runnable requests.")
                    continue

                # NOTE: add_req_and_wait_for_response() is synchronous, will be only called
                # within _dummy_run, only one request will be scheduled
                sched_req_id = sched_output.scheduled_req_ids[0]
                try:
                    runner_output = self.execute_fn(sched_output)
                except EngineDeadError:
                    raise
                except Exception as exc:
                    logger.error("Execution failed for diffusion request %s", sched_req_id, exc_info=True)
                    runner_output = RunnerOutput(
                        req_id=sched_req_id,
                        step_index=None,
                        finished=True,
                        result=DiffusionOutput(error=str(exc)),
                    )

                self._process_aborts_queue()

                finished_req_ids = self.scheduler.update_from_output(sched_output, runner_output)

                # sync func should receive one result
                if not isinstance(runner_output, RunnerOutput) and not len(runner_output) == 1:
                    raise ValueError("Sync func should receive one result at one time")
                if target_sched_req_id in finished_req_ids:
                    runner_output = runner_output.get_req_output(target_sched_req_id)
                    return self._finalize_finished_request(
                        target_sched_req_id,
                        runner_output=runner_output,
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
        request = self.pre_process_func(req) if self.pre_process_func is not None else req
        output = self.add_req_and_wait_for_response(request)
        if output.error:
            raise RuntimeError(f"Dummy run failed: {output.error}")

    def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        unique_reply_rank: int | None = None,
    ) -> Any:
        """Call a method on worker processes and get results immediately.

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

        deadline = None if timeout is None else time.monotonic() + timeout
        acquired = False
        try:
            if deadline is None:
                self._rpc_lock.acquire()
                acquired = True
            else:
                lock_timeout = max(0, deadline - time.monotonic())
                acquired = self._rpc_lock.acquire(timeout=lock_timeout)
            if not acquired:
                raise TimeoutError(f"RPC call to {method} timed out waiting for engine lock.")

            rpc_timeout = None if deadline is None else max(0, deadline - time.monotonic())
            if deadline is not None and rpc_timeout <= 0:
                raise TimeoutError(f"RPC call to {method} timed out.")

            return self.executor.collective_rpc(
                method=method,
                timeout=rpc_timeout,
                args=args,
                kwargs=kwargs,
                unique_reply_rank=unique_reply_rank,
            )
        finally:
            if acquired:
                self._rpc_lock.release()

    def close(self) -> None:
        if hasattr(self, "scheduler"):
            self.scheduler.close()
        if hasattr(self, "stop_event") and self.stop_event is not None:
            with self._cv:
                self.stop_event.set()
                self._cv.notify_all()
        if hasattr(self, "worker_thread") and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=10)
            if self.worker_thread.is_alive():
                logger.warning("Worker thread did not terminate within 10s, resources may not be released.")
            else:
                self._loop_started = False
        if hasattr(self, "executor"):
            self.executor.shutdown()

    def abort(self, request_id: str | Iterable[str]) -> None:
        request_ids = [request_id] if isinstance(request_id, str) else list(request_id)

        with self._cv:
            for req_id in request_ids:
                self.abort_queue.put(req_id)
            self._cv.notify_all()

    def _process_aborts_queue(self) -> None:
        if self.abort_queue.empty():
            return

        request_ids: list[str] = []
        while not self.abort_queue.empty():
            ids = self.abort_queue.get_nowait()
            request_ids.extend((ids,) if isinstance(ids, str) else ids)

        self._abort_requests(request_ids)

    def _abort_requests(self, request_ids: str | Iterable[str]) -> None:
        request_ids = [request_ids] if isinstance(request_ids, str) else list(request_ids)

        sched_req_ids: list[str] = []
        for request_id in dict.fromkeys(request_ids):
            sched_req_id = self.scheduler.get_sched_req_id(request_id)
            if sched_req_id is not None:
                sched_req_ids.append(sched_req_id)

        for sched_req_id in dict.fromkeys(sched_req_ids):
            if self.scheduler.get_request_state(sched_req_id) is not None:
                self.scheduler.finish_requests(sched_req_id, DiffusionRequestStatus.FINISHED_ABORTED)

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
