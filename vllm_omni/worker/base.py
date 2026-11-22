# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Base worker class for vLLM-Omni with device-level GPU memory profiling."""

from __future__ import annotations

import gc
import math
import os
import time
from contextlib import AbstractContextManager, nullcontext

import torch
from vllm import envs
from vllm.config import CUDAGraphMode
from vllm.logger import init_logger
from vllm.utils.mem_utils import format_gib, memory_profiling
from vllm.v1.worker.gpu_worker import CompilationTimes
from vllm.v1.worker.gpu_worker import Worker as GPUWorker

from vllm_omni.diffusion.data import (
    OmniACK,
    OmniSleepTask,
    OmniWakeTask,
)
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


class OmniGPUWorkerBase(GPUWorker):
    """Base GPU worker for vLLM-Omni.

    This class overrides determine_available_memory() to compute the KV cache
    budget from device-level profiling data. Concurrent same-device stage
    initialization is coordinated externally by the parallel-stage-init
    admission control + SH/EX device locks (see
    ``VllmOmniOrchestratorConfig.parallel_stage_init``), which keep each
    measurement quiescent instead of relying on per-process NVML accounting.

    It also replaces vLLM's TorchProfilerWrapper with OmniTorchProfilerWrapper
    for custom trace naming, background gzip, and trace path collection.
    """

    def _capture_auxiliary_graphs(self) -> None:
        """Let opt-in models warm valid inputs before the worker becomes ready."""
        capture = getattr(self.model_runner.model, "capture_auxiliary_graphs", None)
        if callable(capture):
            capture()

    def _profile_auxiliary_graph_memory(self) -> int:
        """Measure the model's ``capture_auxiliary_graphs`` pools for the KV budget.

        Only models that can also ``release_auxiliary_graphs`` are measured: the
        graphs are captured here, measured, and released, and warmup captures
        them for real as before. The model owns these graphs, so the runner's
        throwaway-pool profiling does not apply to them.
        """
        model = getattr(self.model_runner, "model", None)
        capture = getattr(model, "capture_auxiliary_graphs", None)
        release = getattr(model, "release_auxiliary_graphs", None)
        if not callable(capture) or not callable(release):
            return 0
        torch.accelerator.synchronize()
        torch.accelerator.empty_cache()
        free_before = torch.accelerator.get_memory_info()[0]
        try:
            capture()
            torch.accelerator.synchronize()
            # Drop the capture's cached temporaries; the graphs keep their pools.
            torch.accelerator.empty_cache()
            auxiliary_graph_memory = max(free_before - torch.accelerator.get_memory_info()[0], 0)
        finally:
            release()
            torch.accelerator.synchronize()
            torch.accelerator.empty_cache()
        logger.info(
            "Estimated auxiliary CUDA graph memory: %s GiB (%s GiB still held after release)",
            format_gib(auxiliary_graph_memory),
            format_gib(max(free_before - torch.accelerator.get_memory_info()[0], 0)),
        )
        return auxiliary_graph_memory

    def compile_or_warm_up_model(self) -> CompilationTimes:
        result = super().compile_or_warm_up_model()
        self._capture_auxiliary_graphs()
        return result

    def load_model(self, *args, **kwargs):
        with self._maybe_get_memory_pool_context("weights"):
            res = super().load_model(*args, **kwargs)
            current_omni_platform.synchronize()
            gc.collect()
            return res

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Replace vLLM's profiler with platform-specific profiler
        profiler_config = self.vllm_config.profiler_config
        if profiler_config and profiler_config.profiler == "torch":
            from vllm_omni.profiler import create_omni_profiler

            stage_id = getattr(self.vllm_config.model_config, "stage_id", 0)
            worker_name = f"stage{stage_id}_rank{self.rank}"
            self.profiler = create_omni_profiler(
                profiler_config=profiler_config,
                worker_name=worker_name,
                local_rank=self.local_rank,
            )
        elif profiler_config and profiler_config.profiler == "cuda":
            # OmniGPUWorkerBase replaces vLLM's torch profiler above; keep the
            # native CUDA profiler wrapper available for Nsight Systems capture
            # through the /start_profile and /stop_profile endpoints as well.
            from vllm.profiler.wrapper import CudaProfilerWrapper

            self.profiler = CudaProfilerWrapper(profiler_config)

    def profile(
        self,
        is_start: bool = True,
        profile_prefix: str | None = None,
        *,
        delay_iterations: int | None = None,
        max_iterations: int | None = None,
    ):
        """Override to set trace filename before starting the profiler.

        Args:
            is_start: True to start profiling, False to stop.
            profile_prefix: Optional prefix for trace filename (vLLM compat).
            delay_iterations: Optional per-session delayed start (vLLM compat).
            max_iterations: Optional per-session iteration cap (vLLM compat).

        Generate a stage-specific trace filename before delegating the
        lifecycle and iteration controls to the profiler.
        """
        if self.profiler is None:
            raise RuntimeError(
                "Profiling is not enabled. For diffusion models, set --profiler-config via CLI. "
                "For omni models, add profiler_config to the relevant deploy config stage."
            )
        if is_start:
            from vllm_omni.profiler import OmniTorchProfilerWrapper

            if isinstance(self.profiler, OmniTorchProfilerWrapper):
                # Include stage_id and rank in default filename to distinguish
                # traces from different stages profiling on the same local_rank
                stage_id = getattr(self.vllm_config.model_config, "stage_id", 0)
                filename = profile_prefix or f"stage{stage_id}_rank{self.rank}_{int(time.time())}"
                self.profiler.set_trace_filename(filename)
            kwargs: dict[str, int] = {}
            if delay_iterations is not None:
                kwargs["delay_iterations"] = delay_iterations
            if max_iterations is not None:
                kwargs["max_iterations"] = max_iterations
            self.profiler.start(**kwargs)
        else:
            self.profiler.stop()

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Device-level GPU memory profiling for the KV cache budget.

        Algorithm:
            1. requested_memory = total_gpu_memory * gpu_memory_utilization
               (computed in init_device from cache_config)

            2. Measure weights + peak_activation + non_torch_increase with
               ``memory_profiling`` around ``profile_run()``.
               ``non_torch_increase`` is device-level, so it reflects whatever
               else is resident on the GPU at profiling time.

            3. Profile the CUDA graphs the runner captures through
               ``cudagraph_dispatcher``, the encoder graph manager and the
               talker MTP wrapper. This has
               to stay outside the ``memory_profiling`` window above, because it
               allocates a minimal KV cache and captures graphs of its own,
               which would otherwise land in ``non_torch_increase`` and be
               counted twice. It runs only when
               VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS is enabled (it is by
               default), and the estimate is clamped at zero: it comes from
               device free-memory deltas, so a release elsewhere on the device
               during capture must not raise the budget above the request.
               Under ``parallel_stage_init`` this runs inside the profiling
               ``LOCK_EX`` (``engine/stage_phase_lock``), so co-located stages
               cannot allocate or capture meanwhile. On CUDA-alike platforms
               while VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS is on, a model that
               implements both ``capture_auxiliary_graphs`` and
               ``release_auxiliary_graphs`` captures its graphs here, whatever
               the cudagraph mode, and releases them once measured as
               auxiliary_graph_memory; warmup captures them for real. Stages
               without a KV cache never reach this method.

            4. profiled_usage = the step 2 measurements +
               cudagraph_memory_estimate + auxiliary_graph_memory

            5. available_kv_cache = requested_memory - profiled_usage

        Note:
            Process-scoped NVML estimation was removed in favour of the
            device-level profiling fallback plus the parallel-stage-init
            admission + SH/EX device locks (see
            ``VllmOmniOrchestratorConfig.parallel_stage_init``): the EX lock
            makes each measurement quiescent, so the device-level number is the
            correct, conservative budget. The NVML helpers in
            ``gpu_memory_utils`` are retained for diffusion memory reporting.
        """
        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            self.model_runner.profile_run()
            if current_omni_platform.is_rocm():
                torch.accelerator.synchronize()
            return kv_cache_memory_bytes

        with memory_profiling(
            self.init_snapshot,
            weights_memory=int(self.model_runner.model_memory_usage),
        ) as profile_result:
            self.model_runner.profile_run()

        cudagraph_memory_estimate = 0
        if (
            envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS
            and current_omni_platform.is_cuda_alike()
            and self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            cudagraph_memory_estimate = max(0, self.model_runner.profile_cudagraph_memory())

        auxiliary_graph_memory = 0
        if envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS and current_omni_platform.is_cuda_alike():
            auxiliary_graph_memory = self._profile_auxiliary_graph_memory()

        # Compared with what capture_model() reports, so it leaves out the
        # auxiliary graphs, which warmup captures after it.
        self.cudagraph_memory_estimate = cudagraph_memory_estimate
        self.non_torch_memory = profile_result.non_torch_increase
        # Upstream adds the applied estimate here as well, then adds the actual
        # graph size again when it suggests --kv-cache-memory in
        # compile_or_warm_up_model. Keep the activation peak alone so that
        # suggestion counts the graph pool once.
        self.peak_activation_memory = profile_result.torch_peak_increase
        # Upstream 58b2012aa2 added `total_consumed` to the profiling result
        # and reads it in GPUWorker.compile_or_warm_up_model() (when
        # kv_cache_memory_bytes is None and peak_activation_memory is set, both
        # true here). Mirror upstream so the omni override keeps it populated.
        # The auxiliary pools persist once warmup captures them, so the
        # --kv-cache-memory suggestion counts them here.
        self.total_consumed = profile_result.total_consumed + auxiliary_graph_memory

        profiled_usage = (
            int(self.model_runner.model_memory_usage)
            + profile_result.torch_peak_increase
            + profile_result.non_torch_increase
            + cudagraph_memory_estimate
            + auxiliary_graph_memory
        )
        self.available_kv_cache_memory_bytes = max(0, self.requested_memory - profiled_usage)
        logger.debug(
            "Profiling KV budget (PID %d, GPU %d): requested=%s, profiled=%s, available=%s",
            os.getpid(),
            self.local_rank,
            format_gib(self.requested_memory),
            format_gib(profiled_usage),
            format_gib(self.available_kv_cache_memory_bytes),
        )
        logger.info_once(
            "Available KV cache memory: %s GiB (device-level profiling)",
            format_gib(self.available_kv_cache_memory_bytes),
            scope="local",
        )
        self._log_cudagraph_reserve(cudagraph_memory_estimate + auxiliary_graph_memory)

        return int(self.available_kv_cache_memory_bytes)

    def _log_cudagraph_reserve(self, cudagraph_memory_estimate: int) -> None:
        """Tell the operator how the graph reserve moved their KV budget.

        Same guidance upstream ``GPUWorker.determine_available_memory`` prints:
        a stage tuned before the reserve existed keeps its effective KV size by
        raising ``gpu_memory_utilization`` by the graph pool's share. That only
        works while the raised budget still fits in the memory free at startup.
        Otherwise ``request_memory_tolerant`` caps it to the free memory, the
        normal case for co-located Omni stages, and the log says the previous
        size cannot be restored.
        """
        if cudagraph_memory_estimate <= 0:
            return
        total = self.init_snapshot.total_memory
        free = self.init_snapshot.free_memory
        current_util = self.cache_config.gpu_memory_utilization
        # Round up so the suggested value restores at least the reserved amount.
        # (0.8 + 0.1) * 10_000 is 9000.000000000002 in floating point, hence the inner round.
        scaled = round((current_util + cudagraph_memory_estimate / total) * 10_000, 6)
        suggested_util = math.ceil(scaled) / 10_000
        # request_memory_tolerant computes the budget the suggested value would get.
        suggested_memory = math.ceil(total * suggested_util)
        uncapped = self.requested_memory >= math.ceil(total * current_util)
        if uncapped and suggested_memory <= free:
            logger.info_once(
                "Reserved %s GiB for CUDA graphs out of this stage's KV budget; raise "
                "--gpu-memory-utilization from %.4f to %.4f to keep the previous KV size, or set "
                "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 to leave the graph pool unreserved.",
                format_gib(cudagraph_memory_estimate),
                current_util,
                suggested_util,
                scope="local",
            )
            return
        logger.info_once(
            "Reserved %s GiB for CUDA graphs out of this stage's KV budget; no "
            "--gpu-memory-utilization restores the previous KV size, because the budget it needs "
            "(%s GiB) exceeds the %s GiB free on the device at startup. Free memory on the device, "
            "or set VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 to leave the graph pool unreserved.",
            format_gib(cudagraph_memory_estimate),
            format_gib(suggested_memory if uncapped else self.requested_memory + cudagraph_memory_estimate),
            format_gib(free),
            scope="local",
        )

    # Provide memory pool context
    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        v1_config_enabled = False
        if hasattr(self, "vllm_config"):
            model_cfg = getattr(self.vllm_config, "model_config", None)
            v1_config_enabled = getattr(model_cfg, "enable_sleep_mode", False)

        is_sleep_enabled = v1_config_enabled or getattr(self.cache_config, "enable_sleep_mode", False)
        if is_sleep_enabled:
            current_omni_platform.synchronize()
            gc.collect()
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            logger.info(f"[LLM Worker {self.rank}] Sleep Mode ENABLED. Activating CuMem pool for tag: {tag}")
            return allocator.use_memory_pool(tag=tag)
        else:
            logger.warning(f"[LLM Worker {self.rank}] Sleep Mode DISABLED.")
            return nullcontext()

    def sleep(self, level: int = 1) -> bool:
        """
        Put the worker to sleep.
        Args:
            level: 1 (Offload weights to CPU), level: 2 (Total Discard).
        """
        from vllm.device_allocator.cumem import CuMemAllocator

        # Drain in-flight kernels before CuMem unmap. Abort/decode can still
        # have GPU work after the scheduler reports idle; unmapping then
        # returns CUDA_ERROR_INVALID_VALUE (cudaErrorInvalidValue).
        current_omni_platform.synchronize()
        gc.collect()

        mem_before = current_omni_platform.get_current_memory_usage(self.device)
        offload_tags = ("weights",) if level == 1 else tuple()
        allocator = CuMemAllocator.get_instance()
        allocator.sleep(offload_tags=offload_tags)
        # allocator.sleep already gc.collects and empty_cache()s. A second
        # empty_cache on the pluggable allocator is a known CUDA invalid
        # argument (pytorch/pytorch#145168). Only sync for the readout.
        current_omni_platform.synchronize()
        mem_after = current_omni_platform.get_current_memory_usage(self.device)
        freed = max(0, mem_before - mem_after)
        remaining_gb = mem_after / 1024**3
        logger.info(
            f"[LLM Worker {self.rank}] Level {level} Sleep: Freed "
            f"{freed / 1024**3:.2f} GiB. {remaining_gb:.2f}GiB memory "
            "is still in use."
        )
        return True

    def wake_up(self, tags: list[str] | None = None) -> bool:
        "Physical video memory reloading logic"
        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags)
        current_omni_platform.synchronize()
        logger.info(f"[LLM Worker {self.rank}] Wake-up complete.")
        return True

    def handle_sleep_task(self, task: OmniSleepTask) -> OmniACK | None:
        "Handle deterministic Sleep command from the main process"
        try:
            if isinstance(task, dict):
                task = OmniSleepTask(**task)
            logger.info(f"[LLM Worker {self.rank}] Handshake Received: Task {task.task_id}, Level {task.level}")
            if task.level == 2:
                if hasattr(self.model_runner, "graph_runners"):
                    self.model_runner.graph_runners.clear()
                    logger.info(f"[LLM Worker {self.rank}] LLM CUDA Graphs cleared.")
            mem_before = current_omni_platform.get_free_memory(self.device)
            self.sleep(level=task.level)
            mem_after = current_omni_platform.get_free_memory(self.device)
            rank_freed = max(0, mem_after - mem_before)
            if torch.distributed.is_initialized():
                t_freed = torch.tensor([float(rank_freed)], device=self.device)
                torch.distributed.all_reduce(t_freed)
                total_freed = int(t_freed.item())
                torch.distributed.barrier()
            else:
                total_freed = rank_freed
            if self.rank != 0:
                return None
            current_stage_id = getattr(self.vllm_config.model_config, "stage_id", 0)
            try:
                total_mem = current_omni_platform.get_device_total_memory()
            except (NotImplementedError, AttributeError):
                total_mem = torch.cuda.get_device_properties(self.device).total_memory
            residual_gib = (total_mem - mem_after) / 1024**3
            ack = OmniACK(
                task_id=task.task_id,
                status="SUCCESS",
                stage_id=current_stage_id,
                rank=self.rank,
                freed_bytes=total_freed,
                metadata={
                    "source": "omni_platform_audit",
                    "total_freed_gib": f"{total_freed / 1024**3:.2f}",
                    "rank_residual_gib": f"{residual_gib:.2f}",
                },
            )
            if hasattr(self, "result_mq") and self.result_mq:
                self.result_mq.put(ack)
            logger.info(f"[LLM Worker {self.rank}] ACK emitted for Task {task.task_id}")
            return ack
        except Exception as e:
            logger.error(f"[LLM Worker {self.rank}] Sleep Task Failed: {e}", exc_info=True)
            if torch.distributed.is_initialized():
                try:
                    torch.distributed.barrier()
                except Exception:
                    pass
            return OmniACK(task_id=task.task_id, status="ERROR", error_msg=str(e))

    def handle_wake_task(self, task: OmniWakeTask) -> OmniACK | None:
        "Handle deterministic Wakeup command from the main process"
        try:
            if isinstance(task, dict):
                task = OmniWakeTask(**task)
            self.wake_up(tags=task.tags)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            gc.collect()
            current_omni_platform.synchronize()
            free_now = current_omni_platform.get_free_memory(self.device)
            try:
                total_mem = current_omni_platform.get_device_total_memory()
            except (NotImplementedError, AttributeError):
                total_mem = torch.cuda.get_device_properties(self.device).total_memory
            current_used_gib = (total_mem - free_now) / 1024**3
            if self.rank != 0:
                return None
            current_stage_id = getattr(self.vllm_config.model_config, "stage_id", 0)
            ack = OmniACK(
                task_id=task.task_id,
                status="SUCCESS",
                stage_id=current_stage_id,
                rank=self.rank,
                metadata={"state": "WARM", "current_vram_gib": f"{current_used_gib:.2f}"},
            )
            if hasattr(self, "result_mq") and self.result_mq:
                self.result_mq.put(ack)
            logger.info(f"[LLM Worker {self.rank}] Wake-up ACK emitted.")
            return ack
        except Exception as e:
            logger.error(f"[LLM Worker {self.rank}] Wake-up Failed: {e}", exc_info=True)
            if torch.distributed.is_initialized():
                try:
                    torch.distributed.barrier()
                except Exception:
                    pass
            tid = task.task_id if hasattr(task, "task_id") else "unknown"
            return OmniACK(task_id=tid, status="ERROR", error_msg=str(e))

    def encoder_loaded(self) -> bool:
        """Check if encoder weights are loaded in the model.

        This method is exposed via collective_rpc to check encoder availability
        for models that support voice cloning with reference audio.

        Returns:
            bool: True if encoder weights are available, False otherwise.
        """
        if hasattr(self.model_runner, "model") and hasattr(self.model_runner.model, "encoder_loaded"):
            return self.model_runner.model.encoder_loaded()
        return False
