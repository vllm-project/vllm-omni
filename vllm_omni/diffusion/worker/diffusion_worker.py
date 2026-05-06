# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Diffusion Worker for vLLM-Omni.

Handles GPU infrastructure initialization and delegates model operations
to DiffusionModelRunner.
"""

import gc
import multiprocessing as mp
import os
from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import zmq
from vllm.config import CompilationConfig, DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.logger import init_logger
from vllm.profiler.wrapper import CudaProfilerWrapper, WorkerProfiler
from vllm.transformers_utils.config import get_hf_text_config
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.mem_utils import GiB_bytes
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_omni.diffusion.data import (
    DiffusionOutput,
    OmniACK,
    OmniDiffusionConfig,
    OmniSleepTask,
    OmniWakeTask,
)
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.ipc import pack_diffusion_output_shm
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import DiffusionSchedulerOutput
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.utils import BaseRunnerOutput
from vllm_omni.lora.request import LoRARequest
from vllm_omni.platforms import current_omni_platform
from vllm_omni.profiler import OmniTorchProfilerWrapper, create_omni_profiler
from vllm_omni.worker.gpu_memory_utils import get_process_gpu_memory

logger = init_logger(__name__)


@dataclass
class _DiffusionVllmModelConfig:
    model: str
    dtype: torch.dtype
    quantization: str | None = None
    quantization_config: Any | None = None
    hf_config: Any | None = None
    hf_text_config: Any | None = None
    multimodal_config: Any | None = None
    enforce_eager: bool = False
    disable_cascade_attn: bool = False
    enable_return_routed_experts: bool = False
    is_moe: bool = False

    def is_quantized(self) -> bool:
        return self.quantization is not None

    def is_model_moe(self) -> bool:
        return self.is_moe

    def is_nvfp4_quantized(self) -> bool:
        return self.quantization == "modelopt_fp4"


def _make_diffusion_vllm_model_config(od_config: OmniDiffusionConfig) -> _DiffusionVllmModelConfig:
    quant_config = getattr(od_config, "quantization_config", None)
    quantization = quant_config.get_name() if quant_config is not None and hasattr(quant_config, "get_name") else None
    hf_config = getattr(od_config, "tf_model_config", None)
    hf_text_config = get_hf_text_config(hf_config) if hasattr(hf_config, "get_text_config") else hf_config
    return _DiffusionVllmModelConfig(
        model=od_config.model,
        dtype=od_config.dtype,
        quantization=quantization,
        quantization_config=quant_config,
        hf_config=hf_config,
        hf_text_config=hf_text_config,
        enforce_eager=getattr(od_config, "enforce_eager", False),
        is_moe=bool(getattr(od_config, "is_moe", False)),
    )


@contextmanager
def _force_cutlass_fp8_linear_kernel(quant_config: object | None) -> Iterator[None]:
    from vllm.model_executor.layers.quantization import modelopt as vllm_modelopt

    linear_method_cls = getattr(quant_config, "LinearMethodCls", None)
    if linear_method_cls in {
        vllm_modelopt.ModelOptFp8LinearMethod,
        vllm_modelopt.ModelOptFp8PcPtLinearMethod,
    }:
        from vllm.platforms import current_platform

        if current_platform.is_cuda() and current_platform.has_device_capability(89):
            from vllm.model_executor.kernels.linear import CutlassFP8ScaledMMLinearKernel

            original_init_fp8_linear_kernel = vllm_modelopt.init_fp8_linear_kernel

            def init_fp8_linear_kernel_with_cutlass(*args: Any, **kwargs: Any) -> Any:
                kwargs.setdefault("force_kernel", CutlassFP8ScaledMMLinearKernel)
                return original_init_fp8_linear_kernel(*args, **kwargs)

            vllm_modelopt.init_fp8_linear_kernel = init_fp8_linear_kernel_with_cutlass
            logger.info("Using CUTLASS FP8 linear kernels for this ModelOpt FP8 diffusion stage.")
            try:
                yield
            finally:
                vllm_modelopt.init_fp8_linear_kernel = original_init_fp8_linear_kernel
            return

    yield


def _is_unexpected_additional_config_type_error(exc: TypeError) -> bool:
    """Return True only for constructor rejections of the additional_config kwarg."""
    message = str(exc)
    return "unexpected keyword argument" in message and "additional_config" in message


def _create_diffusion_worker_vllm_config(device: torch.device, od_config: OmniDiffusionConfig) -> VllmConfig:
    """Create a worker-local VllmConfig while preserving additional_config when supported."""
    config_kwargs: dict[str, Any] = {
        "compilation_config": CompilationConfig(),
        "device_config": DeviceConfig(device=device),
    }
    if od_config.additional_config:
        config_kwargs["additional_config"] = od_config.additional_config

    try:
        return VllmConfig(**config_kwargs)
    except TypeError as exc:
        if not _is_unexpected_additional_config_type_error(exc):
            raise

        logger.debug("Worker-local VllmConfig does not accept additional_config in constructor: %s", exc)
        config_kwargs.pop("additional_config", None)
        vllm_config = VllmConfig(**config_kwargs)
        try:
            setattr(vllm_config, "additional_config", dict(od_config.additional_config))
        except Exception as set_exc:  # pragma: no cover - defensive for older vLLM builds
            logger.warning("Failed to attach additional_config to worker VllmConfig: %s", set_exc)
        return vllm_config


class DiffusionWorker:
    """
    A worker that manages GPU infrastructure and delegates to the model runner.

    This class handles infrastructure initialization only:
    - Device setup (CUDA device selection)
    - Distributed environment (NCCL, model parallel)
    - Memory management (sleep/wake)

    All model-related operations (loading, compilation, execution) are
    delegated to DiffusionModelRunner.
    """

    def __init__(
        self,
        local_rank: int,
        rank: int,
        od_config: OmniDiffusionConfig,
        skip_load_model: bool = False,
    ):
        self.local_rank = local_rank
        self.rank = rank
        self.od_config = od_config
        self.device: torch.device | None = None
        self.vllm_config: VllmConfig | None = None
        self.model_runner: DiffusionModelRunner | None = None
        self._sleep_saved_buffers: dict[str, torch.Tensor] = {}
        self.lora_manager: DiffusionLoRAManager | None = None
        self.stage_id = getattr(od_config, "stage_id", 0)
        self.init_device()
        # Create model runner using the platform-specified class
        model_runner_cls_path = current_omni_platform.get_diffusion_model_runner_cls()
        model_runner_cls = resolve_obj_by_qualname(model_runner_cls_path)
        self.model_runner = model_runner_cls(
            vllm_config=self.vllm_config,
            od_config=self.od_config,
            device=self.device,
        )
        self.profiler: WorkerProfiler | None = self._create_profiler()
        if not skip_load_model:
            self.load_model(load_format=self.od_config.diffusion_load_format)
            self.init_lora_manager()
        logger.info(f"Worker {self.rank}: Initialization complete.")

    def init_device(self) -> None:
        """Initialize the device and distributed environment."""
        world_size = self.od_config.num_gpus
        rank = self.rank

        # Set environment variables for distributed initialization
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self.od_config.master_port)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        # Setup device
        self.device = current_omni_platform.get_torch_device(rank)
        current_omni_platform.set_device(self.device)

        # Create vllm_config for parallel configuration. Pass explicit device_config
        # so DeviceConfig does not rely on current_platform in worker subprocesses.
        vllm_config = _create_diffusion_worker_vllm_config(self.device, self.od_config)
        vllm_config.parallel_config.tensor_parallel_size = self.od_config.parallel_config.tensor_parallel_size
        vllm_config.parallel_config.data_parallel_size = self.od_config.parallel_config.data_parallel_size
        vllm_config.parallel_config.enable_expert_parallel = self.od_config.parallel_config.enable_expert_parallel
        vllm_config.profiler_config = self.od_config.profiler_config
        vllm_config.model_config = _make_diffusion_vllm_model_config(self.od_config)  # type: ignore[assignment]
        vllm_config.quant_config = self.od_config.quantization_config
        # Since vLLM v0.20.0, IR wraps GPU ops. Set IR op priority preference to enforce GPU op fusion during wrapping.
        # Also need to log, because vLLM internally logs another line in VllmConfig.__post_init__. Avoid confusion.
        vllm_config.kernel_config.ir_op_priority = current_omni_platform.get_default_ir_op_priority(vllm_config)
        logger.info(
            "Final IR op priority after setting vLLM-Omni overrides: %s", vllm_config.kernel_config.ir_op_priority
        )
        self.vllm_config = vllm_config

        # Initialize distributed environment
        with (
            set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config),
            set_current_vllm_config(self.vllm_config),
        ):
            init_distributed_environment(world_size=world_size, rank=rank)
            logger.info(f"Worker {self.rank}: Initialized device and distributed environment.")

            parallel_config = self.od_config.parallel_config
            initialize_model_parallel(
                data_parallel_size=parallel_config.data_parallel_size,
                cfg_parallel_size=parallel_config.cfg_parallel_size,
                sequence_parallel_size=parallel_config.sequence_parallel_size,
                ulysses_degree=parallel_config.ulysses_degree,
                ring_degree=parallel_config.ring_degree,
                tensor_parallel_size=parallel_config.tensor_parallel_size,
                pipeline_parallel_size=parallel_config.pipeline_parallel_size,
                fully_shard_degree=parallel_config.hsdp_shard_size if parallel_config.use_hsdp else 1,
                hsdp_replicate_size=parallel_config.hsdp_replicate_size if parallel_config.use_hsdp else 1,
                enable_expert_parallel=parallel_config.enable_expert_parallel,
            )
            init_workspace_manager(self.device)

    def _create_profiler(self) -> WorkerProfiler | None:
        profiler_config = self.od_config.profiler_config
        profiler_type = getattr(profiler_config, "profiler", None)
        if profiler_type == "torch":
            return create_omni_profiler(
                profiler_config=profiler_config,
                worker_name=f"diffusion_rank{self.rank}",
                local_rank=self.local_rank,
            )
        if profiler_type == "cuda":
            return CudaProfilerWrapper(profiler_config)
        if profiler_type is not None:
            logger.warning("Unknown profiler backend %r on diffusion worker %s", profiler_type, self.rank)
        return None

    def _get_profiler(self) -> WorkerProfiler | None:
        return getattr(self, "profiler", None)

    def load_model(self, load_format: str = "default", custom_pipeline_name: str | None = None, **kwargs) -> None:
        """Load the diffusion model using DiffusionModelRunner."""
        load_format = kwargs.get("load_format", load_format)
        custom_pipeline_name = kwargs.get("custom_pipeline_name", custom_pipeline_name)
        cutlass_fp8_context = (
            _force_cutlass_fp8_linear_kernel(self.od_config.quantization_config)
            if getattr(self.od_config, "force_cutlass_fp8", False)
            else nullcontext()
        )
        with (
            set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config),
            set_current_vllm_config(self.vllm_config),
            cutlass_fp8_context,
        ):
            self.model_runner.load_model(
                memory_pool_context_fn=self._maybe_get_memory_pool_context,
                load_format=load_format,
                custom_pipeline_name=custom_pipeline_name,
            )
            current_omni_platform.synchronize()
            gc.collect()
        process_memory = get_process_gpu_memory(self.local_rank)
        if process_memory is not None:
            logger.info(
                "Worker %d: Process-scoped GPU memory after model loading: %.2f GiB.",
                self.rank,
                process_memory / GiB_bytes,
            )

        # When load_format is "dummy", pipeline will init with custom pipeline later
        if load_format != "dummy":
            assert self.model_runner.pipeline is not None

    def init_lora_manager(self) -> None:
        """Initialize the LoRA manager for this worker."""
        if self.model_runner.pipeline is None:
            return
        self.lora_manager = DiffusionLoRAManager(
            pipeline=self.model_runner.pipeline,
            device=self.device,
            dtype=self.od_config.dtype,
            max_cached_adapters=self.od_config.max_cpu_loras,
            lora_path=self.od_config.lora_path,
            lora_scale=self.od_config.lora_scale,
        )

    def generate(self, request: OmniDiffusionRequest) -> DiffusionOutput:
        """Generate output for the given requests."""
        return self.execute_model(request, self.od_config)

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        """Start or stop profiling for this GPU worker.

        Args:
            is_start: True to start profiling, False to stop.
            profile_prefix: Optional prefix for trace filename.
        """
        profiler = self._get_profiler()
        if profiler is None:
            return

        if is_start:
            if isinstance(profiler, OmniTorchProfilerWrapper):
                import time

                filename = profile_prefix or f"diffusion_rank{self.rank}_{int(time.time())}"
                profiler.set_trace_filename(filename)
            profiler.start()
        else:
            profiler.stop()

    def execute_model(self, req: OmniDiffusionRequest, od_config: OmniDiffusionConfig) -> DiffusionOutput:
        """Execute a forward pass by delegating to the model runner."""
        assert self.model_runner is not None, "Model runner not initialized"
        if self.lora_manager is not None:
            try:
                self.lora_manager.set_active_adapter(req.sampling_params.lora_request, req.sampling_params.lora_scale)
            except Exception as exc:
                if req.sampling_params.lora_request is not None:
                    raise
                logger.warning("LoRA activation skipped: %s", exc)
        profiler = self._get_profiler()
        ctx = profiler.annotate_context_manager("diffusion_forward") if profiler else nullcontext()
        with ctx:
            output = self.model_runner.execute_model(req)
        if profiler:
            profiler.step()
        return output

    def execute_stepwise(self, scheduler_output: DiffusionSchedulerOutput) -> BaseRunnerOutput:
        """Execute one diffusion step by delegating to the model runner."""
        assert self.model_runner is not None, "Model runner not initialized"
        if self.lora_manager is not None:
            # Step mode does not support LoRA yet. Clear any previously active
            # adapter first so worker-local LoRA state cannot leak in.
            self.lora_manager.set_active_adapter(None)

        if any(new_req.req.sampling_params.lora_request is not None for new_req in scheduler_output.scheduled_new_reqs):
            raise ValueError("Step mode does not support LoRA yet.")
        profiler = self._get_profiler()
        ctx = profiler.annotate_context_manager("diffusion_step") if profiler else nullcontext()
        with ctx:
            output = self.model_runner.execute_stepwise(scheduler_output)
        if profiler:
            profiler.step()
        return output

    def load_weights(self, weights) -> set[str]:
        """Load weights by delegating to the model runner."""
        assert self.model_runner is not None, "Model runner not initialized"
        return self.model_runner.load_weights(weights)

    def remove_lora(self, adapter_id: int) -> bool:
        return self.lora_manager.remove_adapter(adapter_id)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        # NOTE (Alex): We have not implemented the API routing
        # for the frontend server yet.
        return self.lora_manager.add_adapter(lora_request)

    def list_loras(self) -> list[int]:
        return self.lora_manager.list_adapters()

    def pin_lora(self, adapter_id: int) -> bool:
        return self.lora_manager.pin_adapter(adapter_id)

    def sleep(self, level: int = 1) -> bool:
        """
        Put the worker to sleep, offloading model weights.

        Args:
            level: Sleep level. Level 1 offloads weights, level 2 also saves buffers.
        """
        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()

        usage_before = allocator.get_current_usage()

        if level == 2 and self.model_runner is not None:
            if hasattr(self.model_runner, "graph_runners"):
                self.model_runner.graph_runners.clear()
                logger.info(f"[Worker {self.rank}] CUDA Graphs cleared.")
            model = self.model_runner.pipeline
            self._sleep_saved_buffers = {name: buffer.cpu().clone() for name, buffer in model.named_buffers()}

        free_mem_before = current_omni_platform.get_free_memory()

        # Level 1: Offload weights; Level 2: Total Discard
        offload_tags = ("weights",) if level == 1 else tuple()
        allocator.sleep(offload_tags=offload_tags)

        current_omni_platform.empty_cache()
        current_omni_platform.synchronize()

        free_mem_after = current_omni_platform.get_free_memory()
        try:
            total_mem = current_omni_platform.get_device_total_memory()
        except (NotImplementedError, AttributeError):
            total_mem = torch.cuda.get_device_properties(self.device).total_memory

        phys_freed_bytes = max(0, free_mem_after - free_mem_before)
        phys_used_bytes = total_mem - free_mem_after

        if usage_before > 0:
            logger.info(
                f"[Diffusion Worker {self.rank}] Sleep Level {level}: "
                f"physically freed {phys_freed_bytes / GiB_bytes:.2f} GiB, "
                f"{phys_used_bytes / GiB_bytes:.2f} GiB is still in use."
            )
        else:
            logger.info(f"[Worker {self.rank}] Sleep Level {level} completed (GPU was already empty).")
        logger.info(f"[Worker {self.rank}] Memory usage before sleep: {usage_before / GiB_bytes:.2f} GiB.")
        return usage_before

    def wake_up(self, tags: list[str] | None = None) -> bool:
        """
        Wake up the worker from sleep mode.

        Re-activates the memory allocator for the specified tags and restores
        model buffers from CPU back to GPU if they were saved during Level 2 sleep.

        Args:
            tags: List of memory pool tags to re-activate (e.g., ["weights"]
                  to match Level 1 sleep). If None, all pools are re-activated.
        """
        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags)
        current_omni_platform.synchronize()
        if len(self._sleep_saved_buffers) and self.model_runner is not None:
            model = self.model_runner.pipeline
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}
            logger.info(f"[Worker {self.rank}] Buffers restored from CPU.")
        logger.info(f"[Worker {self.rank}] Wake-up complete.")
        return True

    def handle_sleep_task(self, task: OmniSleepTask) -> OmniACK:
        from vllm_omni.platforms import current_omni_platform

        try:
            if isinstance(task, dict):
                task = OmniSleepTask(**task)
            logger.info(f"[Worker {self.rank}] Handshake Received: Task {task.task_id}")

            current_omni_platform.synchronize()
            usage_before = current_omni_platform.get_current_memory_usage(self.device)
            self.sleep(level=task.level)
            current_omni_platform.synchronize()
            usage_after = current_omni_platform.get_current_memory_usage(self.device)
            real_freed = max(0, usage_before - usage_after)
            logger.info(f"[Worker {self.rank}] Preparing ACK: freed_bytes={real_freed / GiB_bytes:.2f} GiB.")

            # Ensure all ranks have completed sleep before measuring memory and sending ACK
            if torch.distributed.is_initialized():
                t_freed = torch.tensor([float(real_freed)], device=self.device)
                torch.distributed.all_reduce(t_freed)
                real_freed = int(t_freed.item())

            if self.rank != 0:
                return None

            ack = OmniACK(
                task_id=task.task_id,
                status="SUCCESS",
                stage_id=self.stage_id,
                rank=self.rank,
                freed_bytes=real_freed,
                # return RL need metadata
                metadata={
                    "source": f"Platform_{current_omni_platform.get_device_name()}",
                    "total_freed_gib": f"{real_freed / GiB_bytes:.2f}",
                    "rank_residual_gib": f"{usage_after / GiB_bytes:.2f}",
                },
            )
            logger.info(f"[Worker {self.rank}] ACK emitted. Freed {real_freed / GiB_bytes:.2f} GiB.")
            return ack
        except Exception as e:
            logger.error(f"Sleep failed: {e}", exc_info=True)
            if torch.distributed.is_initialized():
                try:
                    torch.distributed.barrier()
                except Exception:
                    pass
            return OmniACK(task_id=task.task_id, status="ERROR", error_msg=str(e))

    def handle_wake_task(self, task: OmniWakeTask) -> OmniACK:
        from vllm_omni.platforms import current_omni_platform

        try:
            if isinstance(task, dict):
                task = OmniWakeTask(**task)
            logger.info(f"[Worker {self.rank}] Responding to Wake-up Task: {task.task_id}")
            self.wake_up(tags=task.tags)

            logger.info(f"[Worker {self.rank}] wake_up logic finished, entering barrier...")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            current_omni_platform.synchronize()
            usage_now = current_omni_platform.get_current_memory_usage(self.device)
            current_used_gib = usage_now / (1024**3)

            if self.rank != 0:
                return None
            logger.info(f"[Worker {self.rank}] PASSED barrier, about to return to loop.")

            return OmniACK(
                task_id=task.task_id,
                status="SUCCESS",
                stage_id=self.stage_id,
                rank=self.rank,
                metadata={
                    "state": "WARM",
                    "source": f"Platform_{current_omni_platform.get_device_name()}",
                    "current_vram_gib": f"{current_used_gib:.2f}",
                },
            )
        except Exception as e:
            logger.error(f"Wake-up failed on Rank {self.rank}: {e}", exc_info=True)
            if torch.distributed.is_initialized():
                try:
                    torch.distributed.barrier()
                except Exception:
                    pass
            return OmniACK(task_id=task.task_id, status="ERROR", error_msg=str(e))

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        """Get memory pool context for sleep mode support."""
        is_sleep_enabled = getattr(self.od_config, "enable_sleep_mode", False)
        if is_sleep_enabled:
            current_omni_platform.synchronize()
            gc.collect()
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            if tag == "weights":
                assert allocator.get_current_usage() == 0, "Sleep mode can only be used for one instance per process."
            logger.info(f"[Worker {self.rank}] Activating Diffusion CuMem pool for tag: {tag}")
            return allocator.use_memory_pool(tag=tag)
        return nullcontext()

    def shutdown(self) -> None:
        """Shutdown the worker and cleanup distributed environment."""
        destroy_distributed_env()


class CustomPipelineWorkerExtension:
    def re_init_pipeline(self, custom_pipeline_args: dict[str, Any]) -> None:
        """
        Re-initialize the pipeline with custom arguments.

        Args:
            custom_pipeline_args: Dictionary of arguments for custom pipeline initialization
        """

        # Clean up old pipeline
        if self.model_runner.pipeline is not None:
            del self.model_runner.pipeline
            gc.collect()
            torch.accelerator.empty_cache()

        # Get custom pipeline class name
        custom_pipeline_name = custom_pipeline_args["pipeline_class"]

        # Use the DiffusionWorker's load_model method which handles the forward context
        self.load_model(
            load_format="custom_pipeline",
            custom_pipeline_name=custom_pipeline_name,
        )
        self.init_lora_manager()


class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        gpu_id: int,
        broadcast_handle,
        wake_event: mp.Event,
        worker_extension_cls: str | None = None,
        custom_pipeline_args: dict[str, Any] | None = None,
    ):
        self.od_config = od_config
        self.gpu_id = gpu_id
        self.wake_event = wake_event

        # Inter-process Communication
        self.context = zmq.Context(io_threads=2)

        # Initialize MessageQueue reader from handle
        self.mq = MessageQueue.create_from_handle(broadcast_handle, gpu_id)

        self.result_mq = None
        self.result_mq_handle = None

        # Setup result sender (only for rank 0)
        if gpu_id == 0:
            self.result_mq = MessageQueue(n_reader=1, n_local_reader=1, local_reader_ranks=[0])
            self.result_mq_handle = self.result_mq.export_handle()
            WorkerProc._shared_result_handle = self.result_mq_handle
            logger.info(f"Worker {gpu_id} created result MessageQueue")
        else:
            handle = getattr(WorkerProc, "_shared_result_handle", None)
            if handle:
                self.result_mq = MessageQueue.create_from_handle(handle, gpu_id)
                logger.info(f"Worker {gpu_id} attached to shared result MessageQueue")

        assert od_config.master_port is not None

        # Create worker using WorkerWrapperBase for extension support
        self.worker = self._create_worker(gpu_id, od_config, worker_extension_cls, custom_pipeline_args)
        self._running = True

    def _create_worker(
        self,
        gpu_id: int,
        od_config: OmniDiffusionConfig,
        worker_extension_cls: str | None,
        custom_pipeline_args: dict[str, Any] | None = None,
    ) -> DiffusionWorker:
        """Create a worker instance. Override in subclasses for different worker types."""
        worker_cls_path = current_omni_platform.get_diffusion_worker_cls()
        base_worker_class = resolve_obj_by_qualname(worker_cls_path)
        wrapper = WorkerWrapperBase(
            gpu_id=gpu_id,
            od_config=od_config,
            worker_extension_cls=worker_extension_cls,
            custom_pipeline_args=custom_pipeline_args,
            base_worker_class=base_worker_class,
        )
        return wrapper

    def return_result(self, output: Any):
        """Reply to client, only on rank 0."""
        if self.result_mq is not None:
            if isinstance(output, OmniACK):
                self.result_mq.enqueue(output)
                return
            try:
                pack_diffusion_output_shm(output)
            except Exception as e:
                if hasattr(output, "output"):
                    logger.warning("SHM pack failed for model output: %s", e)
            self.result_mq.enqueue(output)

    def recv_message(self):
        """Receive messages from broadcast queue."""
        return self.mq.dequeue(indefinite=True)

    def execute_rpc(self, rpc_request: dict) -> tuple[object | None, bool]:
        """Execute an RPC request and indicate whether to reply."""
        method = rpc_request["method"]
        args = rpc_request.get("args", ())
        kwargs = rpc_request.get("kwargs", {})
        output_rank = rpc_request.get("output_rank")
        exec_all_ranks = rpc_request.get("exec_all_ranks", False)

        should_execute = exec_all_ranks or output_rank is None or output_rank == self.gpu_id
        should_reply = (output_rank is None or output_rank == self.gpu_id) and self.result_mq is not None

        if not should_execute:
            return None, False

        try:
            # Use execute_method from WorkerWrapperBase for consistent method resolution
            result = self.worker.execute_method(method, *args, **kwargs)
            return result, should_reply
        except Exception as e:
            logger.error(f"Error executing RPC: {e}", exc_info=True)
            raise e

    def worker_busy_loop(self) -> None:
        """Main busy loop for Multiprocessing Workers."""
        logger.info(f"Worker {self.gpu_id} ready to receive requests via shared memory")

        while self._running:
            msg = None
            try:
                msg = self.mq.dequeue(timeout=1.0)
            except Exception:
                if self.wake_event and self.wake_event.is_set():
                    self.wake_event.clear()
                    logger.info(f"Worker {self.gpu_id} caught OOB POKE, forcing wake-up sequence.")
                    msg = {"type": "wake_up", "task_id": "recovery-task", "tags": None}
                else:
                    continue
            if msg is None:
                continue

            if msg is None or len(msg) == 0:
                logger.warning("Worker %s: Received empty payload, ignoring", self.gpu_id)
                continue

            if isinstance(msg, dict) and msg.get("type") == "sleep":
                task = OmniSleepTask(level=msg.get("level", 2), task_id=msg.get("task_id", "local"))
                ack = self.worker.handle_sleep_task(task)
                self.return_result(ack)
            elif isinstance(msg, dict) and msg.get("type") == "wake_up":
                task = OmniWakeTask(tags=msg.get("tags"), task_id=msg.get("task_id", "local"))
                ack = self.worker.handle_wake_task(task)
                self.return_result(ack)
            # Route message based on type
            elif isinstance(msg, dict) and msg.get("type") == "rpc":
                try:
                    result, should_reply = self.execute_rpc(msg)
                    if should_reply:
                        self.return_result(result)
                except Exception as e:
                    logger.error(f"Error processing RPC: {e}", exc_info=True)
                    if self.result_mq is not None:
                        self.return_result(DiffusionOutput(error=str(e)))

            elif isinstance(msg, dict) and msg.get("type") == "shutdown":
                logger.info("Worker %s: Received shutdown message", self.gpu_id)
                self._running = False
                continue

            else:
                # Handle generation request
                try:
                    output = self.worker.execute_model(msg, self.od_config)
                except Exception as e:
                    logger.error(
                        f"Error executing forward in event loop: {e}",
                        exc_info=True,
                    )
                    output = DiffusionOutput(error=str(e))

                try:
                    self.return_result(output)
                except zmq.ZMQError as e:
                    logger.error(f"ZMQ error sending reply: {e}")
                    continue

        logger.info("event loop terminated.")
        try:
            self.worker.shutdown()
        except Exception as exc:
            logger.warning("Worker %s: Shutdown encountered an error: %s", self.gpu_id, exc)
        self.context.term()

    @staticmethod
    def worker_main(
        rank: int,
        od_config: OmniDiffusionConfig,
        pipe_writer: mp.connection.Connection,
        broadcast_handle,
        wake_event: mp.Event,
        worker_extension_cls: str | None = None,
        custom_pipeline_args: dict[str, Any] | None = None,
    ) -> None:
        """Worker initialization and execution loops."""
        from vllm_omni.plugins import load_omni_general_plugins

        load_omni_general_plugins()
        worker_proc = WorkerProc(
            od_config,
            gpu_id=rank,
            broadcast_handle=broadcast_handle,
            wake_event=wake_event,
            worker_extension_cls=worker_extension_cls,
            custom_pipeline_args=custom_pipeline_args,
        )
        logger.info(f"Worker {rank}: Scheduler loop started.")
        pipe_writer.send(
            {
                "status": "ready",
                "result_handle": worker_proc.result_mq_handle if rank == 0 else None,
            }
        )
        worker_proc.worker_busy_loop()
        logger.info(f"Worker {rank}: Shutdown complete.")


class WorkerWrapperBase:
    """
    Wrapper base class that creates DiffusionWorker with optional worker_extension_cls support.
    This enables dynamic inheritance for DiffusionWorker to extend with custom functionality.
    """

    def __init__(
        self,
        gpu_id: int,
        od_config: OmniDiffusionConfig,
        base_worker_class: type = DiffusionWorker,
        wake_event: mp.Event = None,
        worker_extension_cls: str | None = None,
        custom_pipeline_args: dict[str, Any] | None = None,
    ):
        """
        Initialize WorkerWrapperBase with support for worker extensions.

        Args:
            gpu_id: GPU device ID
            od_config: OmniDiffusionConfig configuration
            worker_extension_cls: Optional qualified name of worker extension class
            custom_pipeline_args: Optional arguments for custom pipeline initialization
        """
        self.gpu_id = gpu_id
        self.od_config = od_config
        self.base_worker_class = base_worker_class
        self.worker_extension_cls = worker_extension_cls
        self.custom_pipeline_args = custom_pipeline_args

        # Prepare worker class with extension support
        worker_class = self._prepare_worker_class()

        # Create the actual worker instance
        # When custom_pipeline_args is provided, skip initial model loading
        # since re_init_pipeline will handle it. This avoids allocating memory
        # through CuMemAllocator twice, which causes assertion failures in
        # sleep mode.
        self.worker = worker_class(
            local_rank=gpu_id,
            rank=gpu_id,
            od_config=od_config,
            skip_load_model=(self.custom_pipeline_args is not None),
        )

        # Re-initialize pipeline with custom pipeline if provided
        if self.custom_pipeline_args is not None:
            self.worker.re_init_pipeline(self.custom_pipeline_args)

    def _prepare_worker_class(self) -> type:
        """
        Prepare the worker class with optional extension.
        Dynamically extends GPUWorker with worker_extension_cls if provided.

        Returns:
            The worker class (potentially extended)
        """
        worker_class = self.base_worker_class

        # If custom_pipeline_args is provided, use CustomPipelineWorkerExtension
        if self.custom_pipeline_args is not None:
            # Set worker_extension_cls to CustomPipelineWorkerExtension if not already set
            if self.worker_extension_cls is None:
                self.worker_extension_cls = CustomPipelineWorkerExtension

        if self.worker_extension_cls:
            if isinstance(self.worker_extension_cls, str):
                worker_extension_cls = resolve_obj_by_qualname(self.worker_extension_cls)
            else:
                worker_extension_cls = self.worker_extension_cls
            extended_calls = []

            if worker_extension_cls not in worker_class.__bases__:
                # Check for conflicts between worker and extension
                for attr in dir(worker_extension_cls):
                    if attr.startswith("__"):
                        continue
                    if hasattr(worker_class, attr):
                        logger.warning(
                            f"Worker class {worker_class} already has attribute "
                            f"{attr}, which may conflict with worker extension "
                            f"class {worker_extension_cls}."
                        )
                    if callable(getattr(worker_extension_cls, attr)):
                        extended_calls.append(attr)

                # Dynamically inherit the worker extension class
                class_name = f"{worker_class.__name__}With{worker_extension_cls.__name__}"
                worker_class = type(class_name, (worker_extension_cls, worker_class), {})
                logger.info(
                    "Created extended worker class %s from %s for extended calls %s",
                    class_name,
                    worker_extension_cls,
                    extended_calls,
                )

        return worker_class

    def generate(self, requests: list[OmniDiffusionRequest]) -> DiffusionOutput:
        """
        Generate output for the given requests.

        Args:
            requests: List of diffusion requests

        Returns:
            DiffusionOutput with generated results
        """
        return self.worker.generate(requests)

    def execute_model(self, reqs: list[OmniDiffusionRequest], od_config: OmniDiffusionConfig) -> DiffusionOutput:
        """
        Execute a forward pass.

        Args:
            reqs: List of diffusion requests
            od_config: OmniDiffusionConfig configuration

        Returns:
            DiffusionOutput with generated results
        """
        return self.worker.execute_model(reqs, od_config)

    def execute_stepwise(self, scheduler_output: DiffusionSchedulerOutput) -> BaseRunnerOutput:
        """Execute one diffusion step."""
        return self.worker.execute_stepwise(scheduler_output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load model weights.

        Args:
            weights: Iterable of (name, tensor) tuples

        Returns:
            Set of loaded weight names
        """
        return self.worker.load_weights(weights)

    def sleep(self, level: int = 1) -> bool:
        """
        Put the worker to sleep. The worker should not process any requests.
        The caller should guarantee that no requests are being processed
        during the sleep period, before `wake_up` is called.

        Args:
            level: The sleep level. Level 1 sleep will offload the model
                weights and discard the kv cache.
                Currently only support level 1.

        Returns:
            True on success
        """
        return self.worker.sleep(level)

    def wake_up(self, tags: list[str] | None = None) -> bool:
        """
        Wake up the worker from sleep mode. See the sleep function
        method for more details.

        Args:
            tags: An optional list of tags to reallocate the worker memory
                for specific memory allocations. Values must be in
                `("weights")`. If None, all memory is reallocated.
                wake_up should be called with all tags (or None) before the
                worker is used again.

        Returns:
            True on success
        """
        return self.worker.wake_up(tags)

    def handle_sleep_task(self, task):
        return self.worker.handle_sleep_task(task)

    def handle_wake_task(self, task):
        return self.worker.handle_wake_task(task)

    def shutdown(self) -> None:
        """Shutdown the worker and cleanup resources."""
        return self.worker.shutdown()

    def execute_method(self, method: str | bytes, *args, **kwargs) -> Any:
        """
        Execute a method on the worker.

        Args:
            method: Method name (str) or serialized callable (bytes)

        Returns:
            Result of the method execution (type depends on the method)

        Raises:
            Exception: If method execution fails
        """
        try:
            # Method resolution order:
            # 1. If method is defined in this class, it will be called directly
            # 2. Otherwise, since we define `__getattr__` and redirect attribute
            #    query to `self.worker`, the method will be called on the worker
            assert isinstance(method, str), "Method must be str"
            func = getattr(self.worker, method)
            return func(*args, **kwargs)

        except Exception as e:
            msg = f"Error executing method {method!r}. This might cause issues in distributed execution."
            logger.exception(msg)
            raise e

    def __getattr__(self, attr: str):
        """Delegate attribute access to the wrapped worker."""
        return getattr(self.worker, attr)
