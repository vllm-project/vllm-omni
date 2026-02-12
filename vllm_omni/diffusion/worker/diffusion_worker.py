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
from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch
import zmq
from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.mem_utils import GiB_bytes
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_omni.diffusion.data import (
    DiffusionOutput,
    OmniDiffusionConfig,
)
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.profiler import CurrentProfiler
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.lora.request import LoRARequest
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


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
    ):
        self.local_rank = local_rank
        self.rank = rank
        self.od_config = od_config
        self.device: torch.device | None = None
        self.vllm_config: VllmConfig | None = None
        self.model_runner: DiffusionModelRunner | None = None
        self._sleep_saved_buffers: dict[str, torch.Tensor] = {}
        self.lora_manager: DiffusionLoRAManager | None = None
        self.init_device()
        # Create model runner
        self.model_runner = DiffusionModelRunner(
            vllm_config=self.vllm_config,
            od_config=self.od_config,
            device=self.device,
        )
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

        # Create vllm_config for parallel configuration
        vllm_config = VllmConfig(compilation_config=CompilationConfig())
        vllm_config.parallel_config.tensor_parallel_size = self.od_config.parallel_config.tensor_parallel_size
        vllm_config.parallel_config.data_parallel_size = self.od_config.parallel_config.data_parallel_size
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
            )
            init_workspace_manager(self.device)

    def load_model(self, load_format: str = "default", custom_pipeline_name: str | None = None) -> None:
        """Load the diffusion model using DiffusionModelRunner."""
        with (
            set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config),
            set_current_vllm_config(self.vllm_config),
        ):
            self.model_runner.load_model(
                memory_pool_context_fn=self._maybe_get_memory_pool_context,
                load_format=load_format,
                custom_pipeline_name=custom_pipeline_name,
            )

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

    @classmethod
    def start_profile(cls, trace_path_template: str) -> str:
        """Start profiling for this GPU worker."""
        return CurrentProfiler.start(trace_path_template)

    @classmethod
    def stop_profile(cls) -> dict | None:
        """Stop profiling and return the result dictionary."""
        return CurrentProfiler.stop()

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
        return self.model_runner.execute_model(req)

    def load_weights(self, weights) -> set[str]:
        """Load weights by delegating to the model runner."""
        assert self.model_runner is not None, "Model runner not initialized"
        return self.model_runner.load_weights(weights)

    def remove_lora(self, adapter_id: int) -> bool:
        return self.lora_manager.remove_adapter(adapter_id)

    def add_lora(self, lora_request: LoRARequest, lora_scale: float = 1.0) -> bool:
        return self.lora_manager.add_adapter(lora_request, lora_scale)

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

        free_bytes_before_sleep = current_omni_platform.get_free_memory()

        # Save the buffers before level 2 sleep
        if level == 2 and self.model_runner is not None:
            model = self.model_runner.pipeline
            self._sleep_saved_buffers = {name: buffer.cpu().clone() for name, buffer in model.named_buffers()}

        allocator = CuMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())
        free_bytes_after_sleep = current_omni_platform.get_free_memory()
        device_id = self.device.index if self.device.index is not None else 0
        total = current_omni_platform.get_device_total_memory(device_id)
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode freed %.2f GiB memory, %.2f GiB memory is still in use.",
            freed_bytes / GiB_bytes,
            used_bytes / GiB_bytes,
        )
        return True

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
        """
        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags)

        # Restore the buffers after level 2 sleep
        if len(self._sleep_saved_buffers) and self.model_runner is not None:
            model = self.model_runner.pipeline
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}
        return True

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        """Get memory pool context for sleep mode support."""
        if self.od_config.enable_sleep_mode:
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            if tag == "weights":
                assert allocator.get_current_usage() == 0, "Sleep mode can only be used for one instance per process."
            return allocator.use_memory_pool(tag=tag)
        else:
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
            torch.cuda.empty_cache()

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
        worker_extension_cls: str | None = None,
        custom_pipeline_args: dict[str, Any] | None = None,
    ):
        self.od_config = od_config
        self.gpu_id = gpu_id

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
            logger.info(f"Worker {gpu_id} created result MessageQueue")

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
        wrapper = WorkerWrapperBase(
            gpu_id=gpu_id,
            od_config=od_config,
            worker_extension_cls=worker_extension_cls,
            custom_pipeline_args=custom_pipeline_args,
        )
        return wrapper

    def return_result(self, output: DiffusionOutput):
        """Reply to client, only on rank 0."""
        if self.result_mq is not None:
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
                msg = self.recv_message()
            except Exception as e:
                logger.error(
                    f"Error receiving message in worker loop: {e}",
                    exc_info=True,
                )
                continue

            if msg is None or len(msg) == 0:
                logger.warning("Worker %s: Received empty payload, ignoring", self.gpu_id)
                continue

            # Route message based on type
            if isinstance(msg, dict) and msg.get("type") == "rpc":
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
        self.worker = worker_class(
            local_rank=gpu_id,
            rank=gpu_id,
            od_config=od_config,
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
