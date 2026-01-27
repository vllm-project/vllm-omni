# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Diffusion Worker for vLLM-Omni.

Handles GPU infrastructure initialization and delegates model operations
to GPUDiffusionModelRunner.
"""

import multiprocessing as mp
import os
from contextlib import AbstractContextManager, nullcontext

import torch
import zmq
from vllm.config import VllmConfig
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.logger import init_logger
from vllm.utils.mem_utils import GiB_bytes

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
from vllm_omni.diffusion.worker.gpu_diffusion_model_runner import GPUDiffusionModelRunner
from vllm_omni.lora.request import LoRARequest

logger = init_logger(__name__)


class GPUDiffusionWorker:
    """
    A worker that manages GPU infrastructure and delegates to the model runner.

    This class handles infrastructure initialization only:
    - Device setup (CUDA device selection)
    - Distributed environment (NCCL, model parallel)
    - Memory management (sleep/wake)

    All model-related operations (loading, compilation, execution) are
    delegated to GPUDiffusionModelRunner.
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
        self.model_runner: GPUDiffusionModelRunner | None = None
        self._sleep_saved_buffers: dict[str, torch.Tensor] = {}
        self.lora_manager: DiffusionLoRAManager | None = None
        self.init_device()

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
        self.device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(self.device)

        # Create vllm_config for parallel configuration
        vllm_config = VllmConfig()
        vllm_config.parallel_config.tensor_parallel_size = self.od_config.parallel_config.tensor_parallel_size
        vllm_config.parallel_config.data_parallel_size = self.od_config.parallel_config.data_parallel_size
        self.vllm_config = vllm_config

        # Initialize distributed environment
        with set_forward_context(vllm_config=vllm_config, omni_diffusion_config=self.od_config):
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

        # Create model runner and load model
        self.model_runner = GPUDiffusionModelRunner(
            vllm_config=self.vllm_config,
            od_config=self.od_config,
            device=self.device,
        )
        self.model_runner.load_model(
            memory_pool_context_fn=self._maybe_get_memory_pool_context,
        )
        assert self.model_runner.pipeline is not None
        self.lora_manager = DiffusionLoRAManager(
            pipeline=self.model_runner.pipeline,
            device=self.device,
            dtype=self.od_config.dtype,
            max_cached_adapters=self.od_config.max_cpu_loras,
            lora_path=self.od_config.lora_path,
            lora_scale=self.od_config.lora_scale,
        )
        logger.info(f"Worker {self.rank}: Initialization complete.")

    def generate(self, requests: list[OmniDiffusionRequest]) -> DiffusionOutput:
        """Generate output for the given requests."""
        return self.execute_model(requests, self.od_config)

    @classmethod
    def start_profile(cls, trace_path_template: str) -> str:
        """Start profiling for this GPU worker."""
        return CurrentProfiler.start(trace_path_template)

    @classmethod
    def stop_profile(cls) -> dict | None:
        """Stop profiling and return the result dictionary."""
        return CurrentProfiler.stop()

    def execute_model(self, reqs: list[OmniDiffusionRequest], od_config: OmniDiffusionConfig) -> DiffusionOutput:
        """Execute a forward pass by delegating to the model runner."""
        assert self.model_runner is not None, "Model runner not initialized"
        if self.lora_manager is not None and reqs:
            req = reqs[0]

            if len(reqs) > 1:
                # This worker (and the current diffusion model runner) applies
                # a single LoRA to the whole batch. Reject inconsistent LoRA
                # settings to avoid silently applying the wrong adapter.
                def _lora_key(r: OmniDiffusionRequest):
                    if r.lora_request is None:
                        return None
                    lr = r.lora_request
                    return (lr.lora_name, lr.lora_int_id, lr.lora_path, lr.tensorizer_config_dict)

                key0 = _lora_key(req)
                scale0 = req.lora_scale if key0 is not None else None
                for other in reqs[1:]:
                    if _lora_key(other) != key0:
                        raise ValueError("All requests in a diffusion batch must share the same LoRARequest.")
                    if key0 is not None and other.lora_scale != scale0:
                        raise ValueError("All requests in a diffusion batch must share the same lora_scale.")

            try:
                self.lora_manager.set_active_adapter(req.lora_request, req.lora_scale)
            except Exception as exc:
                if req.lora_request is not None:
                    raise
                logger.warning("LoRA activation skipped: %s", exc)
        return self.model_runner.execute_model(reqs)

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

        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]

        # Save the buffers before level 2 sleep
        if level == 2 and self.model_runner is not None:
            model = self.model_runner.pipeline
            self._sleep_saved_buffers = {name: buffer.cpu().clone() for name, buffer in model.named_buffers()}

        allocator = CuMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())
        free_bytes_after_sleep, total = torch.cuda.mem_get_info()
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


class WorkerProc:
    """Wrapper that runs one Worker in a separate process."""

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        gpu_id: int,
        broadcast_handle,
    ):
        self.od_config = od_config

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
        self.worker = self._create_worker(gpu_id, od_config)
        self.gpu_id = gpu_id
        self._running = True

    def _create_worker(self, gpu_id: int, od_config: OmniDiffusionConfig) -> GPUDiffusionWorker:
        """Create a worker instance. Override in subclasses for different worker types."""
        return GPUDiffusionWorker(
            local_rank=gpu_id,
            rank=gpu_id,
            od_config=od_config,
        )

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
            if isinstance(method, str):
                func = getattr(self.worker, method)
                result = func(*args, **kwargs)
            else:
                result = method(self.worker, *args, **kwargs)
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
                        self.return_result({"status": "error", "error": str(e)})

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
    ) -> None:
        """Worker initialization and execution loops."""
        worker_proc = WorkerProc(
            od_config,
            gpu_id=rank,
            broadcast_handle=broadcast_handle,
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
