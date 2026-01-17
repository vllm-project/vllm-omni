# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing as mp
import os
import time
from collections.abc import Iterable
from contextlib import AbstractContextManager, nullcontext

import torch
from vllm.config import LoadConfig, VllmConfig
from vllm.logger import init_logger
from vllm.utils.mem_utils import DeviceMemoryProfiler, GiB_bytes

from vllm_omni.diffusion.cache.selector import get_cache_backend
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
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.gpu_diffusion_worker import WorkerProc

logger = init_logger(__name__)


class NPUWorker:
    """
    A worker that executes the model on a single NPU.
    Inherits from GPUWorker and overrides device-specific initialization.
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
        self.pipeline = None
        self.device = None
        self._sleep_saved_buffers: dict[str, torch.Tensor] = {}
        self.init_device_and_model()

    def init_device_and_model(self) -> None:
        """Initialize the NPU device and load the model."""
        world_size = self.od_config.num_gpus
        rank = self.rank
        # Set environment variables for distributed initialization
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self.od_config.master_port)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        device = torch.device(f"npu:{rank}")
        torch.npu.set_device(device)

        # hack
        vllm_config = VllmConfig()
        vllm_config.parallel_config.tensor_parallel_size = self.od_config.parallel_config.tensor_parallel_size
        vllm_config.parallel_config.data_parallel_size = self.od_config.parallel_config.data_parallel_size
        self.vllm_config = vllm_config
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

            load_config = LoadConfig()
            model_loader = DiffusersPipelineLoader(load_config)
            time_before_load = time.perf_counter()
            with DeviceMemoryProfiler() as m:
                self.pipeline = model_loader.load_model(
                    od_config=self.od_config,
                    load_device=f"npu:{rank}",
                )
            time_after_load = time.perf_counter()

        logger.info(
            "Model loading took %.4f GiB and %.6f seconds",
            m.consumed_memory / GiB_bytes,
            time_after_load - time_before_load,
        )
        logger.info(f"Worker {self.rank}: Model loaded successfully.")

        # Setup cache backend based on type (both backends use enable()/reset() interface)
        self.cache_backend = get_cache_backend(self.od_config.cache_backend, self.od_config.cache_config)

        if self.cache_backend is not None:
            self.cache_backend.enable(self.pipeline)

    def generate(self, requests: list[OmniDiffusionRequest]) -> DiffusionOutput:
        """
        Generate output for the given requests.

        Args:
            requests: List of diffusion requests

        Returns:
            DiffusionOutput with generated results
        """
        return self.execute_model(requests, self.od_config)

    @torch.inference_mode()
    def execute_model(self, reqs: list[OmniDiffusionRequest], od_config: OmniDiffusionConfig) -> DiffusionOutput:
        """
        Execute a forward pass.
        """
        assert self.pipeline is not None
        if not reqs or len(reqs) == 0:
            raise ValueError("Cannot execute model with empty request list")
        # TODO: dealing with first req for now
        req = reqs[0]

        if req.generator is None and req.seed is not None:
            req.generator = torch.Generator(device=self.device).manual_seed(req.seed)

        # Refresh cache context if needed
        if self.cache_backend is not None and self.cache_backend.is_enabled():
            self.cache_backend.refresh(self.pipeline, req.num_inference_steps)
        with set_forward_context(vllm_config=self.vllm_config, omni_diffusion_config=self.od_config):
            output = self.pipeline.forward(req)
        return output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return self.pipeline.load_weights(weights)

    def sleep(self, level: int = 1) -> bool:
        """
        Put the worker to sleep. The worker should not process any requests.
        The caller should guarantee that no requests are being processed
        during the sleep period, before `wake_up` is called.

        Args:
            level: The sleep level. Level 1 sleep will offload the model
                weights and discard the kv cache.
                Currently only support level 1.
        """
        from vllm.device_allocator.cumem import CuMemAllocator

        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]

        # Save the buffers before level 2 sleep
        if level == 2:
            model = self.pipeline
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
        if len(self._sleep_saved_buffers):
            model = self.pipeline
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}
        return True

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        if self.od_config.enable_sleep_mode:
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            if tag == "weights":
                assert allocator.get_current_usage() == 0, "Sleep mode can only be used for one instance per process."
            return allocator.use_memory_pool(tag=tag)
        else:
            return nullcontext()

    def shutdown(self) -> None:
        destroy_distributed_env()


class NPUWorkerProc(WorkerProc):
    """Wrapper that runs one NPUWorker in a separate process."""

    def _create_worker(self, gpu_id: int, od_config: OmniDiffusionConfig) -> NPUWorker:
        """Create an NPUWorker instead of GPUWorker."""
        return NPUWorker(
            local_rank=gpu_id,
            rank=gpu_id,
            od_config=od_config,
        )

    @staticmethod
    def worker_main(
        rank: int,
        od_config: OmniDiffusionConfig,
        pipe_writer: mp.connection.Connection,
        broadcast_handle,
    ) -> None:
        """Worker initialization and execution loops."""

        worker_proc = NPUWorkerProc(
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
