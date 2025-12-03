import gc
import os
from contextlib import contextmanager

import torch
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm.utils import GiB_bytes, MemorySnapshot
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_worker import Worker as GPUWorker
from vllm.v1.worker.gpu_worker import init_worker_distributed_environment

from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner

def _patch_torch_cuda_for_xpu():
    """Permanently patch torch.cuda APIs to use torch.xpu."""
    if not hasattr(torch, 'xpu'):
        return
    
    # Save originals
    if not hasattr(torch.cuda, '_original_Event'):
        torch.cuda._original_Event = torch.cuda.Event
        torch.cuda._original_Stream = torch.cuda.Stream
        torch.cuda._original_synchronize = torch.cuda.synchronize
        torch.cuda._original_current_stream = torch.cuda.current_stream
        torch.cuda._original_default_stream = torch.cuda.default_stream
        torch.cuda._original_stream = torch.cuda.stream
    
    # Apply patches
    torch.cuda.Event = torch.xpu.Event
    torch.cuda.Stream = torch.xpu.Stream
    torch.cuda.synchronize = torch.xpu.synchronize
    torch.cuda.current_stream = torch.xpu.current_stream
    torch.cuda.default_stream = torch.xpu.current_stream
    torch.cuda.stream = torch.xpu.stream

class GPUGenerationWorker(GPUWorker):
    """GPU Worker for Generation model (non-autoregressive waveform generation).

    Usage in stage config:
        worker_cls: "vllm_omni.worker.gpu_generation_model_runner.GPUGenerationModelRunner"
    """

    def init_device(self):
        """Initialize CUDA device and distributed environment."""
        if self.device_config.device.type == "cuda":
            # torch.distributed.all_reduce does not free the input tensor until
            # the synchronization point. This causes the memory usage to grow
            # as the number of all_reduce calls increases. This env var disables
            # this behavior.
            # Related issue:
            # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
            os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            self.device = torch.device(f"cuda:{self.local_rank}")
            current_platform.set_device(self.device)

            current_platform.check_if_supports_dtype(self.model_config.dtype)

            # Initialize the distributed environment BEFORE taking
            # memory snapshot
            # This ensures NCCL buffers are allocated before we measure
            # available memory
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
                current_platform.dist_backend,
            )

            # Set random seed.
            set_random_seed(self.model_config.seed)

            # Now take memory snapshot after NCCL is initialized
            gc.collect()
            torch.cuda.empty_cache()

            # take current memory snapshot
            self.init_snapshot = MemorySnapshot()
            self.requested_memory = self.init_snapshot.total_memory * self.cache_config.gpu_memory_utilization
            if self.init_snapshot.free_memory < self.requested_memory:

                def GiB(b):
                    return round(b / GiB_bytes, 2)

                raise ValueError(
                    f"Free memory on device "
                    f"({GiB(self.init_snapshot.free_memory)}/"
                    f"{GiB(self.init_snapshot.total_memory)} GiB) on startup "
                    f"is less than desired GPU memory utilization "
                    f"({self.cache_config.gpu_memory_utilization}, "
                    f"{GiB(self.requested_memory)} GiB). Decrease GPU memory "
                    f"utilization or reduce GPU memory used by other processes."
                )
        elif self.device_config.device.type == "xpu":
            from vllm.distributed import get_world_group
            
            self.device = torch.device(f"xpu:{self.local_rank}")
            current_platform.set_device(self.device)
            current_platform.check_if_supports_dtype(self.model_config.dtype)
            
            ENV_CCL_ATL_TRANSPORT = os.getenv("CCL_ATL_TRANSPORT", "ofi")
            ENV_LOCAL_WORLD_SIZE = os.getenv(
                "LOCAL_WORLD_SIZE", str(self.parallel_config.world_size)
            )
            os.environ["CCL_ATL_TRANSPORT"] = ENV_CCL_ATL_TRANSPORT
            os.environ["LOCAL_WORLD_SIZE"] = ENV_LOCAL_WORLD_SIZE
            os.environ["LOCAL_RANK"] = str(self.local_rank)

            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
                current_platform.dist_backend,
            )
            
            # global all_reduce needed for overall oneccl warm up
            torch.distributed.all_reduce(
                torch.zeros(1).xpu(), group=get_world_group().device_group
            )

            # Set random seed.
            set_random_seed(self.model_config.seed)
            # Now take memory snapshot after NCCL is initialized
            gc.collect()
            torch.xpu.empty_cache()
            
            self.init_snapshot = torch.xpu.get_device_properties(
                self.local_rank
            )
            self.requested_memory = self.init_snapshot.total_memory * self.cache_config.gpu_memory_utilization
            # if self.init_snapshot.free_memory < self.requested_memory:

            #     def GiB(b):
            #         return round(b / GiB_bytes, 2)

            #     raise ValueError(
            #         f"Free memory on device "
            #         f"({GiB(self.init_snapshot.free_memory)}/"
            #         f"{GiB(self.init_snapshot.total_memory)} GiB) on startup "
            #         f"is less than desired GPU memory utilization "
            #         f"({self.cache_config.gpu_memory_utilization}, "
            #         f"{GiB(self.requested_memory)} GiB). Decrease GPU memory "
            #         f"utilization or reduce GPU memory used by other processes."
            #     )
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        # Construct the model runner
        
        if self.device_config.device.type == "xpu":
            _patch_torch_cuda_for_xpu()

        self.model_runner: GPUGenerationModelRunner = GPUGenerationModelRunner(self.vllm_config, self.device)


        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)
