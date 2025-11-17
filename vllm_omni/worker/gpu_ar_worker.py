import gc
import os

import torch
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm.utils import GiB_bytes, MemorySnapshot
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_worker import Worker as GPUWorker
from vllm.v1.worker.gpu_worker import init_worker_distributed_environment

from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner


class GPUARWorker(GPUWorker):
    def init_device(self):
        if self.device_config.device.type == "cuda":
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
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        # Construct the model runner
        self.model_runner: GPUARModelRunner = GPUARModelRunner(self.vllm_config, self.device)

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)
