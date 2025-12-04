# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import os

import torch
from vllm.model_executor import set_random_seed
from vllm.platforms import current_platform
from vllm.utils import GiB_bytes
from vllm.v1.worker.gpu_worker import init_worker_distributed_environment
from vllm.v1.worker.xpu_worker import XPUWorker

from vllm_omni.worker.xpu.xpu_ar_model_runner import XPUARModelRunner


class XPUARWorker(XPUWorker):
    """XPU AR worker for thinker/talker stages in Omni model."""

    def init_device(self):
        from vllm.distributed import get_world_group

        self.device = torch.device(f"xpu:{self.local_rank}")
        current_platform.set_device(self.device)
        current_platform.check_if_supports_dtype(self.model_config.dtype)

        self.init_gpu_memory = torch.xpu.get_device_properties(self.local_rank).total_memory

        ENV_CCL_ATL_TRANSPORT = os.getenv("CCL_ATL_TRANSPORT", "ofi")
        ENV_LOCAL_WORLD_SIZE = os.getenv("LOCAL_WORLD_SIZE", str(self.parallel_config.world_size))
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
        torch.distributed.all_reduce(torch.zeros(1).xpu(), group=get_world_group().device_group)

        # Set random seed.
        set_random_seed(self.model_config.seed)
        # Now take memory snapshot after NCCL is initialized
        gc.collect()
        torch.xpu.empty_cache()
        torch.xpu.reset_peak_memory_stats()

        free_gpu_memory, total_gpu_memory = torch.xpu.mem_get_info()
        self.requested_memory = total_gpu_memory * self.cache_config.gpu_memory_utilization

        if free_gpu_memory < self.requested_memory:

            def GiB(b):
                return round(b / GiB_bytes, 2)

            raise ValueError(
                f"Free memory on device "
                f"({GiB(free_gpu_memory)}/"
                f"{GiB(total_gpu_memory)} GiB) on startup "
                f"is less than desired GPU memory utilization "
                f"({self.cache_config.gpu_memory_utilization}, "
                f"{GiB(self.requested_memory)} GiB). Decrease GPU memory "
                f"utilization or reduce GPU memory used by other processes."
            )
        self.model_runner: XPUARModelRunner = XPUARModelRunner(self.vllm_config, self.device)

        if self.rank == 0:
            from vllm.v1.utils import report_usage_stats

            report_usage_stats(self.vllm_config)
