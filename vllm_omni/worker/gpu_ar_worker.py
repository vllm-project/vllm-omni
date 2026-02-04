import gc
import os

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.mem_utils import MemorySnapshot, format_gib
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_worker import Worker as GPUWorker
from vllm.v1.worker.gpu_worker import init_worker_distributed_environment
from vllm.v1.worker.utils import request_memory
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.mixins import OmniWorkerMixin

logger = init_logger(__name__)


class GPUARWorker(OmniWorkerMixin, GPUWorker):
    """GPU worker for autoregressive omni model stages.

    Extends the base GPUWorker to initialize and manage autoregressive
    model runners for text generation stages (e.g., thinker stages).
    """

    def init_device(self):
        if self.device_config.device_type == "cuda":
            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            parallel_config = self.parallel_config
            if (
                parallel_config.distributed_executor_backend not in ("ray", "external_launcher")
                and parallel_config.data_parallel_backend != "ray"
                and parallel_config.nnodes_within_dp == 1
            ):
                # Use local DP rank if available, otherwise use global DP rank.
                dp_local_rank = self.parallel_config.data_parallel_rank_local
                if dp_local_rank is None:
                    dp_local_rank = self.parallel_config.data_parallel_index

                tp_pp_world_size = (
                    self.parallel_config.pipeline_parallel_size * self.parallel_config.tensor_parallel_size
                )

                # DP_LOCAL_RANK * TP_PP_WORLD_SIZE + TP_LOCAL_RANK
                self.local_rank += dp_local_rank * tp_pp_world_size
                assert self.local_rank < torch.cuda.device_count(), (
                    f"DP adjusted local rank {self.local_rank} is out of bounds. "
                )
                visible_device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
                assert self.parallel_config.local_world_size <= visible_device_count, (
                    f"local_world_size ({self.parallel_config.local_world_size}) must "
                    f"be less than or equal to the number of visible devices "
                    f"({visible_device_count})."
                )
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
            self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)
            self.requested_memory = request_memory(init_snapshot, self.cache_config)
            logger.debug("worker init memory snapshot: %r", self.init_snapshot)
            logger.debug("worker requested memory: %sGiB", format_gib(self.requested_memory))
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        # Initialize workspace manager
        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1
        init_workspace_manager(self.device, num_ubatches)

        if self.use_v2_model_runner:
            # OMNI: v2 model runner does not yet include omni hooks.
            logger.warning("OMNI GPUARWorker forces v1 model runner for omni hooks.")
            self.use_v2_model_runner = False

        # Construct the model runner
        self.model_runner = GPUARModelRunner(self.vllm_config, self.device)

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)
