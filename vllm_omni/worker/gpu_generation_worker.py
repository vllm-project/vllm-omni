import gc
import os

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.tracing import instrument
from vllm.utils.mem_utils import MemorySnapshot, format_gib
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_worker import (
    CompilationTimes,
    init_worker_distributed_environment,
)
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_omni.platforms import current_omni_platform
from vllm_omni.worker.base import OmniGPUWorkerBase
from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner
from vllm_omni.worker.memory_utils import request_memory_tolerant
from vllm_omni.worker.mixins import OmniWorkerMixin

logger = init_logger(__name__)


def _supports_generation_device_type(device_type: str) -> bool:
    return device_type in ("cuda", "musa")


def _make_compilation_times(language_model_time: float) -> CompilationTimes:
    return CompilationTimes(language_model=language_model_time, encoder=0.0)


class GPUGenerationWorker(OmniWorkerMixin, OmniGPUWorkerBase):
    """GPU Worker for Generation model (non-autoregressive waveform generation).

    Selected for stages whose pipeline topology uses
    ``execution_type=StageExecutionType.LLM_GENERATION``.
    """

    model_runner_cls = GPUGenerationModelRunner

    @instrument(span_name="Init device")
    def init_device(self):
        if _supports_generation_device_type(self.device_config.device_type):
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
                assert self.local_rank < torch.accelerator.device_count(), (
                    f"DP adjusted local rank {self.local_rank} is out of bounds. "
                )
                visible_device_count = torch.accelerator.device_count()
                assert self.parallel_config.local_world_size <= visible_device_count, (
                    f"local_world_size ({self.parallel_config.local_world_size}) must "
                    f"be less than or equal to the number of visible devices "
                    f"({visible_device_count})."
                )
            self.device = current_omni_platform.get_torch_device(self.local_rank)
            torch.accelerator.set_device_index(self.device)

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
            torch.accelerator.empty_cache()

            # take current memory snapshot
            self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)
            self.requested_memory = request_memory_tolerant(init_snapshot, self.cache_config)
            logger.debug("worker init memory snapshot: %r", self.init_snapshot)
            logger.debug("worker requested memory: %sGiB", format_gib(self.requested_memory))
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        # Initialize workspace manager
        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1
        init_workspace_manager(self.device, num_ubatches)

        self.use_v2_model_runner = bool(getattr(self.model_config, "use_v2_model_runner", False))
        if self.use_v2_model_runner:
            from vllm_omni.worker_v2.omni_generation_model_runner import (
                OmniGenerationModelRunner,
            )

            logger.info("Using MR v2 OmniGenerationModelRunner for generation stage.")
            self.model_runner = OmniGenerationModelRunner(self.vllm_config, self.device)
        else:
            self.model_runner = GPUGenerationModelRunner(self.vllm_config, self.device)

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    @instrument(span_name="Compile/warmup")
    def compile_or_warm_up_model(self) -> float:
        """Generation stages have no KV cache or sampler — skip warmup_kernels."""
        if not self.use_v2_model_runner:
            return super().compile_or_warm_up_model()
        import time

        start = time.perf_counter()
        self.model_runner.profile_run()
        return _make_compilation_times(time.perf_counter() - start)
