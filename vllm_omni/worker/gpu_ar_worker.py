import gc
import os

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.tracing import instrument
from vllm.utils.mem_utils import MemorySnapshot, format_gib
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_worker import init_worker_distributed_environment
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_omni.diffusion.data import OmniACK, OmniSleepTask, OmniWakeTask
from vllm_omni.worker.base import OmniGPUWorkerBase
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.memory_utils import request_memory_tolerant
from vllm_omni.worker.mixins import OmniWorkerMixin

logger = init_logger(__name__)

VLLM_OMNI_USE_V2_RUNNER = bool(int(os.environ.get("VLLM_OMNI_USE_V2_RUNNER", "0")))


class GPUARWorker(OmniWorkerMixin, OmniGPUWorkerBase):
    """GPU worker for autoregressive omni model stages.

    Extends the base GPUWorker to initialize and manage autoregressive
    model runners for text generation stages (e.g., thinker stages).
    """

    @instrument(span_name="Init device")
    def init_device(self):
        if self.device_config.device_type in ("cuda", "musa"):
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

        if VLLM_OMNI_USE_V2_RUNNER or self.use_v2_model_runner:
            from vllm_omni.worker_v2.omni_ar_model_runner import (
                OmniARModelRunner,
            )

            logger.info("Using MR v2 OmniARModelRunner for omni AR stage.")
            self.use_v2_model_runner = True
            self.model_runner = OmniARModelRunner(self.vllm_config, self.device)
        else:
            self.model_runner = GPUARModelRunner(self.vllm_config, self.device)

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    @instrument(span_name="Compile/warmup")
    def compile_or_warm_up_model(self) -> float:
        """Skip warmup_kernels for V2 Omni AR models.

        Upstream ``compile_or_warm_up_model`` calls ``warmup_kernels()``
        only for V2 model runners.  ``warmup_kernels`` creates fake
        requests with hardcoded block_ids that bypass KVCacheManager
        and runs real (non-dummy) execute_model + sample_tokens.  For
        Omni AR models with preprocess, this pollutes model-internal
        state and produces incorrect logits for subsequent real
        requests (observed 30x more decode steps than V1).

        This override runs the parent implementation but patches out
        the ``warmup_kernels`` call, matching V1 behavior.
        """
        if not self.use_v2_model_runner:
            return super().compile_or_warm_up_model()

        # Temporarily disable warmup_kernels during parent call.
        # Must patch both the module attribute AND the local name in
        # gpu_worker (from-import creates a local binding).
        import vllm.v1.worker.gpu_worker as _gw

        _orig = _gw.warmup_kernels

        def _noop(*a, **kw):
            return None

        _gw.warmup_kernels = _noop
        # Also patch the module where it was originally defined
        import vllm.v1.worker.gpu.warmup as _wm

        _orig_wm = _wm.warmup_kernels
        _wm.warmup_kernels = _noop
        try:
            result = super().compile_or_warm_up_model()
        finally:
            _gw.warmup_kernels = _orig
            _wm.warmup_kernels = _orig_wm
        logger.info("compile_or_warm_up_model: skipped warmup_kernels for Omni AR V2")
        return result

    def handle_sleep_task(self, task: OmniSleepTask | dict) -> OmniACK:
        """
        Explicitly handle sleep commands.
        Calls the implementation in the base class OmniGPUWorkerBase.
        """
        logger.debug(f"[AR Worker {self.rank}] Resolving handle_sleep_task dispatch")
        if isinstance(task, dict):
            task = OmniSleepTask(**task)
        return super().handle_sleep_task(task)

    def handle_wake_task(self, task: OmniWakeTask | dict) -> OmniACK:
        """
        Explicitly handle wake-up commands.
        Calls the implementation in the base class OmniGPUWorkerBase.
        """
        logger.debug(f"[AR Worker {self.rank}] Resolving handle_wake_task dispatch")
        if isinstance(task, dict):
            task = OmniWakeTask(**task)
        return super().handle_wake_task(task)
