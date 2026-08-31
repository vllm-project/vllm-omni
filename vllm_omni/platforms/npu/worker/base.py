# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Base NPU worker class for vLLM-Omni with OmniProfiler support."""

import logging
import time

from vllm_omni.platforms.npu._310p import is_310p

if is_310p():
    from vllm_ascend._310p.worker_310p import NPUWorker310 as NPUWorker
else:
    from vllm_ascend.worker.worker import NPUWorker

_NPU_WORKER_LOGGER = logging.getLogger(NPUWorker.__module__)


class OmniNPUWorkerBase(NPUWorker):
    """Base NPU worker for vLLM-Omni with OmniProfiler support.

    This class replaces vllm-ascend's profiler with OmniProfiler for
    unified profiling across all platforms.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Replace vllm-ascend's profiler with OmniProfiler
        profiler_config = self.vllm_config.profiler_config
        if profiler_config and profiler_config.profiler == "torch":
            from vllm_omni.profiler import create_omni_profiler

            stage_id = getattr(self.vllm_config.model_config, "stage_id", 0)
            worker_name = f"stage{stage_id}_rank{self.rank}"
            self.profiler = create_omni_profiler(
                profiler_config=profiler_config,
                worker_name=worker_name,
                local_rank=self.local_rank,
            )

    def profile_memory(self) -> None:
        """Walk the NPU caching allocator only when someone will read the result.

        ``NPUWorker.profile_memory`` samples ``torch.npu.memory_reserved`` and
        ``memory_allocated`` on every ``execute_model`` call, and the values it
        collects are consumed by nothing but its own DEBUG log line. Skip the
        walk while the vLLM-Ascend worker logger is above DEBUG, and keep the
        original diagnostics exactly as they were once it is turned on.
        """
        if not _NPU_WORKER_LOGGER.isEnabledFor(logging.DEBUG):
            return
        super().profile_memory()

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        """Override to set trace filename before starting the profiler.

        NPUWorker's profile() accepts profile_prefix, so we use it to generate
        a descriptive trace filename for OmniProfiler.
        """
        if self.profiler is None:
            raise RuntimeError(
                "Profiling is not enabled. For diffusion models, set --profiler-config via CLI. "
                "For omni models, add profiler_config to your stage config file."
            )
        if is_start:
            from vllm_omni.profiler import OmniTorchProfilerWrapper

            if isinstance(self.profiler, OmniTorchProfilerWrapper):
                # Include stage_id and rank in default filename to distinguish
                # traces from different stages profiling on the same local_rank
                stage_id = getattr(self.vllm_config.model_config, "stage_id", 0)
                filename = profile_prefix or f"stage{stage_id}_rank{self.rank}_{int(time.time())}"
                self.profiler.set_trace_filename(filename)
            self.profiler.start()
        else:
            self.profiler.stop()
