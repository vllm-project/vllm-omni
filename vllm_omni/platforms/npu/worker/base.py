# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Base NPU worker class for vLLM-Omni with OmniProfiler support."""

import time

from vllm_omni.platforms.npu._310p import is_310p

if is_310p():
    from vllm_ascend._310p.worker_310p import NPUWorker310 as NPUWorker
else:
    from vllm_ascend.worker.worker import NPUWorker


class OmniNPUWorkerBase(NPUWorker):
    """Base NPU worker for vLLM-Omni with OmniProfiler support.

    This class replaces vllm-ascend's profiler with OmniProfiler for
    unified profiling across all platforms.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Ascend inherits vLLM's global runner default (V2 in vLLM 0.32),
        # while Omni resolves the runner independently for each deploy stage.
        # Keep runner construction and inherited worker lifecycle methods on
        # the same stage-selected path, including mixed V2 AR / V1 DAC deploys.
        self.use_v2_model_runner = bool(getattr(self.vllm_config.model_config, "use_v2_model_runner", False))

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

    def get_stage_post_warmup_memory_stats(self) -> None:
        """NPU workers do not provide the optional CUDA warmup snapshot."""
        return None

    def profile(
        self,
        is_start: bool = True,
        profile_prefix: str | None = None,
        *,
        delay_iterations: int | None = None,
        max_iterations: int | None = None,
    ):
        """Override to set trace filename before starting the profiler.

        NPUWorker's profile() accepts profile_prefix, so we use it to generate
        a descriptive trace filename for OmniProfiler.
        Per-session iteration controls are handled by the profiler wrapper.
        """
        if self.profiler is None:
            raise RuntimeError(
                "Profiling is not enabled. For diffusion models, set --profiler-config via CLI. "
                "For omni models, add profiler_config to the relevant deploy config stage."
            )
        if is_start:
            from vllm_omni.profiler import OmniTorchProfilerWrapper

            if isinstance(self.profiler, OmniTorchProfilerWrapper):
                # Include stage_id and rank in default filename to distinguish
                # traces from different stages profiling on the same local_rank
                stage_id = getattr(self.vllm_config.model_config, "stage_id", 0)
                filename = profile_prefix or f"stage{stage_id}_rank{self.rank}_{int(time.time())}"
                self.profiler.set_trace_filename(filename)
            kwargs: dict[str, int] = {}
            if delay_iterations is not None:
                kwargs["delay_iterations"] = delay_iterations
            if max_iterations is not None:
                kwargs["max_iterations"] = max_iterations
            self.profiler.start(**kwargs)
        else:
            self.profiler.stop()

    def shutdown(self) -> None:
        try:
            runner = getattr(self, "model_runner", None)
            if runner is not None:
                runner._close_omni_aux_output()
        finally:
            super().shutdown()
