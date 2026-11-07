# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Base NPU worker class for vLLM-Omni with OmniProfiler support."""

import time

from vllm import envs
from vllm.config import CUDAGraphMode
from vllm.logger import init_logger

from vllm_omni.platforms.npu._310p import is_310p

if is_310p():
    from vllm_ascend._310p.worker_310p import NPUWorker310 as NPUWorker
else:
    from vllm_ascend.worker.worker import NPUWorker


logger = init_logger(__name__)


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

    def determine_available_memory(self) -> int:
        available = super().determine_available_memory()
        self.encoder_cudagraph_memory_estimate = 0
        if (
            envs.VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS
            and self.vllm_config.compilation_config.cudagraph_mm_encoder
            and self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            and not self.cache_config.kv_cache_memory_bytes
        ):
            model = self.model_runner.get_model()
            if not (
                getattr(model, "encoder_cudagraph_single_replay", False)
                or getattr(model, "_audio_encoder_graphs", None) is not None
            ):
                return available
            self.encoder_cudagraph_memory_estimate = self.model_runner.profile_encoder_cudagraph_memory()
            logger.info("Reserved %.2f GiB for encoder NPU graphs", self.encoder_cudagraph_memory_estimate / (1 << 30))
            available = max(0, available - self.encoder_cudagraph_memory_estimate)
            self.available_kv_cache_memory_bytes = available
        return available

    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        self._capture_auxiliary_graphs()
        return result

    def _capture_auxiliary_graphs(self) -> None:
        """Let opt-in models warm valid inputs before the worker becomes ready.

        Mirrors ``OmniGPUWorkerBase._capture_auxiliary_graphs``: the CUDA base
        has called this from ``compile_or_warm_up_model`` for a while, the NPU
        base did not, so a model that relies on the hook (e.g. capturing its
        audio graphs at startup rather than on the first request) silently
        skipped it on Ascend.
        """
        capture = getattr(self.model_runner.model, "capture_auxiliary_graphs", None)
        if callable(capture):
            capture()

    def profile(
        self,
        is_start: bool = True,
        profile_prefix: str | None = None,
        *,
        delay_iterations: int | None = None,
        max_iterations: int | None = None,
        profiler_kwargs: dict | None = None,
    ):
        """Override to set trace filename before starting the profiler.

        NPUWorker's profile() accepts profile_prefix, so we use it to generate
        a descriptive trace filename for OmniProfiler.
        Per-session iteration controls and constructor overrides are handled
        by the profiler wrapper.
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
            self.profiler.set_profiler_kwargs(profiler_kwargs)
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
