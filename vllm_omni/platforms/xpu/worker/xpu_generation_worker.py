# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import time

from vllm.logger import init_logger
from vllm.tracing import instrument
from vllm.v1.worker.gpu_worker import CompilationTimes
from vllm.v1.worker.xpu_worker import XPUWorker

from vllm_omni.platforms.xpu.worker.xpu_generation_model_runner import XPUGenerationModelRunner
from vllm_omni.worker.base import OmniGPUWorkerBase
from vllm_omni.worker.mixins import OmniWorkerMixin

logger = init_logger(__name__)


class XPUGenerationWorker(OmniWorkerMixin, OmniGPUWorkerBase, XPUWorker):
    """XPU generation worker for the code2wav (non-AR waveform generation) stage in the Omni model."""

    model_runner_cls = XPUGenerationModelRunner

    def _make_model_runner(self):
        # Built once here, from XPUWorker's init_device: model_config's
        # use_v2_model_runner (deploy ``model_runner``) picks the omni runner.
        if self.use_v2_model_runner:
            from vllm_omni.platforms.xpu.worker.xpu_omni_v2_model_runner import XPUOmniGenerationModelRunner

            logger.info("Using MR v2 XPUOmniGenerationModelRunner for omni generation stage.")
            return XPUOmniGenerationModelRunner(self.vllm_config, self.device)
        return self.model_runner_cls(self.vllm_config, self.device)

    @instrument(span_name="Compile/warmup")
    def compile_or_warm_up_model(self) -> CompilationTimes:
        """Same as GPUGenerationWorker: v2 generation stages have no KV cache or sampler."""
        if not self.use_v2_model_runner:
            return super().compile_or_warm_up_model()
        start = time.perf_counter()
        self.model_runner.profile_run()
        self._capture_auxiliary_graphs()
        return CompilationTimes(language_model=time.perf_counter() - start, encoder=0.0)
