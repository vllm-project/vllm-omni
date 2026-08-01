# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.v1.worker.xpu_worker import XPUWorker

from vllm_omni.platforms.xpu.worker.xpu_generation_model_runner import XPUGenerationModelRunner
from vllm_omni.worker.mixins import OmniWorkerMixin

logger = init_logger(__name__)


class XPUGenerationWorker(OmniWorkerMixin, XPUWorker):
    """XPU generation worker for the code2wav (non-AR waveform generation) stage in the Omni model."""

    def init_device(self):
        super().init_device()
        if self.use_v2_model_runner:
            # OMNI: v2 model runner does not yet include omni hooks.
            logger.warning("OMNI XPUGenerationWorker forces v1 model runner for omni hooks.")
            self.use_v2_model_runner = False
        self.model_runner: XPUGenerationModelRunner = XPUGenerationModelRunner(self.vllm_config, self.device)
