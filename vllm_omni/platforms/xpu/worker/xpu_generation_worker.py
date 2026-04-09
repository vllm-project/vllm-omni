# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.worker.xpu_worker import XPUWorker

from vllm_omni.platforms.xpu.worker.xpu_generation_model_runner import XPUGenerationModelRunner
from vllm_omni.worker.mixins import OmniWorkerMixin


class XPUGenerationWorker(OmniWorkerMixin, XPUWorker):
    """XPU generation worker for the code2wav (non-AR waveform generation) stage in the Omni model."""

    def init_device(self):
        super().init_device()
        self.model_runner: XPUGenerationModelRunner = XPUGenerationModelRunner(self.vllm_config, self.device)
