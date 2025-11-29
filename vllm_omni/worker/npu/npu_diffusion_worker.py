# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_ascend.worker.worker_v1 import NPUWorker

from vllm_omni.worker.npu.npu_diffusion_model_runner import NPUDiffusionModelRunner


class NPUDiffusionWorker(NPUWorker):
    """NPU diffusion worker for code2wav stage in Omni model."""

    def init_device(self):
        device = self._init_device()

        self.model_runner: NPUDiffusionModelRunner = NPUDiffusionModelRunner(self.vllm_config, device)
