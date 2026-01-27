# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.worker.workspace import init_workspace_manager
from vllm_ascend.worker.worker import NPUWorker

from vllm_omni.worker.npu.npu_generation_model_runner import NPUGenerationModelRunner


class NPUGenerationWorker(NPUWorker):
    """NPU generation worker for code2wav stage in Omni model."""

    def init_device(self):
        self.device = self._init_device()
        num_ubatches = 1
        init_workspace_manager(self.device, num_ubatches)

        self.model_runner = NPUGenerationModelRunner(self.vllm_config, self.device)
