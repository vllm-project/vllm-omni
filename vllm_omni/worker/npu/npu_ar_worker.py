# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.worker.workspace import init_workspace_manager
from vllm_ascend.worker.worker import NPUWorker

from vllm_omni.worker.npu.npu_ar_model_runner import NPUARModelRunner


class NPUARWorker(NPUWorker):
    """NPU AR worker for thinker/talker stages in Omni model."""

    def init_device(self):
        self.device = self._init_device()
        num_ubatches = 1
        init_workspace_manager(self.device, num_ubatches)

        self.model_runner = NPUARModelRunner(self.vllm_config, self.device)
