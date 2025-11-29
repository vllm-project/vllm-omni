# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_ascend.worker.worker_v1 import NPUWorker

from vllm_omni.worker.npu.npu_ar_model_runner import NPUARModelRunner


class NPUARWorker(NPUWorker):
    """NPU AR worker for thinker/talker stages in Omni model."""

    def init_device(self):
        device = self._init_device()

        self.model_runner: NPUARModelRunner = NPUARModelRunner(self.vllm_config, device)
