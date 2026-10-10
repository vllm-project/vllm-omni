# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.worker.workspace import init_workspace_manager

from vllm_omni.platforms import current_omni_platform
from vllm_omni.platforms.npu.worker.base import OmniNPUWorkerBase
from vllm_omni.platforms.npu.worker.npu_ar_model_runner import NPUARModelRunner
from vllm_omni.worker.mixins import OmniWorkerMixin


class NPUARWorker(OmniWorkerMixin, OmniNPUWorkerBase):
    """NPU AR worker for thinker/talker stages in Omni model."""

    model_runner_cls = NPUARModelRunner

    def init_device(self):
        self.device = self._init_device()
        num_ubatches = 1
        init_workspace_manager(self.device, num_ubatches)

        # Install platform-specific AR runtime patches (e.g. the MOSS-TTS
        # depth whole-loop NPUGraph adapter) before model loading. MOSS stage 0
        # runs on this worker, whose init_device inherits vllm-ascend's
        # _init_device and never reaches NPUOmniPlatform.set_device, so this
        # is the guaranteed AR startup path for such registration.
        current_omni_platform.init_ar_worker_runtime(self.vllm_config, self.device)

        self.model_runner = self.model_runner_cls(self.vllm_config, self.device)
