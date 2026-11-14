# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_omni.platforms import current_omni_platform
from vllm_omni.platforms.npu.ascend_warmup_patch import apply_ascend_warmup_patch
from vllm_omni.platforms.npu.worker.base import OmniNPUWorkerBase
from vllm_omni.platforms.npu.worker.npu_ar_model_runner import NPUARModelRunner
from vllm_omni.worker.mixins import OmniWorkerMixin

logger = init_logger(__name__)


class NPUARWorker(OmniWorkerMixin, OmniNPUWorkerBase):
    """NPU worker for autoregressive stages in Omni."""

    model_runner_cls = NPUARModelRunner

    def init_device(self):
        if not hasattr(self, "device") or self.device is None:
            self.device = self._init_device()
        # activate ascend warmup patch before model initialization
        apply_ascend_warmup_patch()
        # Initialize workspace manager after device setup (following vllm-ascend pattern)
        num_ubatches = 1
        init_workspace_manager(self.device, num_ubatches)

        # Install platform-specific AR runtime patches before model loading;
        # MOSS stage 0 runs on this worker, whose init_device never reaches
        # NPUOmniPlatform.set_device.
        current_omni_platform.init_ar_worker_runtime(self.vllm_config, self.device)

        if self.use_v2_model_runner:
            from vllm_omni.platforms.npu.worker.npu_omni_v2_model_runner import NPUOmniARModelRunner

            logger.info("Using MR v2 NPUOmniARModelRunner on NPU.")
            self.model_runner = NPUOmniARModelRunner(self.vllm_config, self.device)
        else:
            self.model_runner = self.model_runner_cls(self.vllm_config, self.device)
