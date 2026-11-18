# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from vllm.logger import init_logger
from vllm.v1.worker.xpu_worker import XPUWorker

from vllm_omni.platforms.xpu.worker.xpu_ar_model_runner import XPUARModelRunner
from vllm_omni.worker.base import OmniGPUWorkerBase
from vllm_omni.worker.mixins import OmniWorkerMixin

logger = init_logger(__name__)


class XPUARWorker(OmniWorkerMixin, OmniGPUWorkerBase, XPUWorker):
    """XPU AR worker for thinker/talker stages in Omni model."""

    model_runner_cls = XPUARModelRunner

    def _make_model_runner(self):
        # Built once here, from XPUWorker's init_device: model_config's
        # use_v2_model_runner (deploy ``model_runner``) picks the omni runner.
        if self.use_v2_model_runner:
            from vllm_omni.platforms.xpu.worker.xpu_omni_v2_model_runner import XPUOmniARModelRunner

            logger.info("Using MR v2 XPUOmniARModelRunner for omni AR stage.")
            return XPUOmniARModelRunner(self.vllm_config, self.device)
        return self.model_runner_cls(self.vllm_config, self.device)
