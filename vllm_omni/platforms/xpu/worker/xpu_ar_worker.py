# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.v1.worker.xpu_worker import XPUWorker

from vllm_omni.platforms.xpu.worker.xpu_ar_model_runner import XPUARModelRunner
from vllm_omni.worker.mixins import OmniWorkerMixin

logger = init_logger(__name__)


class XPUARWorker(OmniWorkerMixin, XPUWorker):
    """XPU AR worker for thinker/talker stages in Omni model."""

    def init_device(self):
        super().init_device()
        if self.use_v2_model_runner:
            # OMNI: v2 model runner does not yet include omni hooks.
            logger.warning("OMNI XPUARWorker forces v1 model runner for omni hooks.")
            self.use_v2_model_runner = False
        self.model_runner: XPUARModelRunner = XPUARModelRunner(self.vllm_config, self.device)
