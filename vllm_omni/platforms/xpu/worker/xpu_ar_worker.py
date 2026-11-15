# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from vllm.logger import init_logger
from vllm.v1.worker.xpu_worker import XPUWorker

from vllm_omni.worker.base import OmniGPUWorkerBase
from vllm_omni.worker.mixins import OmniWorkerMixin

logger = init_logger(__name__)


class XPUARWorker(OmniWorkerMixin, OmniGPUWorkerBase, XPUWorker):
    """XPU AR worker for thinker/talker stages in Omni model."""

    def _make_model_runner(self):
        from vllm_omni.platforms.xpu.worker.xpu_omni_v2_model_runner import XPUOmniARModelRunner

        logger.info("Using MR v2 XPUOmniARModelRunner.")
        return XPUOmniARModelRunner(self.vllm_config, self.device)
