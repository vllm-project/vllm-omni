# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.worker.workspace import init_workspace_manager

from vllm_omni.platforms.npu.ascend_warmup_patch import apply_ascend_warmup_patch
from vllm_omni.platforms.npu.worker.base import OmniNPUWorkerBase
from vllm_omni.platforms.npu.worker.npu_ar_model_runner import NPUARModelRunner
from vllm_omni.worker.mixins import OmniWorkerMixin

# Install at import time: this module is imported while the stage worker is
# built, which is earlier than the ascend Triton kernel warmup (that runs
# after load_model in the same process). The guard resolves the SoC per call,
# so installing this early is safe -- the probe only has to be right when a
# warmup actually runs.
apply_ascend_warmup_patch()


class NPUARWorker(OmniWorkerMixin, OmniNPUWorkerBase):
    """NPU AR worker for thinker/talker stages in Omni model."""

    model_runner_cls = NPUARModelRunner

    def init_device(self):
        self.device = self._init_device()
        num_ubatches = 1
        init_workspace_manager(self.device, num_ubatches)

        self.model_runner = self.model_runner_cls(self.vllm_config, self.device)
