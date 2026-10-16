# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.worker.workspace import init_workspace_manager

from vllm_omni.platforms.npu.ascend_warmup_patch import apply_ascend_warmup_patch
from vllm_omni.platforms.npu.worker.base import OmniNPUWorkerBase
from vllm_omni.platforms.npu.worker.npu_generation_model_runner import NPUGenerationModelRunner
from vllm_omni.worker.mixins import OmniWorkerMixin

# See npu_ar_worker: install before the worker is used, so the guard is in
# place when the ascend Triton kernel warmup runs later in this process.
apply_ascend_warmup_patch()


class NPUGenerationWorker(OmniWorkerMixin, OmniNPUWorkerBase):
    """NPU generation worker for code2wav stage in Omni model."""

    model_runner_cls = NPUGenerationModelRunner

    def init_device(self):
        self.device = self._init_device()
        num_ubatches = 1
        init_workspace_manager(self.device, num_ubatches)

        self.model_runner = self.model_runner_cls(self.vllm_config, self.device)
