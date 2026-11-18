# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import time

from vllm.logger import init_logger
from vllm.v1.worker.gpu_worker import CompilationTimes
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_omni.platforms.npu.ascend_warmup_patch import apply_ascend_warmup_patch
from vllm_omni.platforms.npu.worker.base import OmniNPUWorkerBase
from vllm_omni.platforms.npu.worker.npu_generation_model_runner import NPUGenerationModelRunner
from vllm_omni.worker.mixins import OmniWorkerMixin

# See npu_ar_worker: install before the worker is used, so the guard is in
# place when the ascend Triton kernel warmup runs later in this process.
apply_ascend_warmup_patch()
logger = init_logger(__name__)


class NPUGenerationWorker(OmniWorkerMixin, OmniNPUWorkerBase):
    """NPU generation worker for code2wav stage in Omni model."""

    model_runner_cls = NPUGenerationModelRunner

    def init_device(self):
        self.device = self._init_device()
        num_ubatches = 1
        init_workspace_manager(self.device, num_ubatches)

        if self.use_v2_model_runner:
            from vllm_omni.platforms.npu.worker.npu_generation_model_runner_v2 import NPUGenerationModelRunnerV2

            logger.info("Using MR v2 NPUGenerationModelRunnerV2 for generation stage.")
            self.model_runner = NPUGenerationModelRunnerV2(self.vllm_config, self.device)
        else:
            self.model_runner = self.model_runner_cls(self.vllm_config, self.device)

    def compile_or_warm_up_model(self) -> CompilationTimes:
        """Skip language-model warmup for codec IDs and retain inner graphs."""
        if not self.use_v2_model_runner:
            return super().compile_or_warm_up_model()
        start = time.perf_counter()
        self.model_runner.profile_run()
        self._capture_auxiliary_graphs()
        return CompilationTimes(language_model=time.perf_counter() - start, encoder=0.0)
