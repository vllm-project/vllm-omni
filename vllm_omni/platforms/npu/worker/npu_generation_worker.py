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

logger = init_logger(__name__)

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

        # model_config.use_v2_model_runner carries the stage deploy field
        # (patch.py writes it for OmniModelConfig). Generation stages on
        # Ascend V2 run the generic OmniGenerationModelRunner contract: no
        # KV cache and no sampler, the model owns its NPU graph capture.
        if self.use_v2_model_runner:
            from vllm_omni.platforms.npu.worker.npu_omni_v2_model_runner import (
                NPUOmniGenerationModelRunner,
            )

            logger.info("Using MR v2 NPUOmniGenerationModelRunner on NPU.")
            self.model_runner = NPUOmniGenerationModelRunner(self.vllm_config, self.device)
        else:
            self.model_runner = self.model_runner_cls(self.vllm_config, self.device)

    def compile_or_warm_up_model(self):
        """Generation stages have no KV cache or sampler -- skip warmup_kernels."""
        if not self.use_v2_model_runner:
            return super().compile_or_warm_up_model()
        start = time.perf_counter()
        self.model_runner.profile_run()
        self._capture_auxiliary_graphs()
        return CompilationTimes(language_model=time.perf_counter() - start, encoder=0.0)
