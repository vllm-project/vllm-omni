# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.worker.cpu_worker import CPUWorker

from vllm_omni.worker.cpu_ar_model_runner import CPUARModelRunner
from vllm_omni.worker.mixins import OmniWorkerMixin


class CPUARWorker(OmniWorkerMixin, CPUWorker):
    def init_device(self):
        super().init_device()
        self.model_runner: CPUARModelRunner = CPUARModelRunner(self.vllm_config, self.device)
