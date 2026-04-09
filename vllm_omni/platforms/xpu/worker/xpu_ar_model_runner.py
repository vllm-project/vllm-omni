# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm_omni.platforms.xpu.utils import torch_cuda_wrapper
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner


class XPUARModelRunner(GPUARModelRunner):
    def __init__(self, *args, **kwargs):
        with torch_cuda_wrapper():
            super().__init__(*args, **kwargs)

    def _init_device_properties(self):
        self.num_sms = None

    def _sync_device(self) -> None:
        torch.xpu.synchronize()
