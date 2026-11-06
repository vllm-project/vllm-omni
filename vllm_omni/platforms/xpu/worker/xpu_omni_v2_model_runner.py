# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Omni Model Runner V2 runners on XPU.

The omni V2 runners built under ``torch_cuda_wrapper``, the same way vLLM's
``XPUModelRunnerV2`` wraps ``GPUModelRunnerV2``. The XPU workers import this
module only when a stage selects ``model_runner: v2``.
"""

from vllm_omni.platforms.xpu.utils import torch_cuda_wrapper
from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner
from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner


class XPUOmniARModelRunner(OmniARModelRunner):
    def __init__(self, *args, **kwargs):
        with torch_cuda_wrapper():
            super().__init__(*args, **kwargs)


class XPUOmniGenerationModelRunner(OmniGenerationModelRunner):
    def __init__(self, *args, **kwargs):
        with torch_cuda_wrapper():
            super().__init__(*args, **kwargs)
