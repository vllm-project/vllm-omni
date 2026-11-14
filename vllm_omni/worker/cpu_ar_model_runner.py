# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.worker.cpu_model_runner import OmniCPUModelRunner
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner


# GPUARModelRunner provides omni AR hooks (sample_tokens, hidden-state
# extraction, connector init); OmniCPUModelRunner supplies the CPU-safe
# overrides from vLLM's CPUModelRunner via MRO.
class CPUARModelRunner(OmniCPUModelRunner, GPUARModelRunner):
    pass
