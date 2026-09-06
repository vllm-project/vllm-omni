# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.worker.cpu_model_runner import OmniCPUModelRunner
from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner


# GPUGenerationModelRunner provides omni generation hooks;
# OmniCPUModelRunner supplies the CPU-safe overrides from vLLM's
# CPUModelRunner via MRO.
class CPUGenerationModelRunner(OmniCPUModelRunner, GPUGenerationModelRunner):
    pass
