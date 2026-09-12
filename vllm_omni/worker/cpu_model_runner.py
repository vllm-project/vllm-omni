# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config import CompilationMode
from vllm.v1.worker.cpu_model_runner import CPUModelRunner

from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner


class OmniCPUModelRunner(CPUModelRunner, OmniGPUModelRunner):
    def load_model(self, *args, **kwargs) -> None:
        CPUModelRunner.load_model(self, *args, **kwargs)
        self._omni_post_load_model()

    def warming_up_model(self) -> None:
        if self.compilation_config.mode == CompilationMode.NONE:
            return
        super().warming_up_model()

    def _should_use_async_omni_output(self) -> bool:
        # Omni async output overlaps a GPU->CPU copy with the next step's
        # GPU kernels via a dedicated CUDA stream/event. There is no such
        # overlap opportunity on CPU, and building it would unconditionally
        # touch torch.cuda APIs that don't work without a CUDA device.
        return False
