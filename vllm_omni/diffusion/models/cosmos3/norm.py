# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch

from vllm_omni.diffusion.layers.norm import RMSNorm as _VllmRMSNorm


class RMSNorm(_VllmRMSNorm):
    """Cosmos3 RMSNorm that uses the FP32 native implementation."""

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward_npu(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)
