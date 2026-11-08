# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch
import torch.nn.functional as F

from vllm_omni.diffusion.layers.norm import RMSNorm as _VllmRMSNorm


class RMSNorm(_VllmRMSNorm):
    """Cosmos3 RMSNorm with shared NPU fusion and native CUDA/HIP paths."""

    def forward_cuda(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_native(x, residual)

    def forward_hip(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_native(x, residual)

    def forward_npu(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return super().forward_npu(x, residual)


class QKRMSNorm(RMSNorm):
    """Preserve functional Q/K normalization outside the fused NPU path."""

    def forward_native(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            residual = residual + x
            x = residual
        output = F.rms_norm(x, (self.hidden_size,), self.weight, eps=self.variance_epsilon)
        return (output, residual) if residual is not None else output
