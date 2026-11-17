# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch
import torch.nn.functional as F

from vllm_omni.diffusion.layers.norm import RMSNorm as _VllmRMSNorm


class RMSNorm(_VllmRMSNorm):
    """Cosmos3 Transformer RMSNorm with NPU fusion and native CUDA/HIP paths."""

    def forward_cuda(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_native(x, residual)

    def forward_hip(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_native(x, residual)


class VAERMSNorm(RMSNorm):
    """Preserve FP32 native normalization for the Cosmos3 LiDAR VAE."""

    def forward_npu(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_native(x, residual)


class QKRMSNorm(RMSNorm):
    """Preserve functional Q/K normalization outside the fused NPU path."""

    def forward_native(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        if residual is not None:
            raise ValueError("QKRMSNorm does not support residual input.")
        return F.rms_norm(x, (self.hidden_size,), self.weight, eps=self.variance_epsilon)
