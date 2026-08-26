# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wan-specific normalization layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni.diffusion.layers.custom_op import CustomOp


class RMSNormVAE(CustomOp):
    """Root Mean Square Layer Normalization for Channel-First or Last"""

    def __init__(
        self,
        dim: int,
        channel_first: bool = True,
        images: bool = True,
        bias: bool = False,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else None
        self.epsilon = epsilon

        self.gamma_rmsnorm = None

    def forward_cuda(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x)

    def forward_hip(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x)

    def forward_npu(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        import torch_npu

        if self.gamma_rmsnorm is None:
            self.gamma_rmsnorm = self.gamma.reshape(-1)

        if self.channel_first:
            x = x.transpose(1, -1)
            out = torch_npu.npu_rms_norm(x, self.gamma_rmsnorm, epsilon=self.epsilon)[0].transpose(1, -1)
        else:
            out = torch_npu.npu_rms_norm(x, self.gamma_rmsnorm, epsilon=self.epsilon)[0]

        if self.bias is not None:
            out = out + self.bias
        return out

    def forward_xpu(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_native(x)

    def forward_native(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        out = (
            F.normalize(
                x,
                dim=(1 if self.channel_first else -1),
                eps=self.epsilon,
            )
            * self.scale
            * self.gamma
        )
        if self.bias is not None:
            out = out + self.bias
        return out
