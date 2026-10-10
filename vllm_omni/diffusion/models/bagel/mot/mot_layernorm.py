# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# ruff: noqa: N803, E741
"""Mixture-of-Tokens (MoT) RMS Normalization layer.

Holds two sets of weights (text / gen) and routes tokens to the
appropriate weight based on indices.  When text_indices is None the
layer degrades to a standard RMSNorm using self.weight (und mode).
"""

import torch
import torch.nn as nn
from vllm import ir

from vllm_omni.diffusion.layers.custom_op import CustomOp


class MoTRMSNorm(CustomOp):
    """Mixture-of-Tokens RMS Normalization.

    In *und* mode (``text_indices is None``), every token is normalised
    with ``self.weight`` – exactly like a vanilla RMSNorm.

    In *gen* mode, text tokens are normalised with ``self.weight`` and
    gen tokens are normalised with ``self.gen_weight``. CUDA uses a fused
    Triton kernel; NPU uses the fused CANN RMSNorm operator for each subset.
    """

    def __init__(
        self,
        hidden_size: int,
        head_norm: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.gen_weight = nn.Parameter(torch.ones(hidden_size))
        self.head_norm = head_norm

    # ------------------------------------------------------------------
    # Native (pure-PyTorch) fallback
    # ------------------------------------------------------------------
    def forward_native(
        self,
        x: torch.Tensor,
        text_indices: torch.Tensor | None = None,
        vae_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if text_indices is None:
            return self._rms_norm_native(x, self.weight)

        output = torch.empty_like(x)
        output[text_indices] = self._rms_norm_native(x[text_indices], self.weight)
        output[vae_indices] = self._rms_norm_native(x[vae_indices], self.gen_weight)
        return output

    # ------------------------------------------------------------------
    # CUDA fast-path
    # ------------------------------------------------------------------
    def forward_cuda(
        self,
        x: torch.Tensor,
        text_indices: torch.Tensor | None = None,
        vae_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if text_indices is None:
            # und mode – delegate to vllm's highly-optimised CUDA kernel
            return ir.ops.rms_norm(x, self.weight.data, self.variance_epsilon)

        # gen mode – fused MoT Triton kernel
        from vllm_omni.diffusion.models.bagel.mot.ops.mot_rms_norm import (
            mot_rms_norm,
        )

        return mot_rms_norm(
            x,
            self.weight.data,
            self.gen_weight.data,
            text_indices,
            vae_indices,
            head_norm=self.head_norm,
            eps=self.variance_epsilon,
        )

    # ------------------------------------------------------------------
    # NPU fast-path (fused CANN RMSNorm operator)
    # ------------------------------------------------------------------
    def forward_npu(
        self,
        x: torch.Tensor,
        text_indices: torch.Tensor | None = None,
        vae_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if text_indices is None:
            return self._rms_norm_npu(x, self.weight)

        output = torch.empty_like(x)
        if text_indices.numel() > 0:
            output[text_indices] = self._rms_norm_npu(x[text_indices], self.weight)
        if vae_indices.numel() > 0:
            output[vae_indices] = self._rms_norm_npu(x[vae_indices], self.gen_weight)
        return output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _rms_norm_native(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return (x * weight.float()).to(orig_dtype)

    def _rms_norm_npu(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # Use the fused CANN operator to replace the native Pow/ReduceMean/rsqrt
        # sequence. The NPU operator requires gamma to match the input dtype.
        import torch_npu

        if weight.dtype != x.dtype or weight.device != x.device:
            weight = weight.to(device=x.device, dtype=x.dtype)
        normalized, _ = torch_npu.npu_rms_norm(x, weight, epsilon=self.variance_epsilon)
        return normalized

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.variance_epsilon}"
