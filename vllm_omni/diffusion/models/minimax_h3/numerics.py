# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Optional eager arithmetic matching the released H3 Diffusers transformer.

BF16 intermediates are part of the checkpoint's inference behavior. Fusing a
normalization, modulation, or residual addition can change its rounding points.
These operations deliberately preserve the reference's intermediate casts.
"""

import torch
import torch.nn.functional as F

from vllm_omni.diffusion.layers.norm import RMSNorm


class MiniMaxH3RMSNorm(RMSNorm):
    reference_numerics: bool = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.reference_numerics:
            return F.rms_norm(x, (self.hidden_size,), self.weight, self.variance_epsilon)
        return super().forward(x)


def reference_scale_shift(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    return x * (1 + scale.index_select(0, indices)) + shift.index_select(0, indices)


def reference_gate(
    residual: torch.Tensor, gate: torch.Tensor, hidden: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    return residual + gate.index_select(0, indices) * hidden
