# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared NPU rotary embedding helpers."""

from __future__ import annotations

import torch
import torch_npu


def _bnsd_rotary_shape_is_supported(hidden_states: torch.Tensor) -> bool:
    """Return whether the fused half-mode BNSD tiler supports the shape.

    For the aligned-head-dimension branch, CANN requires
    ``batch * num_heads <= seq_len * 8``. Other alignments are handled with
    the BSND layout, which does not have this short-sequence limit.
    """
    if hidden_states.ndim != 4:
        return False

    batch_size, num_heads, seq_len, head_dim = hidden_states.shape
    element_size = hidden_states.element_size()
    if head_dim % 2 != 0 or element_size <= 0 or 32 % element_size != 0:
        return False

    half_dim_alignment = 32 // element_size
    return head_dim // 2 % half_dim_alignment == 0 and batch_size * num_heads <= seq_len * 8


def npu_rotary_mul_with_bsnd_fallback(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int,
) -> torch.Tensor:
    """Apply fused rotary multiplication, falling back from BNSD to BSND.

    ``unsqueeze_dim=1`` denotes BNSD input following the Transformers rotary
    embedding convention. Supported BNSD shapes stay on the fast path. Shapes
    rejected by the CANN BNSD tiler are transposed to BSND for the fused op and
    transposed back before returning.
    """
    if unsqueeze_dim != 1 or _bnsd_rotary_shape_is_supported(hidden_states):
        return torch_npu.npu_rotary_mul(
            hidden_states,
            cos.unsqueeze(unsqueeze_dim),
            sin.unsqueeze(unsqueeze_dim),
        )

    hidden_states_bsnd = hidden_states.transpose(1, 2).contiguous()
    output_bsnd = torch_npu.npu_rotary_mul(
        hidden_states_bsnd,
        cos.unsqueeze(2),
        sin.unsqueeze(2),
    )
    return output_bsnd.transpose(1, 2)
