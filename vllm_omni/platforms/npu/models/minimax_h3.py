# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NPU patches for the MiniMax H3 Qwen3-VL text encoder."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
import torch_npu
from vllm.logger import init_logger

from vllm_omni.platforms.npu.layers.rotary_embedding import (
    npu_rotary_mul_with_bsnd_fallback,
)

logger = init_logger(__name__)

_ROPE_PATCHED = False
_SWIGLU_PATCHED = False


def _apply_rotary_pos_emb_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen3-VL text RoPE with fused BNSD rotary multiplication."""
    return (
        npu_rotary_mul_with_bsnd_fallback(q, cos, sin, unsqueeze_dim=1),
        npu_rotary_mul_with_bsnd_fallback(k, cos, sin, unsqueeze_dim=1),
    )


def apply_minimax_h3_qwen3vl_patch() -> None:
    """Route MiniMax H3 Qwen3-VL text RoPE to the Ascend fused operator."""
    global _ROPE_PATCHED
    if _ROPE_PATCHED:
        return

    from vllm_omni.diffusion.models.minimax_h3 import encoder

    encoder._apply_rotary_pos_emb = _apply_rotary_pos_emb_npu
    _ROPE_PATCHED = True
    logger.debug("Applied NPU fused RoPE patch for MiniMax H3 Qwen3-VL text encoder")


def npu_swiglu_from_packed(gate_up: torch.Tensor) -> torch.Tensor:
    """Apply fused SwiGLU to a packed gate/up projection tensor."""
    return torch_npu.npu_swiglu(gate_up, dim=-1)


def _forward_minimax_h3_qwen3vl_text_mlp_npu(self: Any, x: torch.Tensor) -> torch.Tensor:
    """Run the Qwen3-VL MLP with one packed GEMM and fused SwiGLU."""
    gate_up = F.linear(x, self.gate_up_proj.weight)
    return self.down_proj(npu_swiglu_from_packed(gate_up))


def apply_minimax_h3_qwen3vl_swiglu_patch() -> None:
    """Route MiniMax H3 Qwen3-VL text MLP to the Ascend fused SwiGLU path."""
    global _SWIGLU_PATCHED
    if _SWIGLU_PATCHED:
        return

    from vllm_omni.diffusion.models.minimax_h3 import encoder

    encoder.MiniMaxH3Qwen3VLTextMLP.forward = _forward_minimax_h3_qwen3vl_text_mlp_npu
    _SWIGLU_PATCHED = True
