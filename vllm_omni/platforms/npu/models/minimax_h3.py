# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NPU patches for the MiniMax H3 Qwen3-VL text encoder."""

from __future__ import annotations

import torch
from vllm.logger import init_logger

from vllm_omni.platforms.npu.layers.rotary_embedding import (
    npu_rotary_mul_with_bsnd_fallback,
)

logger = init_logger(__name__)

_PATCHED = False


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
    global _PATCHED
    if _PATCHED:
        return

    from vllm_omni.diffusion.models.minimax_h3 import encoder

    encoder._apply_rotary_pos_emb = _apply_rotary_pos_emb_npu
    _PATCHED = True
    logger.debug("Applied NPU fused RoPE patch for MiniMax H3 Qwen3-VL text encoder")
