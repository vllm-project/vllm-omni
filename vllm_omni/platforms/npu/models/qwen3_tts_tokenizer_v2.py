# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NPU patches for the Qwen3-TTS 12Hz tokenizer decoder."""

from __future__ import annotations

import torch
import torch_npu
from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False


def _bnsd_rotary_shape_is_supported(hidden_states: torch.Tensor) -> bool:
    """Return whether the fused half-mode BNSD RoPE tiler supports this shape.

    Qwen3-TTS uses ``[batch, heads, seq_len, head_dim]`` tensors.  For the
    aligned-head-dimension branch used by its 64-wide heads, CANN requires
    ``batch * heads <= seq_len * 8``.  Treat other alignments conservatively
    and use the BSND layout, which does not have this short-sequence limit.
    """
    if hidden_states.ndim != 4:
        return False

    batch_size, num_heads, seq_len, head_dim = hidden_states.shape
    element_size = hidden_states.element_size()
    if head_dim % 2 != 0 or element_size <= 0 or 32 % element_size != 0:
        return False

    half_dim_alignment = 32 // element_size
    return head_dim // 2 % half_dim_alignment == 0 and batch_size * num_heads <= seq_len * 8


def _rotary_mul_npu(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int,
) -> torch.Tensor:
    if unsqueeze_dim != 1 or _bnsd_rotary_shape_is_supported(hidden_states):
        return torch_npu.npu_rotary_mul(
            hidden_states,
            cos.unsqueeze(unsqueeze_dim),
            sin.unsqueeze(unsqueeze_dim),
        )

    hidden_states_bsnd = hidden_states.transpose(1, 2)
    if not hidden_states_bsnd.is_contiguous():
        hidden_states_bsnd = hidden_states_bsnd.contiguous()
    output_bsnd = torch_npu.npu_rotary_mul(
        hidden_states_bsnd,
        cos.unsqueeze(2),
        sin.unsqueeze(2),
    )
    return output_bsnd.transpose(1, 2)


def _apply_rotary_pos_emb_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids=None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    del position_ids
    return (
        _rotary_mul_npu(q, cos, sin, unsqueeze_dim),
        _rotary_mul_npu(k, cos, sin, unsqueeze_dim),
    )


def _rms_norm_forward_npu(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]


def apply_qwen3_tts_tokenizer_v2_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return

    from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz import (
        modeling_qwen3_tts_tokenizer_v2,
    )

    modeling_qwen3_tts_tokenizer_v2.apply_rotary_pos_emb = _apply_rotary_pos_emb_npu
    modeling_qwen3_tts_tokenizer_v2.Qwen3TTSTokenizerV2DecoderRMSNorm.forward = _rms_norm_forward_npu
    _PATCHED = True
    logger.debug("Applied NPU patch for Qwen3-TTS 12Hz tokenizer decoder")
