# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NPU patches for the Qwen3-TTS 12Hz tokenizer decoder."""

from __future__ import annotations

import torch
import torch_npu
from vllm.logger import init_logger

logger = init_logger(__name__)

_PATCHED = False


def _apply_rotary_pos_emb_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids=None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    del position_ids
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return torch_npu.npu_rotary_mul(q, cos, sin), torch_npu.npu_rotary_mul(k, cos, sin)


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
