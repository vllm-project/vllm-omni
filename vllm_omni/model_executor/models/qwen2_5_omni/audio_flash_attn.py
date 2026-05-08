# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Override HF audio-encoder attention to use the project's FA dispatcher.

HF's ``Qwen*OmniAudioAttention.forward`` routes through ``_attn_implementation``
and only takes the FA path when the upstream ``flash_attn`` package is
installed. To drop that source-build dependency we class-swap each attention
instance to a subclass whose ``forward`` calls ``flash_attn_varlen_func`` from
``vllm_omni.diffusion.attention.backends.utils.fa`` directly.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from vllm_omni.diffusion.attention.backends.utils.fa import flash_attn_varlen_func


def _fa_audio_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,  # unused; kept for signature compat
    **_: Any,
) -> torch.Tensor:
    seq_length, _ = hidden_states.size()
    q = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
    k = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
    v = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)

    cu_seqlens = cu_seqlens.to(torch.int32)
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().to(torch.int32)

    attn_output = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        softmax_scale=self.scaling,
        causal=False,
    )

    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    return self.out_proj(attn_output)


def patch_audio_tower_attention(audio_tower: nn.Module, target_class: type) -> None:
    """Class-swap every ``target_class`` instance under ``audio_tower`` so its
    ``forward`` calls ``flash_attn_varlen_func``. Loaded weights and submodules
    are preserved — only the bound ``forward`` changes."""
    new_class = type(
        f"FA{target_class.__name__}",
        (target_class,),
        {"forward": _fa_audio_forward},
    )
    for module in audio_tower.modules():
        if type(module) is target_class:
            module.__class__ = new_class
