# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""MUSA FlashAttention-3 with FP32 sink correction and exact short-sequence fallback."""

from __future__ import annotations

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.utils.fa import (
    flash_attn_3_varlen,
    flash_attn_3_varlen_unsupported,
)
from vllm_omni.platforms import current_omni_platform

from ..native.attention_ops import correct_out_lse_with_sink, torch_varlen_attention_with_sink

logger = init_logger(__name__)

# MATE's Mubin kernel truncates BF16 probabilities for short packed sequences.
_MUSA_FA3_MIN_TOKENS = 32


def varlen_attention_with_sink(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    softcap: float = -1.0,
    sink: torch.Tensor | None = None,
) -> torch.Tensor:
    """Preserve MUSA eligibility, provider-failure propagation and sink precision."""

    if max_seqlen_q is None:
        max_seqlen_q = int(torch.diff(cu_seqlens_q).max().item()) if cu_seqlens_q.numel() > 1 else 0
    if max_seqlen_k is None:
        max_seqlen_k = int(torch.diff(cu_seqlens_k).max().item()) if cu_seqlens_k.numel() > 1 else 0
    if (
        current_omni_platform.is_musa()
        and q.device.type == current_omni_platform.device_type
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.shape[0] > 0
        and k.shape[0] > 0
        and max_seqlen_q >= _MUSA_FA3_MIN_TOKENS
    ):
        unsupported = flash_attn_3_varlen_unsupported(softcap > 0, True)
        if unsupported is None:
            out, lse = flash_attn_3_varlen(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q.to(device=q.device, dtype=torch.int32).contiguous(),
                cu_seqlens_k=cu_seqlens_k.to(device=q.device, dtype=torch.int32).contiguous(),
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=q.shape[-1] ** -0.5,
                softcap=softcap,
                return_softmax_lse=True,
            )
            # Correct the learned sinks in FP32; FA3 itself runs in the activation dtype.
            if sink is not None:
                out = out.float()
            return correct_out_lse_with_sink(out, lse, sink)[0].to(q.dtype)
        logger.warning_once(
            "MAGI-2 FlashAttention-3 is unavailable on MUSA; using Torch reference attention: %s", unsupported
        )
    return torch_varlen_attention_with_sink(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q.cpu(),
        cu_seqlens_k=cu_seqlens_k.cpu(),
        softcap=softcap,
        sink=sink,
    )


__all__ = ["varlen_attention_with_sink"]
