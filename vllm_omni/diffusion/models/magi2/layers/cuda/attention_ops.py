# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""CUDA packed attention for MAGI-2."""

from __future__ import annotations

import logging
import os
from functools import cache

import torch

from vllm_omni.diffusion.attention.backends.utils.fa import (
    resolve_vllm_flash_attn_version,
    vllm_flash_attn_varlen_with_lse,
)

from ..native.attention_ops import correct_out_lse_with_sink
from ..native.attention_ops import varlen_attention_with_sink as native_varlen_attention_with_sink

logger = logging.getLogger(__name__)


@cache
def _resolve_flash_attn_version() -> int:
    """Apply MAGI-2's operator override to the shared FA2/FA3/FA4 resolver."""

    requested = os.environ.get("MAGI2_FLASH_ATTN_VERSION")
    version = resolve_vllm_flash_attn_version(requested)
    logger.info("MAGI-2 selected FlashAttention %d", version)
    return version


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
    """Run vLLM FlashAttention and fold MAGI-2 sink logits into its LSE."""

    if not q.is_cuda:
        return native_varlen_attention_with_sink(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softcap=softcap,
            sink=sink,
        )
    if max_seqlen_q is None:
        max_seqlen_q = int(torch.diff(cu_seqlens_q).max().item()) if cu_seqlens_q.numel() > 1 else 0
    if max_seqlen_k is None:
        max_seqlen_k = int(torch.diff(cu_seqlens_k).max().item()) if cu_seqlens_k.numel() > 1 else 0
    cu_q = cu_seqlens_q.to(device=q.device, dtype=torch.int32).contiguous()
    cu_k = cu_seqlens_k.to(device=q.device, dtype=torch.int32).contiguous()
    out, lse = vllm_flash_attn_varlen_with_lse(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softcap=softcap,
        deterministic=os.environ.get("MAGI2_DETERMINISTIC", "0") == "1",
        fa_version=_resolve_flash_attn_version(),
    )
    return correct_out_lse_with_sink(out, lse, sink)[0]


__all__ = ["varlen_attention_with_sink"]
