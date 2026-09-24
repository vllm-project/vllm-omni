# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""Platform-neutral MAGI-2 packed attention and Ulysses orchestration.

Device-specific packed-attention kernels are selected by :mod:`.layers`.
This module owns only packed-sequence metadata, Ulysses communication, and the
shared diffusion Attention adapter.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

from .layers import varlen_attention_with_sink
from .parallel import (
    Magi2ParallelGroup,
    get_magi2_ulysses_group,
    scatter_heads_gather_seqlen,
    scatter_seqlen_gather_heads,
)


@dataclass(frozen=True)
class VarlenHandler:
    """Packed-sequence metadata consumed by MAGI-2 attention."""

    cu_seqlens_q: torch.Tensor | None
    cu_seqlens_k: torch.Tensor | None
    max_seqlen_q: int | None = None
    max_seqlen_k: int | None = None

    def resolved(self, q_tokens: int, k_tokens: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        if self.cu_seqlens_q is None:
            cu_q = torch.tensor([0, q_tokens], device="cpu", dtype=torch.int32)
        else:
            cu_q = self.cu_seqlens_q
        if self.cu_seqlens_k is None:
            cu_k = torch.tensor([0, k_tokens], device="cpu", dtype=torch.int32)
        else:
            cu_k = self.cu_seqlens_k
        max_q = self.max_seqlen_q
        max_k = self.max_seqlen_k
        if max_q is None:
            max_q = int(torch.diff(cu_q).max().item()) if cu_q.numel() > 1 else 0
        if max_k is None:
            max_k = int(torch.diff(cu_k).max().item()) if cu_k.numel() > 1 else 0
        return cu_q, cu_k, max_q, max_k


def packed_attention_with_sink(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    varlen: VarlenHandler,
    *,
    softcap: float = -1.0,
    sink: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the platform-selected packed attention after Ulysses exchange."""

    cu_q, cu_k, max_q, max_k = varlen.resolved(q.shape[0], k.shape[0])
    return varlen_attention_with_sink(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        softcap=softcap,
        sink=sink,
    )


def ulysses_packed_attention_with_sink(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    varlen: VarlenHandler,
    split_sizes: list[int] | torch.Tensor,
    *,
    softcap: float = -1.0,
    sink: torch.Tensor | None = None,
    group: Magi2ParallelGroup | None = None,
) -> torch.Tensor:
    """MAGI-2 attention with overlapping Ulysses CP/head exchange."""

    group = group or get_magi2_ulysses_group()
    if isinstance(split_sizes, torch.Tensor):
        split_sizes = [int(v) for v in split_sizes.detach().cpu().tolist()]
    if group.world_size > 1:
        q, k, v = scatter_heads_gather_seqlen((q, k, v), split_sizes, group)
        if sink is not None:
            if sink.shape[-1] % group.world_size:
                raise ValueError("attention sink heads must divide across Ulysses ranks")
            sink = torch.chunk(sink, group.world_size, dim=-1)[group.rank].contiguous()
    output = packed_attention_with_sink(q, k, v, varlen, softcap=softcap, sink=sink)
    if group.world_size > 1:
        output = scatter_seqlen_gather_heads(output.contiguous(), split_sizes, group)
        assert isinstance(output, torch.Tensor)
    return output


class Magi2PackedAttentionKernel(nn.Module):
    """Model kernel plugged into the shared diffusion Attention contract."""

    def __init__(self, softcap: float) -> None:
        super().__init__()
        self.softcap = softcap

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            raise ValueError("MAGI-2 packed attention requires attention metadata")
        varlen = attn_metadata.extra.get("magi2_varlen")
        split_sizes = attn_metadata.extra.get("magi2_split_sizes")
        sink = attn_metadata.extra.get("magi2_sink")
        if not isinstance(varlen, VarlenHandler):
            raise TypeError("magi2_varlen must be a VarlenHandler")
        if not isinstance(split_sizes, (list, torch.Tensor)):
            raise TypeError("magi2_split_sizes must be a list or tensor")
        if sink is not None and not isinstance(sink, torch.Tensor):
            raise TypeError("magi2_sink must be a tensor or None")
        return ulysses_packed_attention_with_sink(
            query,
            key,
            value,
            varlen,
            split_sizes,
            softcap=self.softcap,
            sink=sink,
        )


__all__ = [
    "VarlenHandler",
    "Magi2PackedAttentionKernel",
    "packed_attention_with_sink",
    "ulysses_packed_attention_with_sink",
]
