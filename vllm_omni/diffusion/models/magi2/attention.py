# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""MAGI-2 packed-sequence metadata, model attention and Ulysses exchange."""

from __future__ import annotations

import weakref
from dataclasses import dataclass

import torch
import torch.nn as nn

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

from .layers import varlen_attention_with_sink
from .layers.native.attention_ops import _sink_lse
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


def _rank_sink(sink: torch.Tensor, group: Magi2ParallelGroup) -> torch.Tensor:
    """Select this rank's sink heads after the Ulysses head exchange."""

    if group.world_size == 1:
        return sink
    if sink.shape[-1] % group.world_size:
        raise ValueError("attention sink heads must divide across Ulysses ranks")
    return torch.chunk(sink, group.world_size, dim=-1)[group.rank].contiguous()


class _SinkLseCache:
    """One layer's rank-local sink log-sum-exp, reused across forward calls.

    The entry is reused only for the same sink tensor with an unchanged version
    counter, storage, layout, device and Ulysses rank. It holds a weak reference
    to the sink. Inference tensors have no version counter, so for them the
    storage stands in for it. Writes that bypass the version counter, through
    ``.data`` or in place on an inference tensor, are not detected. ``get``
    returns ``None`` (compute per call) while compiling, for tensor subclasses
    and while grad mode is enabled.
    """

    def __init__(self) -> None:
        self._sink: weakref.ReferenceType[torch.Tensor] | None = None
        self._key: tuple | None = None
        self._sink_lse: torch.Tensor | None = None

    def get(self, sink: torch.Tensor | None, group: Magi2ParallelGroup) -> torch.Tensor | None:
        if (
            sink is None
            or sink.numel() == 0
            or torch.compiler.is_compiling()
            or type(sink) not in (torch.Tensor, nn.Parameter)
            or torch.is_grad_enabled()
        ):
            return None
        key = (
            None if sink.is_inference() else sink._version,
            sink.data_ptr(),
            sink.device,
            sink.dtype,
            tuple(sink.shape),
            sink.stride(),
            group.rank,
            group.world_size,
        )
        if self._sink is None or self._sink() is not sink or self._key != key:
            self._sink_lse = _sink_lse(_rank_sink(sink, group))
            self._sink = weakref.ref(sink)
            self._key = key
        return self._sink_lse


def packed_attention_with_sink(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    varlen: VarlenHandler,
    *,
    softcap: float = -1.0,
    sink: torch.Tensor | None = None,
    sink_lse: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the selected platform layer after Ulysses head exchange.

    ``sink_lse`` optionally carries this rank's ``_sink_lse`` of ``sink``.
    """

    bounds_q, bounds_k, max_q, max_k = varlen.resolved(q.shape[0], k.shape[0])
    return varlen_attention_with_sink(
        q,
        k,
        v,
        cu_seqlens_q=bounds_q,
        cu_seqlens_k=bounds_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        softcap=softcap,
        sink=sink,
        sink_lse=sink_lse,
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
    sink_lse: torch.Tensor | None = None,
) -> torch.Tensor:
    """MAGI-2 attention with overlapping Ulysses CP/head exchange.

    ``sink_lse`` optionally carries this rank's ``_sink_lse`` of ``sink``.
    """

    group = group or get_magi2_ulysses_group()
    if isinstance(split_sizes, torch.Tensor):
        split_sizes = [int(v) for v in split_sizes.detach().cpu().tolist()]
    if group.world_size > 1:
        q, k, v = scatter_heads_gather_seqlen((q, k, v), split_sizes, group)
        if sink is not None:
            sink = _rank_sink(sink, group)
    output = packed_attention_with_sink(q, k, v, varlen, softcap=softcap, sink=sink, sink_lse=sink_lse)
    if group.world_size > 1:
        output = scatter_seqlen_gather_heads(output.contiguous(), split_sizes, group)
        assert isinstance(output, torch.Tensor)
    return output


class Magi2PackedAttentionKernel(nn.Module):
    """Model kernel plugged into the shared diffusion Attention contract."""

    def __init__(self, softcap: float) -> None:
        super().__init__()
        self.softcap = softcap
        self._sink_lse_cache = _SinkLseCache()

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
        group = get_magi2_ulysses_group()
        return ulysses_packed_attention_with_sink(
            query,
            key,
            value,
            varlen,
            split_sizes,
            softcap=self.softcap,
            sink=sink,
            group=group,
            sink_lse=self._sink_lse_cache.get(sink, group),
        )


__all__ = [
    "VarlenHandler",
    "Magi2PackedAttentionKernel",
    "packed_attention_with_sink",
    "ulysses_packed_attention_with_sink",
]
