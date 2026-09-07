# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""Native MAGI-2 packed attention with sinks and Ulysses exchange.

The sink and context-parallel math is adapted from SandAI's Apache-2.0
MAGI-2 preview implementation.  This version uses vLLM's bundled
FlashAttention extension and vLLM-Omni's existing Ulysses process group; the
PyTorch path is an exact, portable oracle for small tests.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import cache

import torch
import torch.nn as nn

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.utils.fa import (
    resolve_vllm_flash_attn_version,
    vllm_flash_attn_varlen_with_lse,
)

from .parallel import (
    Magi2ParallelGroup,
    get_magi2_ulysses_group,
    scatter_heads_gather_seqlen,
    scatter_seqlen_gather_heads,
)

logger = logging.getLogger(__name__)


@cache
def _resolve_flash_attn_version() -> int:
    """Apply MAGI-2's operator override to the shared FA2/FA3/FA4 resolver."""

    requested = os.environ.get("MAGI2_FLASH_ATTN_VERSION")
    version = resolve_vllm_flash_attn_version(requested)
    logger.info("MAGI-2 selected FlashAttention %d", version)
    return version


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


def rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    interleaved: bool = False,
) -> torch.Tensor:
    """Apply the released element-wise RoPE layout to ``[...,H,D]``."""

    rotary_dim = cos.shape[-1] * 2
    if rotary_dim > x.shape[-1]:
        raise ValueError(f"RoPE dimension {rotary_dim} exceeds head dimension {x.shape[-1]}")
    if interleaved:
        cos = cos.unsqueeze(-2).repeat_interleave(2, dim=-1)
        sin = sin.unsqueeze(-2).repeat_interleave(2, dim=-1)
    else:
        cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
        sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2)
    rotated = x[..., :rotary_dim] * cos + rotate_half(x[..., :rotary_dim], interleaved) * sin
    return torch.cat((rotated, x[..., rotary_dim:]), dim=-1)


def correct_out_lse_with_sink(
    out: torch.Tensor,
    lse: torch.Tensor,
    sink: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add zero-valued attention sinks to an already-computed softmax.

    FlashAttention returns ``out[T,H,D]`` and conventionally ``lse[H,T]``.
    A MAGI sink contains additional logits ``[num_sink,H]`` whose values are
    zero vectors, so only the denominator and LSE change.
    """

    if sink is None or sink.numel() == 0:
        return out, lse
    if out.ndim != 3 or lse.ndim != 2 or sink.ndim != 2:
        raise ValueError(
            "expected out[T,H,D], lse[H,T], sink[num_sink,H], got "
            f"{tuple(out.shape)}, {tuple(lse.shape)}, {tuple(sink.shape)}"
        )
    old_lse = lse.float().transpose(0, 1)
    sink_lse = torch.logsumexp(sink.float(), dim=0).unsqueeze(0)
    if old_lse.shape[-1] != sink_lse.shape[-1]:
        raise ValueError("attention sink and FlashAttention head counts differ")
    new_lse = torch.logaddexp(old_lse, sink_lse)
    delta = old_lse - new_lse
    delta = torch.where(torch.isfinite(delta), delta, torch.full_like(delta, -torch.inf))
    corrected = out * torch.exp(delta).unsqueeze(-1).to(out.dtype)
    return corrected, new_lse.transpose(0, 1).contiguous()


def _repeat_kv_heads(tensor: torch.Tensor, query_heads: int) -> torch.Tensor:
    kv_heads = tensor.shape[1]
    if kv_heads == query_heads:
        return tensor
    if query_heads % kv_heads:
        raise ValueError(f"query heads {query_heads} must be divisible by KV heads {kv_heads}")
    return tensor.repeat_interleave(query_heads // kv_heads, dim=1)


def torch_varlen_attention_with_sink(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    softcap: float = -1.0,
    sink: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference packed attention, including GQA and sink logits."""

    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("packed attention expects q/k/v shaped [tokens,heads,dim]")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError("query and key cumulative-length arrays must contain the same batch count")
    output = torch.empty_like(q)
    scale = q.shape[-1] ** -0.5
    for batch_idx in range(cu_seqlens_q.numel() - 1):
        q_start, q_end = (int(v) for v in cu_seqlens_q[batch_idx : batch_idx + 2].tolist())
        k_start, k_end = (int(v) for v in cu_seqlens_k[batch_idx : batch_idx + 2].tolist())
        q_part = q[q_start:q_end].float()
        k_part = _repeat_kv_heads(k[k_start:k_end], q.shape[1]).float()
        v_part = _repeat_kv_heads(v[k_start:k_end], q.shape[1]).float()
        scores = torch.einsum("qhd,khd->hqk", q_part, k_part) * scale
        if softcap > 0:
            scores = softcap * torch.tanh(scores / softcap)
        if sink is not None and sink.numel() > 0:
            sink_scores = sink.float().transpose(0, 1).unsqueeze(1).expand(-1, q_part.shape[0], -1)
            probabilities = torch.softmax(torch.cat((scores, sink_scores), dim=-1), dim=-1)[..., : k_part.shape[0]]
        else:
            probabilities = torch.softmax(scores, dim=-1)
        output[q_start:q_end] = torch.einsum("hqk,khd->qhd", probabilities, v_part).to(output.dtype)
    return output


def packed_attention_with_sink(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    varlen: VarlenHandler,
    *,
    softcap: float = -1.0,
    sink: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run packed attention on one rank after Ulysses head exchange."""

    cu_q, cu_k, max_q, max_k = varlen.resolved(q.shape[0], k.shape[0])
    cu_q = cu_q.to(device=q.device, dtype=torch.int32).contiguous()
    cu_k = cu_k.to(device=q.device, dtype=torch.int32).contiguous()
    if q.is_cuda:
        out, lse = vllm_flash_attn_varlen_with_lse(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            softcap=softcap,
            deterministic=os.environ.get("MAGI2_DETERMINISTIC", "0") == "1",
            fa_version=_resolve_flash_attn_version(),
        )
        return correct_out_lse_with_sink(out, lse, sink)[0]
    return torch_varlen_attention_with_sink(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
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
    "apply_rotary_emb",
    "correct_out_lse_with_sink",
    "packed_attention_with_sink",
    "torch_varlen_attention_with_sink",
    "ulysses_packed_attention_with_sink",
]
