# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""Portable PyTorch packed-attention implementation for MAGI-2."""

from __future__ import annotations

import torch


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
        q_start, q_end = (int(value) for value in cu_seqlens_q[batch_idx : batch_idx + 2].tolist())
        k_start, k_end = (int(value) for value in cu_seqlens_k[batch_idx : batch_idx + 2].tolist())
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
    """Platform API for the Native packed-attention fallback."""

    del max_seqlen_q, max_seqlen_k
    return torch_varlen_attention_with_sink(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        softcap=softcap,
        sink=sink,
    )


__all__ = [
    "correct_out_lse_with_sink",
    "torch_varlen_attention_with_sink",
    "varlen_attention_with_sink",
]
