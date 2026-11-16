# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packed prefix attention and its exact-length dense buckets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm_omni.diffusion.attention.backends.abstract import PACKED_CAUSAL_KEY, AttentionMetadata

if TYPE_CHECKING:
    from vllm_omni.diffusion.attention.layer import Attention


def sequence_attention(
    attn: Attention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    cu_query: torch.Tensor,
    cu_key: torch.Tensor,
    max_query: int,
    max_key: int,
    causal: bool,
    lengths: tuple[tuple[int, ...], tuple[int, ...]] | None,
) -> torch.Tensor:
    """Run independent Q/KV documents, with bottom-right causal alignment."""
    _, backend, spec = attn.effective_attention()
    if backend is not None and backend.supports_packed_prefix_kv(kv_cache_dtype=attn._effective_kv_cache_dtype()):
        metadata = AttentionMetadata(
            extra={
                "cu_seqlens_q": cu_query,
                "cu_seqlens_k": cu_key,
                "max_seqlen_q": max_query,
                "max_seqlen_k": max_key,
                PACKED_CAUSAL_KEY: causal,
                "attention_mask_mode": "none",
            }
        )
        return attn.forward(query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0), metadata).reshape_as(query)

    if lengths is None:
        lengths = (tuple(cu_query.diff().tolist()), tuple(cu_key.diff().tolist()))
    qlens, klens = lengths
    if len(qlens) != len(klens) or sum(qlens) != query.shape[0] or sum(klens) != key.shape[0]:
        raise ValueError("Packed prefix Q/K lengths do not cover their tensors")
    mask_capable = backend is not None and backend.supports_dense_4d_mask(spec)
    buckets: dict[tuple[int, int], list[tuple[int, int]]] = {}
    qoff = koff = 0
    for qlen, klen in zip(qlens, klens, strict=True):
        if qlen <= 0 or klen < qlen:
            raise ValueError("Prefix attention requires 0 < query length <= KV length")
        if causal and not mask_capable:
            # Each query becomes a non-causal segment over exactly its visible
            # KV prefix. This also supports dense kernels without mask/causal
            # support; no query is allowed to see future tokens.
            for pos in range(qlen):
                buckets.setdefault((1, klen - qlen + pos + 1), []).append((qoff + pos, koff))
        else:
            buckets.setdefault((qlen, klen), []).append((qoff, koff))
        qoff += qlen
        koff += klen
    result = torch.empty_like(query)
    for (qlen, klen), rows in buckets.items():
        qs = torch.tensor([q for q, _ in rows], device=query.device, dtype=torch.long)
        ks = torch.tensor([k for _, k in rows], device=key.device, dtype=torch.long)
        qi = (qs[:, None] + torch.arange(qlen, device=query.device)).flatten()
        ki = (ks[:, None] + torch.arange(klen, device=key.device)).flatten()
        mask = None
        if causal and mask_capable:
            mask = torch.arange(klen, device=query.device)[None, :] <= (
                torch.arange(qlen, device=query.device)[:, None] + klen - qlen
            )
            mask = mask[None, None]
        output = attn.forward(
            query.index_select(0, qi).reshape(len(rows), qlen, *query.shape[1:]),
            key.index_select(0, ki).reshape(len(rows), klen, *key.shape[1:]),
            value.index_select(0, ki).reshape(len(rows), klen, *value.shape[1:]),
            AttentionMetadata(
                attn_mask=mask, extra={"attention_mask_mode": "arbitrary" if mask is not None else "none"}
            ),
        )
        result.index_copy_(0, qi, output.reshape(len(rows) * qlen, *query.shape[1:]))
    return result
