# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2026 SandAI. All Rights Reserved.

"""Ascend fused packed attention for MAGI-2."""

from __future__ import annotations

import logging
from functools import cache

import torch

from ..native.attention_ops import varlen_attention_with_sink as native_varlen_attention_with_sink

logger = logging.getLogger(__name__)


@cache
def _log_npu_attention_backend() -> None:
    logger.info("MAGI-2 selected Ascend fused-infer-attention-v2 (TND varlen with learnable sink)")


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
    """Run Ascend fused-infer-attention-v2 with MAGI-2 sink semantics."""

    # The fused NPU operator has no positive-softcap contract.
    if q.device.type != "npu" or softcap > 0:
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
    del max_seqlen_q, max_seqlen_k
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("packed attention expects q/k/v shaped [tokens,heads,dim]")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError("query and key cumulative-length arrays must contain the same batch count")

    import torch_npu

    actual_q = [int(length) for length in cu_seqlens_q.detach().cpu().tolist()[1:]]
    actual_kv = [int(length) for length in cu_seqlens_k.detach().cpu().tolist()[1:]]
    learnable_sink = None
    if sink is not None and sink.numel() > 0:
        if sink.ndim != 2 or sink.shape[-1] != q.shape[1]:
            raise ValueError(
                "attention sink must be [num_sink,query_heads], got "
                f"{tuple(sink.shape)} for {q.shape[1]} heads"
            )
        # The Ascend op accepts one zero-value sink logit per query head.
        # logsumexp exactly folds multiple MAGI sink logits into that one logit.
        learnable_sink = torch.logsumexp(sink.float(), dim=0)
        learnable_sink = learnable_sink.to(device=q.device, dtype=torch.bfloat16).contiguous()

    _log_npu_attention_backend()
    output, _ = torch_npu.npu_fused_infer_attention_score_v2(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        actual_seq_qlen=actual_q,
        actual_seq_kvlen=actual_kv,
        learnable_sink=learnable_sink,
        num_query_heads=q.shape[1],
        num_key_value_heads=k.shape[1],
        softmax_scale=q.shape[-1] ** -0.5,
        input_layout="TND",
        sparse_mode=0,
        return_softmax_lse=False,
    )
    return output


__all__ = ["varlen_attention_with_sink"]
