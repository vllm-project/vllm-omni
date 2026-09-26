# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared text KV attention with log-sum-exp state merging."""

import torch
from vllm import _custom_ops as ops
from vllm.vllm_flash_attn import flash_attn_varlen_func

from .na_ops import LocalWindowContext


def shared_text_attention(
    query: torch.Tensor,
    video_key: torch.Tensor,
    video_value: torch.Tensor,
    text_key: torch.Tensor,
    text_value: torch.Tensor,
    ctx: LocalWindowContext,
    softmax_scale: float,
) -> torch.Tensor:
    video, video_lse = flash_attn_varlen_func(
        query,
        video_key,
        video_value,
        max_seqlen_q=ctx.max_joint_len,
        cu_seqlens_q=ctx.joint_cu_seqlens,
        max_seqlen_k=ctx.max_joint_len - ctx.text_len,
        cu_seqlens_k=ctx.video_cu_seqlens,
        softmax_scale=softmax_scale,
        return_softmax_lse=True,
    )
    text, text_lse = flash_attn_varlen_func(
        query,
        text_key,
        text_value,
        max_seqlen_q=ctx.joint_len,
        cu_seqlens_q=ctx.prefix_query_offsets,
        max_seqlen_k=ctx.text_len,
        cu_seqlens_k=ctx.prefix_key_offsets,
        softmax_scale=softmax_scale,
        return_softmax_lse=True,
    )
    output = torch.empty_like(query)
    ops.merge_attn_states(output, text, text_lse, video, video_lse)
    return output
