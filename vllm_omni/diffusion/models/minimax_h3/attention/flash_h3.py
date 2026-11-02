# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-owned BF16 prefix attention for H3's explicit NPU fused selection."""

from __future__ import annotations

import os

import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl


def _prefix_bnsd(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float) -> torch.Tensor:
    from mindiesd import attention_forward

    batch, rows, heads, dim = query.shape

    def fold(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.transpose(1, 2).contiguous().reshape(batch * heads, 1, tensor.shape[1], dim)

    output = attention_forward(
        fold(query),
        fold(key),
        fold(value),
        attn_mask=None,
        scale=scale,
        head_first=True,
        opt_mode="manual",
        op_type="fused_attn_score",
        layout="BNSD",
    )
    # Match the original packed varlen output layout before shared SP restores
    # local sequence shards. Padding query rows are not consumed by H3.
    return output.reshape(batch, heads, rows, dim).transpose(1, 2).contiguous()


class MiniMaxH3FlashAttentionImpl(FlashAttentionImpl):
    """Use independent head batches for explicit long BF16 prefix attention.

    Default and other operator selections preserve their parent behavior.
    Mathematical equivalence alone does not promise bit-exact model outputs;
    each numerical qualification must cover the actual workload.
    """

    def forward_fa_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if os.environ.get("MINDIE_SD_FA_TYPE") != "fused_attn_score" or attn_metadata is None:
            return super().forward_fa_npu(query, key, value, attn_metadata)
        extra = attn_metadata.extra
        if (
            self.causal
            # Keep the short text refiners on their original varlen kernels.
            or query.shape[1] < 4096
            or attn_metadata.attn_mask is not None
            or attn_metadata.full_attn_spans is not None
            or attn_metadata.query_ranges is not None
            or not extra.get("npu_attn_varlen", False)
            or extra.get("minimax_h3_compact_reference_rows", 0)
            or query.dtype != torch.bfloat16
            or query.dtype != key.dtype
            or query.dtype != value.dtype
            or query.shape[2:] != key.shape[2:]
            or query.shape[2:] != value.shape[2:]
            or key.shape[1] != value.shape[1]
        ):
            return super().forward_fa_npu(query, key, value, attn_metadata)
        resolved = self._resolve_packed_seq_npu(query, key, extra)
        if resolved is None:
            return super().forward_fa_npu(query, key, value, attn_metadata)
        used_k = resolved[1][0]
        return _prefix_bnsd(query, key[:, :used_k], value[:, :used_k], self.softmax_scale)
