# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in dense BF16 FlashAttention implementation for MiniMax-H3 on gfx950."""

import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl

from ..rocm_wide_fmha import _CUTS, _SHAPE, try_wide_fmha


def verified_wide_cut(bounds: torch.Tensor, cut: int, total: int) -> bool:
    """Check the actual two documents once before all the DiT blocks."""
    if total != _SHAPE[0] or cut not in _CUTS or bounds.shape != (3,) or bounds.dtype != torch.int32:
        return False
    # A graph replay may change the device prefixes without re-entering Python.
    if bounds.is_cuda and torch.cuda.is_current_stream_capturing():
        return False
    return bounds.tolist() == [0, cut, total]


class MiniMaxH3WideFlashImpl(FlashAttentionImpl):
    def forward_hip(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if attn_metadata is not None and (path := attn_metadata.extra.get("rocm_h3_wide_fmha_path")):
            extra = attn_metadata.extra
            if (
                self._kernel_variant == "aiter"
                and not (self.causal or self.is_cross_attn or self.fa_deterministic)
                and self.softmax_scale == 128**-0.5
                and extra.get("kv_cache_dtype") in (None, "float", "auto")
                and isinstance(extra.get("cu_seqlens_q"), torch.Tensor)
                and extra["cu_seqlens_q"] is extra.get("cu_seqlens_k")
                and extra.get("max_seqlen_q") == extra.get("max_seqlen_k") == extra.get("valid_kv_length")
                and "max_seqlen_q" in extra
                and attn_metadata.attn_mask is None
                and attn_metadata.full_attn_spans is None
                and attn_metadata.query_ranges is None
                and attn_metadata.joint_query is None
                and attn_metadata.joint_key is None
                and attn_metadata.joint_value is None
                and attn_metadata.joint_attn_mask is None
                and attn_metadata.cross_query_mask is None
                and not torch.are_deterministic_algorithms_enabled()
                and not torch.is_grad_enabled()
                and all(
                    t.layout == torch.strided
                    and t.shape == (1, *_SHAPE)
                    and t.device == query.device
                    and t.dtype == torch.bfloat16
                    and not t.requires_grad
                    for t in (query, key, value)
                )
            ):
                q, k, v = (t.squeeze(0) for t in (query, key, value))
                out = try_wide_fmha(q, k, v, extra["max_seqlen_q"], path)
                if out is not None:
                    return out.unsqueeze(0)
        return super().forward_hip(query, key, value, attn_metadata)
