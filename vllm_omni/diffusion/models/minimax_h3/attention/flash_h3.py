# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax-H3 grouped ROCm FlashAttention for the packed Ulysses head shard."""

import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
from vllm_omni.diffusion.attention.backends.utils.rocm_h3_bounds import has_packed_fmha_bounds
from vllm_omni.platforms import current_omni_platform

# Only the ordinary two-document H3 owner publishes these fields. A mode
# adding further semantics (e.g. VDN, reference KV, or quantization) delegates
# to the selected FlashAttention implementation.
_GROUPED_KEYS = frozenset(
    {
        "cu_seqlens_q",
        "cu_seqlens_k",
        "max_seqlen_q",
        "max_seqlen_k",
        "minimax_h3_grouped_fmha",
        "accepts_strided_ulysses_qkv",
        "valid_kv_length",
        "npu_attn_varlen",
        "laser_input_scale",
        "vsa_h3_tile_size",
        "vsa_h3_ref_policy",
        "vsa_h3_ref_keep_rate",
    }
)


class MiniMaxH3FlashAttentionImpl(FlashAttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        *,
        backend_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        options = dict(backend_kwargs or {})
        grouped = options.pop("rocm_h3_grouped_fmha", False)
        super().__init__(num_heads, head_size, softmax_scale, backend_kwargs=options, **kwargs)
        self.rocm_h3_grouped_fmha = (
            grouped
            and current_omni_platform.is_rocm()
            and self._kernel_variant == "aiter"
            and not (self.causal or self.is_cross_attn or self.fa_deterministic)
            and head_size == 128
            and softmax_scale == 128**-0.5
        )

    def forward_hip(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if self.rocm_h3_grouped_fmha and attn_metadata is not None:
            extra = attn_metadata.extra
            if extra.get("minimax_h3_grouped_fmha") is True:
                cu = extra.get("cu_seqlens_q")
                max_seqlen = extra.get("max_seqlen_q")
                if (
                    attn_metadata.attn_mask is None
                    and attn_metadata.full_attn_spans is None
                    and attn_metadata.joint_attn_mask is None
                    and attn_metadata.joint_query is None
                    and attn_metadata.joint_key is None
                    and attn_metadata.joint_value is None
                    and attn_metadata.query_ranges is None
                    and attn_metadata.cross_query_mask is None
                    and extra.keys() <= _GROUPED_KEYS
                    and cu is not None
                    and cu is extra.get("cu_seqlens_k")
                    and max_seqlen == extra.get("max_seqlen_k") == extra.get("valid_kv_length")
                    and has_packed_fmha_bounds(cu, max_seqlen)
                ):
                    from vllm_omni.diffusion.attention.backends.utils.rocm_h3_fmha import try_grouped_fmha

                    output = try_grouped_fmha(query, key, value, cu, cu, max_seqlen=max_seqlen)
                    if output is not None:
                        return output

                # This rank can decline after Ulysses has delivered head planes
                # as views. Restore the dense layout the generic backend expects.
                query, key, value = query.contiguous(), key.contiguous(), value.contiguous()
        return super().forward_hip(query, key, value, attn_metadata)
