# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-local BF16 tile64 specialization of the Triton attention backend."""

import torch
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

from .tiled_attention_kernel import unified_attention


class MossTiledAttentionImpl(TritonAttentionImpl):
    """Keep upstream metadata/cache ownership and specialize only BF16 compute."""

    def forward(
        self, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None, output_block_scale=None
    ):
        supported = (
            query.dtype == torch.bfloat16
            and kv_cache.dtype == torch.bfloat16
            and query.shape[0] >= 32
            and query.shape[1:] == (32, 128)
            and attn_metadata is not None
            and attn_metadata.causal is True
            and attn_metadata.num_actual_tokens >= 32
            and not attn_metadata.use_cascade
            and attn_metadata.mm_prefix_range_tensor is None
            and attn_metadata.rswa_prefix_lens is None
            and output_scale is None
            and output_block_scale is None
        )
        if not supported:
            return super().forward(
                layer, query, key, value, kv_cache, attn_metadata, output, output_scale, output_block_scale
            )
        count = attn_metadata.num_actual_tokens
        # Same logical (blocks, heads, slots, 2*dim) layout as the upstream backend.
        keys, values = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        scale_shape = (attn_metadata.query_start_loc.shape[0] - 1, self.num_kv_heads)
        unified_attention(
            q=query[:count],
            k=keys,
            v=values,
            out=output[:count],
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=True,
            window_size=self.sliding_window,
            block_table=attn_metadata.block_table,
            softcap=0.0,
            q_descale=None,
            k_descale=layer._k_scale.expand(scale_shape),
            v_descale=layer._v_scale.expand(scale_shape),
            seq_threshold_3D=attn_metadata.seq_threshold_3D,
            num_par_softmax_segments=attn_metadata.num_par_softmax_segments,
            softmax_segm_output=attn_metadata.softmax_segm_output,
            softmax_segm_max=attn_metadata.softmax_segm_max,
            softmax_segm_expsum=attn_metadata.softmax_segm_expsum,
            use_td=self.use_td,
        )
        return output


def install(model: torch.nn.Module) -> int:
    """Replace eligible MOSS instances; never change upstream module globals."""
    count = 0
    for module in model.modules():
        original = getattr(module, "impl", None)
        if type(original) is not TritonAttentionImpl:
            continue
        if not (
            (original.num_heads, original.num_kv_heads, original.head_size) == (32, 8, 128)
            and original.attn_type == AttentionType.DECODER
            and original.kv_cache_dtype in ("auto", "bfloat16")
            and original.alibi_slopes is None
            and original.sinks is None
            and original.sliding_window == (-1, -1)
            and original.logits_soft_cap == 0.0
            and original.chunk_lookback == -1
        ):
            continue
        module.impl = MossTiledAttentionImpl(
            num_heads=original.num_heads,
            head_size=original.head_size,
            scale=original.scale,
            num_kv_heads=original.num_kv_heads,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=original.kv_cache_dtype,
            kv_sharing_target_layer_name=original.kv_sharing_target_layer_name,
        )
        count += 1
    return count
