# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Released short-decode FA2 reduction on the current native paged cache."""

from __future__ import annotations

import torch
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl


class LycheeFlashAttentionImpl(FlashAttentionImpl):
    """Retain the released 128-wide K tile for short A100 BF16 decode.

    The current FA2 single-split path uses a 64-wide tile for head size 128.
    Two splits select its 128-wide path; for at most 128 keys the second
    split is empty. Native metadata, registry and KV updates remain owned by
    the original backend. Prefill and unsupported configurations delegate.
    """

    def _use_released_split(self, query, kv_cache, metadata, output_scale, output_block_scale):
        if metadata is None or output_scale is not None or output_block_scale is not None:
            return False
        requests = metadata.seq_lens.shape[0]
        return (
            query.is_cuda
            and query.dtype == torch.bfloat16
            and kv_cache.dtype == query.dtype
            and query.ndim == 3
            and query.shape[1:] == (28, 128)
            and kv_cache.ndim == 4
            and kv_cache.shape[1] == 4
            and kv_cache.shape[2] in (16, 128)
            and kv_cache.shape[3] == 256
            and self.vllm_flash_attn_version == 2
            and self.attn_type == AttentionType.DECODER
            and self.num_heads == 28
            and self.num_kv_heads == 4
            and self.head_size == 128
            and self.kv_cache_dtype in ("auto", "bfloat16")
            and self.dcp_world_size == 1
            and not self.batch_invariant_enabled
            and self.alibi_slopes is None
            and self.sliding_window == (-1, -1)
            and self.logits_soft_cap == 0
            and self.sinks is None
            and self.kv_sharing_target_layer_name is None
            and 1 <= requests <= 4
            and metadata.num_actual_tokens == requests
            and query.shape[0] >= requests
            and metadata.query_start_loc.shape == (requests + 1,)
            and metadata.block_table.shape[0] == requests
            and metadata.max_query_len == 1
            and 1 <= metadata.max_seq_len <= 128
            and metadata.causal is True
            and not metadata.use_cascade
            and metadata.max_num_splits == 0
            and metadata.scheduler_metadata is None
            and metadata.mm_prefix_query_range_tensor is None
            and metadata.rswa_prefix_lens is None
        )

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        if not self._use_released_split(query, kv_cache, attn_metadata, output_scale, output_block_scale):
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
            )
        n = attn_metadata.num_actual_tokens
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        # FA2 ignores cu_seqlens_k when seqused_k is supplied. Reusing the
        # native int32 Q offsets avoids a per-layer dummy CUDA allocation.
        torch.ops._vllm_fa2_C.varlen_fwd(
            query[:n],
            key_cache,
            value_cache,
            output[:n],
            attn_metadata.query_start_loc,
            attn_metadata.query_start_loc,
            attn_metadata.seq_lens,
            None,
            attn_metadata.block_table,
            None,
            1,
            attn_metadata.max_seq_len,
            0.0,
            self.scale,
            False,
            True,
            -1,
            -1,
            0.0,
            False,
            2,
            None,
        )
        return output


def adapt_released_attention(impl):
    """Wrap only this model's already-configured native FA2 implementation."""
    if type(impl) is not FlashAttentionImpl or not current_platform.is_device_capability(80):
        return impl
    adapted = object.__new__(LycheeFlashAttentionImpl)
    adapted.__dict__.update(impl.__dict__)
    return adapted
