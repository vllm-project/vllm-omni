# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Respect HY3 image-prefix attention spans with the native Ascend cache."""

from functools import wraps
from types import MethodType


def image_segments(metadata):
    starts = [0] + list(metadata.actual_seq_lengths_q)
    for request, spans in (getattr(metadata, "mm_prefix_range", None) or {}).items():
        begin, end = starts[request : request + 2]
        cached = int(metadata.seq_lens_list[request]) - (end - begin)
        for image_start, image_end in spans:
            left = max(int(image_start), cached)
            right = min(int(image_end) + 1, int(metadata.seq_lens_list[request]))
            if left >= right:
                continue
            if image_end >= metadata.seq_lens_list[request]:
                raise ValueError("Cannot attend to an incomplete bidirectional image span")
            yield request, begin + left - cached, begin + right - cached, int(image_end) + 1


def correct_image_queries(query, key_cache, value_cache, metadata, output, *, num_heads, num_kv_heads, scale, fia):
    segments = list(image_segments(metadata))
    if not segments:
        return output
    blocks, block_size = key_cache.shape[:2]
    keys = key_cache.view(blocks, block_size, -1)
    values = value_cache.view(blocks, block_size, -1)
    for request, begin, end, prefix_length in segments:
        result, _ = fia(
            query=query[begin:end],
            key=keys,
            value=values,
            block_table=metadata.block_tables[request : request + 1],
            block_size=block_size,
            input_layout="TND",
            actual_seq_lengths=[end - begin],
            actual_seq_lengths_kv=[prefix_length],
            num_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            scale=scale,
            sparse_mode=0,
        )
        output[begin:end].copy_(result.reshape_as(output[begin:end]))
    return output


def enable_image_prefix_lm(attention):
    """Adapt this AR layer instance; preserve upstream cache writes and decode."""
    import torch_npu
    from vllm_ascend.attention.attention_v1 import AscendAttentionBackendImpl, AscendAttentionState

    impl = attention.impl
    if not isinstance(impl, AscendAttentionBackendImpl):
        raise TypeError(f"Unsupported HY3 image-prefix attention backend: {type(impl)}")
    parallel = impl.vllm_config.parallel_config
    if any(
        getattr(parallel, name, 1) != 1 for name in ("prefill_context_parallel_size", "decode_context_parallel_size")
    ):
        raise ValueError("HY3 Ascend image prefixes currently require PCP=1 and DCP=1")
    if getattr(impl, "_hy3_image_prefix_enabled", False):
        return
    original = impl.forward_fused_infer_attention

    @wraps(original)
    def forward(self, query, key, value, attn_metadata, output, kv_cache=None):
        result = original(query, key, value, attn_metadata, output, kv_cache)
        if attn_metadata.attn_state == AscendAttentionState.DecodeOnly or not getattr(
            attn_metadata, "mm_prefix_range", None
        ):
            return result
        return correct_image_queries(
            query,
            self.key_cache,
            self.value_cache,
            attn_metadata,
            result,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            scale=self.scale,
            fia=torch_npu.npu_fused_infer_attention_score,
        )

    impl.forward_fused_infer_attention = MethodType(forward, impl)
    impl._hy3_image_prefix_enabled = True
