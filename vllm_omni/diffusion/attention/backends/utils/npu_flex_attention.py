# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compiled Qwen-Image 2.1 NPU prefill, selected explicitly in model source."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

_BLOCK_SIZE = 128
_compiled_flex = None


def _make_span_mask(spans: tuple[tuple[int, int], ...], seq_len: int, device: torch.device | str = "cpu"):
    # Capture tensor data rather than Python boundaries so requests with new
    # lengths, image counts and spans can reuse the compiled attention graph.
    padded_len = (seq_len + _BLOCK_SIZE - 1) // _BLOCK_SIZE * _BLOCK_SIZE
    # -1 denotes text; -2 hides padding in the partially occupied final block.
    image_ids = torch.full((padded_len,), -2, dtype=torch.int32, device="cpu")
    image_ids[:seq_len] = -1
    for image_id, (start, end) in enumerate(spans):
        image_ids[start:end] = image_id
    image_ids = image_ids.to(device)

    def mask_mod(b, h, q_idx, kv_idx):
        q_image = image_ids[q_idx]
        kv_image = image_ids[kv_idx]
        same_image = (q_image >= 0) & (q_image == kv_image)
        return ((q_idx >= kv_idx) | same_image) & (q_image != -2) & (kv_image != -2)

    return mask_mod


def _can_use_flex(query, key, value, metadata: AttentionMetadata) -> bool:
    # The supported contract is a single, unpadded BF16 prefill. Parallel/query
    # slices, masks, decode and quantized attention keep their existing paths.
    if (
        query.device.type != "npu"
        or query.ndim != 4
        or query.shape != key.shape
        or key.shape != value.shape
        or query.shape[0] != 1
        or query.dtype != torch.bfloat16
        or key.dtype != query.dtype
        or value.dtype != query.dtype
        or key.device != query.device
        or value.device != query.device
        or metadata.full_attn_spans is None
        or len(metadata.full_attn_spans) != 1
        or metadata.attn_mask is not None
        or metadata.query_ranges is not None
        or metadata.joint_query is not None
        or metadata.joint_key is not None
        or metadata.joint_value is not None
        or metadata.extra.get("kv_cache_dtype") not in (None, "float", "auto")
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
    ):
        return False
    cursor = 0
    for start, end in metadata.full_attn_spans[0]:
        if not (cursor <= start < end <= query.shape[1]):
            return False
        cursor = end
    return query.shape[1] > 0


def try_npu_flex_attention(query, key, value, metadata: AttentionMetadata, softmax_scale: float):
    """Use compiled Flex for supported inputs; return None for other contracts.

    BlockMask lives in this prefill's shared metadata, so transformer layers
    reuse it without retaining request tensors in a process-wide mask cache.
    Compilation and execution errors propagate to the caller.
    """
    global _compiled_flex
    if not _can_use_flex(query, key, value, metadata):
        return None
    # Register NPU FlexAttention validation and lowering before importing the
    # public API or entering Dynamo; lazy backend loading happens too late.
    import torch_npu._inductor  # noqa: F401
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    spans = tuple(metadata.full_attn_spans[0])
    mask_key = (spans, query.shape[1], str(query.device))
    # Attention may shallow-copy extra to attach its per-layer dtype.
    # Keep the cache inside a shared dict across those copies as well.
    cache = metadata.extra.setdefault("_qwen_image21_flex_cache", {})
    cached = cache.get("mask")
    if cached is None or cached[0] != mask_key:
        block_mask = create_block_mask(
            _make_span_mask(spans, query.shape[1], query.device),
            B=1,
            H=1,
            Q_LEN=query.shape[1],
            KV_LEN=key.shape[1],
            device=str(query.device),
            BLOCK_SIZE=_BLOCK_SIZE,
        )
        cache["mask"] = (mask_key, block_mask)
    else:
        block_mask = cached[1]
    if _compiled_flex is None:
        _compiled_flex = torch.compile(flex_attention, backend="inductor", fullgraph=True, dynamic=True)
    output = _compiled_flex(
        query.transpose(1, 2).contiguous(),
        key.transpose(1, 2).contiguous(),
        value.transpose(1, 2).contiguous(),
        block_mask=block_mask,
        scale=softmax_scale,
    )
    return output.transpose(1, 2).contiguous()
