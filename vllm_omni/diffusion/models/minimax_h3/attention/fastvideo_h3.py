# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax-H3 VSA prefix routing, tile layout, and learned compression gate."""

from __future__ import annotations

import functools
import math

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.fastvideo_vsa import (
    FastVideoVSAImpl,
    _construct_variable_block_sizes,
    _get_gate_compress,
    _get_non_pad_index,
    _get_tile_partition_indices,
)
from vllm_omni.diffusion.attention.ops.block_sparse import (
    build_prefix_dense_block_map,
    fastvideo_block_sparse_attn_bshd,
    mean_pool_tiles,
)

logger = init_logger(__name__)


@functools.lru_cache(maxsize=32)
def _get_h3_tile_metadata(
    prefix_segments: tuple[int, ...],
    video_shape: tuple[int, int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Official FastVideo H3 geometry: pure prefix chunks + 3-D video tiles."""
    block_size = (4, 4, 4)
    block_elements = 64
    prefix_len = sum(prefix_segments)
    prefix_sizes: list[int] = []
    for segment in prefix_segments:
        full, remainder = divmod(segment, block_elements)
        prefix_sizes.extend([block_elements] * full)
        if remainder:
            prefix_sizes.append(remainder)

    video_indices = _get_tile_partition_indices(video_shape, block_size, device) + prefix_len
    video_sizes = _construct_variable_block_sizes(video_shape, block_size, device)
    partition = torch.cat([torch.arange(prefix_len, device=device, dtype=torch.long), video_indices])
    sizes = torch.cat([torch.tensor(prefix_sizes, device=device, dtype=torch.int32), video_sizes.to(torch.int32)])
    non_pad = _get_non_pad_index(sizes, block_elements)
    untile = non_pad[torch.argsort(partition)]
    total = prefix_len + math.prod(video_shape)
    if int(sizes.sum()) != total or untile.numel() != total:
        raise ValueError(
            f"invalid H3 VSA geometry: prefix={prefix_segments}, video={video_shape}, "
            f"sizes_sum={int(sizes.sum())}, total={total}"
        )
    return partition, sizes, non_pad, untile, len(prefix_sizes), int(video_sizes.numel())


def _get_h3_layout(
    attn_metadata: AttentionMetadata | None,
) -> tuple[tuple[int, ...], tuple[int, int, int], int] | None:
    if attn_metadata is None or attn_metadata.video_layout is None:
        return None
    prefix = attn_metadata.extra.get("vsa_h3_prefix_segments")
    if not isinstance(prefix, (tuple, list)):
        return None
    target = next(
        (span for span in reversed(attn_metadata.video_layout.video_spans) if span.role == "target"),
        None,
    )
    if target is None:
        return None
    return tuple(int(x) for x in prefix if int(x) > 0), target.latent_grid, target.start


class MiniMaxH3VSAImpl(FastVideoVSAImpl):
    """Apply H3 tile64 routing after shared parallel attention dispatch."""

    def _forward_h3(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        layout = _get_h3_layout(attn_metadata)
        if layout is None:
            raise ValueError("incomplete VSA-H3 layout metadata")
        prefix_segments, video_shape, target_start = layout
        if sum(prefix_segments) != target_start:
            raise ValueError(f"VSA-H3 prefix segments sum to {sum(prefix_segments)}, target starts at {target_start}")
        expected = target_start + math.prod(video_shape)
        if query.shape[1] != expected:
            raise ValueError(f"VSA-H3 layout has {expected} rows but attention received {query.shape[1]}")
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"VSA-H3 requires fp16/bf16 tensors, got {query.dtype}")
        gate = _get_gate_compress(attn_metadata)
        if gate is not None:
            # H3 pads the packed document to 64 rows, while VSA operates on
            # the valid prefix. Validate/slice before launching attention so a
            # metadata error can never trigger an unsafe dense fallback after
            # an asynchronous custom kernel.
            if gate.shape[0] != query.shape[0] or gate.shape[2:] != query.shape[2:] or gate.shape[1] < query.shape[1]:
                raise ValueError(f"gate_compress shape {gate.shape} cannot cover query shape {query.shape}")
            gate = gate[:, : query.shape[1]]

        partition, sizes, non_pad, untile, prefix_blocks, video_blocks = _get_h3_tile_metadata(
            prefix_segments, video_shape, query.device
        )
        logical_blocks = int(sizes.numel())
        # The native sm100a kernel assigns pairs of query blocks to CTAs. Its
        # contract requires an even block count; the synthetic partner is
        # transport-only and is removed before returning.
        pair_pad = logical_blocks % 2
        kernel_blocks = logical_blocks + pair_pad
        target_shape = (query.shape[0], kernel_blocks * 64, query.shape[2], query.shape[3])
        q_tiled = torch.zeros(target_shape, device=query.device, dtype=query.dtype)
        k_tiled = torch.zeros_like(q_tiled)
        v_tiled = torch.zeros_like(q_tiled)
        q_tiled[:, non_pad] = query[:, partition]
        k_tiled[:, non_pad] = key[:, partition]
        v_tiled[:, non_pad] = value[:, partition]

        q_pool = mean_pool_tiles(q_tiled[:, : logical_blocks * 64], sizes, block_size=64)
        k_pool = mean_pool_tiles(k_tiled[:, : logical_blocks * 64], sizes, block_size=64)
        scores = torch.matmul(q_pool, k_pool.transpose(-2, -1)) * self.softmax_scale
        block_map = build_prefix_dense_block_map(scores, prefix_blocks, video_blocks, self.topk)
        kernel_sizes = sizes
        if pair_pad:
            block_map = torch.nn.functional.pad(block_map, (0, 1, 0, 1), value=False)
            kernel_sizes = torch.nn.functional.pad(sizes, (0, 1), value=0)

        logger.info_once(
            "FASTVIDEO_VSA H3 routing: seq_len=%d, prefix_segments=%s, video_shape=%s, "
            "prefix_blocks=%d, video_blocks=%d, topk=%d, kernel_blocks=%d",
            query.shape[1],
            prefix_segments,
            video_shape,
            prefix_blocks,
            video_blocks,
            min(self.topk, video_blocks),
            kernel_blocks,
        )
        output = fastvideo_block_sparse_attn_bshd(
            q_tiled.contiguous(),
            k_tiled.contiguous(),
            v_tiled.contiguous(),
            block_map.contiguous(),
            kernel_sizes.contiguous(),
            logical_blocks,
        )[:, : logical_blocks * 64]

        if gate is not None:
            gate_tiled = torch.zeros_like(q_tiled[:, : logical_blocks * 64])
            gate_tiled[:, non_pad] = gate[:, partition]
            v_pool = mean_pool_tiles(v_tiled[:, : logical_blocks * 64], sizes, block_size=64)
            compressed = torch.matmul(torch.softmax(scores, dim=-1), v_pool)
            compressed = compressed.permute(0, 2, 1, 3).to(output.dtype)
            output = (
                output.view(output.shape[0], logical_blocks, 64, output.shape[2], output.shape[3])
                + compressed.unsqueeze(2)
                * gate_tiled.view(gate_tiled.shape[0], logical_blocks, 64, gate_tiled.shape[2], gate_tiled.shape[3])
            ).view_as(output)
        return output[:, untile].contiguous()

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if _get_h3_layout(attn_metadata) is None:
            return super().forward_cuda(query, key, value, attn_metadata)

        original_query, original_key, original_value = query, key, value
        original_seq_len = query.shape[1]
        valid_seq_len = original_seq_len
        if attn_metadata is not None and attn_metadata.packed_padding is not None:
            valid_seq_len = attn_metadata.packed_padding.q_length
            if attn_metadata.packed_padding.kv_length != valid_seq_len:
                return self._fallback(
                    original_query, original_key, original_value, attn_metadata, "packed Q/KV lengths must match"
                )
            query = query[:, :valid_seq_len]
            key = key[:, :valid_seq_len]
            value = value[:, :valid_seq_len]

        try:
            output = self._forward_h3(query, key, value, attn_metadata)
            if valid_seq_len == original_seq_len:
                return output
            restored = torch.zeros_like(original_query)
            restored[:, :valid_seq_len] = output
            return restored
        except Exception as exc:
            # A CUDA fault poisons the process context; attempting SDPA
            # afterwards obscures the original kernel failure and cannot
            # recover the request.
            if isinstance(exc, torch.AcceleratorError):
                raise
            if not self.fallback_on_error:
                raise
            return self._fallback(
                original_query, original_key, original_value, attn_metadata, f"VSA-H3 kernel failed: {exc}"
            )
