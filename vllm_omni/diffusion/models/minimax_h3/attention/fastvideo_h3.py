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
    _get_gate_compress,
)
from vllm_omni.diffusion.attention.ops.block_sparse import (
    block_sparse_attn_bshd,
    build_prefix_dense_block_map,
    mean_pool_tiles,
    resolve_block_sparse_provider,
)
from vllm_omni.diffusion.attention.ops.video_tiles import (
    construct_variable_block_sizes,
    get_non_pad_index,
    get_tile_partition_indices,
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

    video_indices = get_tile_partition_indices(video_shape, block_size, device) + prefix_len
    video_sizes = construct_variable_block_sizes(video_shape, block_size, device)
    partition = torch.cat([torch.arange(prefix_len, device=device, dtype=torch.long), video_indices])
    sizes = torch.cat([torch.tensor(prefix_sizes, device=device, dtype=torch.int32), video_sizes.to(torch.int32)])
    non_pad = get_non_pad_index(sizes, block_elements)
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


@functools.lru_cache(maxsize=32)
def _get_h3_multiregion_tile_metadata(
    prefix_segments: tuple[int, ...],
    regions: tuple[tuple[int, tuple[int, int, int]], ...],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, tuple[tuple[int, int], ...]]:
    """P2 layout: dense modality chunks first, then each independently tiled video.

    Reference images stay dense. Video spans come from packed Ref2VA metadata,
    so audio/video references may be interleaved in their original input order.
    """
    tile_shape, block_size = (4, 4, 8), 128
    if not regions or regions[-1][0] != sum(prefix_segments):
        raise ValueError("P2 video regions must end with the target at the prefix boundary")
    intervals = [(start, start + math.prod(shape)) for start, shape in regions]
    if any(start < 0 or any(d <= 0 for d in shape) for start, shape in regions):
        raise ValueError("P2 video regions require nonnegative offsets and positive grids")
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError("P2 video regions must be ordered and non-overlapping")
    dense_indices, dense_sizes = [], []
    cursor = 0
    for length in prefix_segments:
        end = cursor + length
        remaining = [(cursor, end)]
        for video_start, video_end in intervals[:-1]:
            remaining = [
                piece
                for start, stop in remaining
                for piece in ((start, min(stop, video_start)), (max(start, video_end), stop))
                if piece[0] < piece[1]
            ]
        for start, stop in remaining:
            dense_indices.append(torch.arange(start, stop, device=device, dtype=torch.long))
            dense_sizes.extend(min(block_size, stop - pos) for pos in range(start, stop, block_size))
        cursor = end
    partitions = dense_indices
    size_parts = [torch.tensor(dense_sizes, device=device, dtype=torch.int32)]
    block_cursor = len(dense_sizes)
    spans = []
    for start, shape in regions:
        partitions.append(get_tile_partition_indices(shape, tile_shape, device) + start)
        sizes = construct_variable_block_sizes(shape, tile_shape, device).to(torch.int32)
        size_parts.append(sizes)
        spans.append((block_cursor, block_cursor + sizes.numel()))
        block_cursor += sizes.numel()
    partition, sizes = torch.cat(partitions), torch.cat(size_parts)
    total = intervals[-1][1]
    if partition.numel() != total or not torch.equal(partition.sort().values, torch.arange(total, device=device)):
        raise ValueError("P2 geometry must cover every packed row exactly once")
    non_pad = get_non_pad_index(sizes, block_size)
    return partition, sizes, non_pad, non_pad[torch.argsort(partition)], len(dense_sizes), tuple(spans)


def _build_h3_region_block_map(
    scores: torch.Tensor,
    prefix_blocks: int,
    spans: tuple[tuple[int, int], ...],
    keep_rates: tuple[float, ...],
) -> torch.Tensor:
    """Dense prefix queries/keys, with an independent top-k budget per video."""
    mask = torch.zeros_like(scores, dtype=torch.bool)
    for (start, end), rate in zip(spans, keep_rates, strict=True):
        count = max(1, min(math.ceil(rate * (end - start)), end - start))
        indices = scores[..., start:end].topk(count, dim=-1).indices + start
        mask.scatter_(-1, indices, True)
    mask[..., :prefix_blocks] = True
    mask[..., :prefix_blocks, :] = True
    return mask


class MiniMaxH3VSAImpl(FastVideoVSAImpl):
    """Apply release-specific H3 routing after parallel attention dispatch."""

    _supports_tile64_provider = True

    def _forward_h3(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
        *,
        provider: str,
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

        block_size = attn_metadata.extra.get("vsa_h3_tile_size", 64)
        sparsity = attn_metadata.extra.get("vsa_h3_sparsity")
        if block_size == 128:
            if attn_metadata.extra.get("vsa_h3_ref_policy") != "p2_multi_region":
                raise ValueError("FastH3 Omni tile 128 requires p2_multi_region routing")
            regions = tuple((span.start, span.latent_grid) for span in attn_metadata.video_layout.video_spans)
            partition, sizes, non_pad, untile, prefix_blocks, spans = _get_h3_multiregion_tile_metadata(
                prefix_segments, regions, query.device
            )
            video_blocks = sum(end - start for start, end in spans)
        elif block_size == 64:
            partition, sizes, non_pad, untile, prefix_blocks, video_blocks = _get_h3_tile_metadata(
                prefix_segments, video_shape, query.device
            )
            spans = ((prefix_blocks, prefix_blocks + video_blocks),)
        else:
            raise ValueError(f"unsupported VSA-H3 tile size: {block_size}")
        logical_blocks = int(sizes.numel())
        target_shape = (query.shape[0], logical_blocks * block_size, query.shape[2], query.shape[3])
        q_tiled = torch.zeros(target_shape, device=query.device, dtype=query.dtype)
        k_tiled = torch.zeros_like(q_tiled)
        v_tiled = torch.zeros_like(q_tiled)
        q_tiled[:, non_pad] = query.index_select(1, partition)
        k_tiled[:, non_pad] = key.index_select(1, partition)
        v_tiled[:, non_pad] = value.index_select(1, partition)

        q_pool = mean_pool_tiles(q_tiled[:, : logical_blocks * block_size], sizes, block_size=block_size)
        k_pool = mean_pool_tiles(k_tiled[:, : logical_blocks * block_size], sizes, block_size=block_size)
        scores = torch.matmul(q_pool, k_pool.transpose(-2, -1)) * self.softmax_scale
        if sparsity is None and block_size == 64:
            block_map = build_prefix_dense_block_map(scores, prefix_blocks, video_blocks, self.topk)
            topk_by_region = (min(self.topk, video_blocks),)
        else:
            target_rate = 1.0 - float(sparsity)
            ref_rate = attn_metadata.extra.get("vsa_h3_ref_keep_rate")
            # Match upstream's 1 - (1 - ref_keep_rate) rounding in ceil(topk).
            ref_rate = target_rate if ref_rate is None else 1.0 - (1.0 - float(ref_rate))
            rates = (ref_rate,) * (len(spans) - 1) + (target_rate,)
            block_map = _build_h3_region_block_map(scores, prefix_blocks, spans, rates)
            topk_by_region = tuple(
                max(1, min(math.ceil(rate * (end - start)), end - start))
                for (start, end), rate in zip(spans, rates, strict=True)
            )
        logger.info_once(
            "FASTVIDEO_VSA H3 routing: seq_len=%d, prefix_segments=%s, video_shape=%s, "
            "prefix_blocks=%d, video_blocks=%d, video_topk_by_region=%s, logical_blocks=%d",
            query.shape[1],
            prefix_segments,
            video_shape,
            prefix_blocks,
            video_blocks,
            topk_by_region,
            logical_blocks,
        )
        output = block_sparse_attn_bshd(
            q_tiled,
            k_tiled,
            v_tiled,
            block_map,
            sizes,
            self.softmax_scale,
            provider=provider,
            precision=self.precision,
            block_size=block_size,
        )
        logger.info_once(
            "H3 VSA executing provider=%s precision=%s block_size=%d", provider, self.precision, block_size
        )

        if gate is not None:
            gate_tiled = torch.zeros_like(q_tiled[:, : logical_blocks * block_size])
            gate_tiled[:, non_pad] = gate.index_select(1, partition)
            v_pool = mean_pool_tiles(v_tiled[:, : logical_blocks * block_size], sizes, block_size=block_size)
            compressed = torch.matmul(torch.softmax(scores, dim=-1), v_pool)
            compressed = compressed.permute(0, 2, 1, 3).to(output.dtype)
            output = (
                output.view(output.shape[0], logical_blocks, block_size, output.shape[2], output.shape[3])
                + compressed.unsqueeze(2)
                * gate_tiled.view(
                    gate_tiled.shape[0], logical_blocks, block_size, gate_tiled.shape[2], gate_tiled.shape[3]
                )
            ).view_as(output)
        return output.index_select(1, untile).contiguous()

    def forward_sparse(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        requires_tile128 = attn_metadata is not None and attn_metadata.extra.get("vsa_h3_tile_size", 64) == 128
        requires_sparse = requires_tile128 or (
            attn_metadata is not None and attn_metadata.extra.get("vsa_h3_sparsity") is not None
        )
        if _get_h3_layout(attn_metadata) is None:
            if requires_sparse:
                raise ValueError("FastH3 requires complete packed video layout metadata")
            return super().forward_cuda(query, key, value, attn_metadata)

        provider = resolve_block_sparse_provider(
            self.provider,
            self.precision,
            query.device,
            query.dtype,
            self.head_size,
            block_size=128 if requires_tile128 else 64,
        )
        original_query, original_key, original_value = query, key, value
        original_seq_len = query.shape[1]
        valid_seq_len = original_seq_len
        if attn_metadata is not None and attn_metadata.packed_padding is not None:
            valid_seq_len = attn_metadata.packed_padding.q_length
            if attn_metadata.packed_padding.kv_length != valid_seq_len:
                if requires_sparse:
                    raise ValueError("FastH3 packed Q/KV lengths must match")
                return self._fallback(
                    original_query, original_key, original_value, attn_metadata, "packed Q/KV lengths must match"
                )
            query = query[:, :valid_seq_len]
            key = key[:, :valid_seq_len]
            value = value[:, :valid_seq_len]

        try:
            output = self._forward_h3(query, key, value, attn_metadata, provider=provider)
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
            if requires_sparse or not self.fallback_on_error:
                raise
            return self._fallback(
                original_query, original_key, original_value, attn_metadata, f"VSA-H3 kernel failed: {exc}"
            )

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        return self.forward_sparse(query, key, value, attn_metadata)

    def forward_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None or attn_metadata.extra.get("vsa_h3_tile_size", 64) != 128:
            return super().forward_npu(query, key, value, attn_metadata)
        valid = attn_metadata.extra.get("valid_kv_length", query.shape[1])
        output = self.forward_sparse(query[:, :valid], key[:, :valid], value[:, :valid], attn_metadata)
        if valid == query.shape[1]:
            return output
        restored = torch.zeros_like(query)
        restored[:, :valid] = output
        return restored
