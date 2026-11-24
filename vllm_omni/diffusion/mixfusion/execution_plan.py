# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MixFusion sequence-packing layout primitives.

Chunked joint streams and prefix-KV documents use the same chunk geometry.
The plan maps chunks to requests and supplies the indices and sequence
boundaries for attention and output restoration. It holds layout data only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm_omni.diffusion.mixfusion.adapter import AttentionLayout


@dataclass(frozen=True)
class PackedDocumentGeometry:
    request_id: str
    branch: int
    query_length: int
    padded_length: int
    prefix_length: int
    full_spans: tuple[tuple[int, int], ...]
    # Half-open image range in the logical Q+KV sequence.
    image_span: tuple[int, int]
    # Capture the invariant prefix on the first denoising step only.
    capture_length: int | None = None


@dataclass
class PackedAttentionSegments:
    causal: bool
    query_indices: torch.Tensor
    kv_indices: torch.Tensor
    cu_query: torch.Tensor
    cu_key: torch.Tensor
    query_lengths: tuple[int, ...]
    key_lengths: tuple[int, ...]


@dataclass
class PackedExecutionPlan:
    """Chunk geometry and attention indices shared by all layers of a step."""

    chunk_size: int
    request_chunk_ranges: list[tuple[int, int]]
    image_row_counts: tuple[int, ...]
    chunk_to_request: torch.Tensor

    # Joint text/image streams share Q/K boundaries.
    layout: AttentionLayout | None = None
    request_stream_lens: dict[str, tuple[int, ...]] = field(default_factory=dict)
    stream_padded_lens: dict[str, int] = field(default_factory=dict)
    cu_seqlens_tensor: torch.Tensor | None = None
    max_seq_len: int = 0
    output_indices: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = field(default_factory=dict)

    # Prefix-KV documents index the same chunk rows, followed by other queries.
    documents: tuple[PackedDocumentGeometry, ...] = ()
    document_indices: tuple[torch.Tensor, ...] = ()
    segments: tuple[PackedAttentionSegments, ...] = ()

    @property
    def num_chunks(self) -> int:
        return self.request_chunk_ranges[-1][1] if self.request_chunk_ranges else 0

    def image_chunk_rows(self, tokens: torch.Tensor) -> torch.Tensor:
        """View contiguous image tokens as rows of the common chunk size."""
        return tokens.reshape(-1, self.chunk_size, *tokens.shape[1:])


def build_chunk_plan(
    image_lengths: Sequence[int], image_row_counts: Sequence[int], device: torch.device
) -> PackedExecutionPlan:
    """Allocate equal-length chunk rows from the actual image token lengths."""
    if not image_lengths or min(image_lengths) <= 0 or sum(image_row_counts) != len(image_lengths):
        raise ValueError("MixFusion chunk rows must cover a nonempty image batch")
    chunk_size = math.gcd(*image_lengths)
    ranges = []
    owners: list[int] = []
    for row, length in enumerate(image_lengths):
        start = len(owners)
        owners.extend([row] * (length // chunk_size))
        ranges.append((start, len(owners)))
    return PackedExecutionPlan(chunk_size, ranges, tuple(image_row_counts), torch.tensor(owners, device=device))


def build_execution_plan(
    *,
    layout: AttentionLayout,
    chunk_size: int,
    request_chunk_ranges: list[tuple[int, int]],
    image_row_counts: Sequence[int],
    request_stream_lens: dict[str, Sequence[int]],
    stream_padded_lens: dict[str, int],
    chunk_to_request: torch.Tensor,
    device: torch.device,
) -> PackedExecutionPlan:
    """Lay each request out as its streams in ``layout`` order.

    Inputs come straight from the adapter that just split the latents, so
    only the cross-field invariants are asserted.
    """
    lens_by_stream = {name: tuple(int(length) for length in request_stream_lens[name]) for name in layout.names}
    assert sum(image_row_counts) == len(request_chunk_ranges), "image_row_counts must cover every chunked row"

    cu_seqlens = [0]
    max_seq_len = 0
    gather_rows: dict[str, list[int]] = {name: [] for name in layout.names}
    for req_idx in range(len(request_chunk_ranges)):
        offset = cu_seqlens[-1]
        for spec in layout.streams:
            length = lens_by_stream[spec.name][req_idx]
            gather_rows[spec.name].extend(range(offset, offset + length))
            offset += length
        cu_seqlens.append(offset)
        max_seq_len = max(max_seq_len, offset - cu_seqlens[-2])

    output_indices: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
    for spec in layout.streams:
        gather = torch.tensor(gather_rows[spec.name], dtype=torch.long, device=device)
        scatter = None
        lens = lens_by_stream[spec.name]
        if spec.kind == "request" and any(length < stream_padded_lens[spec.name] for length in lens):
            padded = stream_padded_lens[spec.name]
            scatter_rows = [req_idx * padded + pos for req_idx, length in enumerate(lens) for pos in range(length)]
            scatter = torch.tensor(scatter_rows, dtype=torch.long, device=device)
        output_indices[spec.name] = (gather, scatter)

    return PackedExecutionPlan(
        layout=layout,
        chunk_size=chunk_size,
        request_chunk_ranges=request_chunk_ranges,
        image_row_counts=tuple(image_row_counts),
        chunk_to_request=chunk_to_request,
        request_stream_lens=lens_by_stream,
        stream_padded_lens=dict(stream_padded_lens),
        cu_seqlens_tensor=torch.tensor(cu_seqlens, dtype=torch.int32, device=device),
        max_seq_len=max_seq_len,
        output_indices=output_indices,
    )


def build_document_plan(documents: tuple[PackedDocumentGeometry, ...], device: torch.device) -> PackedExecutionPlan:
    """Address each request's queries in image-chunk then non-image storage."""
    from vllm_omni.diffusion.attention.backends.utils.piecewise_attn import build_segments

    image_lengths = [doc.image_span[1] - doc.image_span[0] for doc in documents]
    # Each request/CFG branch supplies one image row.
    plan = build_chunk_plan(image_lengths, [1] * len(documents), device)
    other_offset = plan.num_chunks * plan.chunk_size
    kv_offset = sum(doc.query_length for doc in documents)
    document_indices = []
    grouped: dict[bool, list[tuple[list[int], list[int]]]] = {}
    for doc, (chunk_start, chunk_end) in zip(documents, plan.request_chunk_ranges, strict=True):
        qlen, plen = doc.query_length, doc.prefix_length
        if not 0 < qlen <= doc.padded_length or plen < 0:
            raise ValueError("Invalid packed sequence query/prefix length")
        start, stop = doc.image_span
        if not plen <= start < stop <= plen + qlen:
            raise ValueError("Image chunks must cover current queries")
        before, after = start - plen, plen + qlen - stop
        indices = [
            *range(other_offset, other_offset + before),
            *range(chunk_start * plan.chunk_size, chunk_end * plan.chunk_size),
            *range(other_offset + before, other_offset + before + after),
        ]
        document_indices.append(torch.tensor(indices, device=device, dtype=torch.long))
        other_offset += before + after
        end = 0
        for start, stop in doc.full_spans:
            if not end <= start < stop <= plen + qlen:
                raise ValueError("Packed full-attention spans must be ordered and within the document")
            end = stop
        if doc.capture_length is not None and not plen <= doc.capture_length <= plen + qlen:
            raise ValueError("Captured prefix must include imported KV and fit the first step")
        keys_in_order = [*range(kv_offset, kv_offset + plen), *indices]
        for segment in build_segments(doc.full_spans, plen, qlen):
            query = indices[segment.q_start - plen : segment.q_end - plen]
            keys = keys_in_order[: segment.kv_end]
            grouped.setdefault(segment.mode == "causal", []).append((query, keys))
        kv_offset += plen
    batches = []
    for causal, segments in grouped.items():
        qi, ki, cq, ck = [], [], [0], [0]
        for query, keys in segments:
            qi.extend(query)
            ki.extend(keys)
            cq.append(len(qi))
            ck.append(len(ki))
        batches.append(
            PackedAttentionSegments(
                causal,
                torch.tensor(qi, device=device, dtype=torch.long),
                torch.tensor(ki, device=device, dtype=torch.long),
                torch.tensor(cq, device=device, dtype=torch.int32),
                torch.tensor(ck, device=device, dtype=torch.int32),
                tuple(len(q) for q, _ in segments),
                tuple(len(k) for _, k in segments),
            )
        )
    plan.documents = documents
    plan.document_indices = tuple(document_indices)
    plan.segments = tuple(batches)
    return plan


def pack_document_queries(rows: list[torch.Tensor], plan: PackedExecutionPlan) -> torch.Tensor:
    """Pack contiguous image rows followed by the remaining queries."""
    images, others = [], []
    for row, doc in zip(rows, plan.documents, strict=True):
        start, stop = (pos - doc.prefix_length for pos in doc.image_span)
        images.append(row[start:stop])
        if start:
            others.append(row[:start])
        if stop < doc.query_length:
            others.append(row[stop : doc.query_length])
    return torch.cat(images + others)


def expand_per_chunk(temb: torch.Tensor, chunk_to_request: torch.Tensor) -> torch.Tensor:
    """Expand per-request conditioning rows to per-chunk rows.

    Chunk-level computations (AdaLN modulation) need request-level
    conditioning duplicated per chunk.
    """
    return temb.index_select(0, chunk_to_request)


def reassemble_chunk_rows(
    chunk_rows: torch.Tensor,
    request_chunk_ranges: Sequence[tuple[int, int]],
    row_counts: Sequence[int],
) -> list[torch.Tensor]:
    """Rebuild one output tensor per request from its contiguous chunk rows.

    Request ``i`` owns ``row_counts[i]`` image rows whose chunks are contiguous
    in ``request_chunk_ranges``; the result is one ``[num_images, S, C]``
    tensor per request.
    """
    outputs: list[torch.Tensor] = []
    row_idx = 0
    last_dim = chunk_rows.shape[-1]
    for num_images in row_counts:
        rows = []
        for _ in range(num_images):
            chunk_start, chunk_end = request_chunk_ranges[row_idx]
            row_idx += 1
            rows.append(chunk_rows[chunk_start:chunk_end].reshape(1, -1, last_dim))
        outputs.append(torch.cat(rows, dim=0) if num_images > 1 else rows[0])
    return outputs
