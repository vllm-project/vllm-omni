# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Piecewise attention for mixed causal / full (bidirectional) masks.

Dispatches each segment as a separate attention call whose causal flag
follows FlashAttention's bottom-right convention (``K[:e]`` is attended by
``Q[s:e]``, with causal alignment anchored at the bottom-right corner).

Per segment:
  - causal segment ``[s, e)``: ``attn(Q[:, s:e], K[:, :e], V[:, :e], causal=True)``
  - full-attn span ``[a, b)`` intersecting the query range at ``[s, e)``:
    ``attn(Q[:, s:e], K[:, :b], V[:, :b], causal=False)``
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple

import torch

if TYPE_CHECKING:
    from vllm_omni.diffusion.attention.backends.abstract import QueryRange


class Segment(NamedTuple):
    q_start: int
    q_end: int
    kv_end: int
    mode: Literal["causal", "full"]


def build_segments(full_attn_spans, query_offset, query_len):
    """
    full_attn_spans: list of (start, end) half-open spans in global coordinates
    query_offset: starting position of query in the global sequence
    query_len: length of the query

    return:
        List[Segment] in global coordinates, clipped to
        [query_offset, query_offset + query_len). Full-attention segments retain
        the original span end as kv_end so a local query shard can attend past
        its own boundary.
    """
    q_start = query_offset
    q_end = query_offset + query_len

    segments: list[Segment] = []
    cur = q_start

    for span_start, span_end in full_attn_spans:
        # clip span to query range
        overlap_start = max(span_start, q_start)
        overlap_end = min(span_end, q_end)
        if overlap_start >= overlap_end:
            continue

        if cur < overlap_start:
            segments.append(Segment(cur, overlap_start, overlap_start, "causal"))
        segments.append(Segment(overlap_start, overlap_end, span_end, "full"))
        cur = overlap_end

    if cur < q_end:
        segments.append(Segment(cur, q_end, q_end, "causal"))

    return segments


def _check_homogeneous(
    full_attn_spans: list[list[tuple[int, int]]],
) -> None:
    """Assert all samples share identical spans."""
    if len(full_attn_spans) > 1:
        ref = full_attn_spans[0]
        for i, s in enumerate(full_attn_spans[1:], 1):
            if s != ref:
                raise ValueError(
                    f"piecewise_attn requires homogeneous batch: sample 0 spans {ref} != sample {i} spans {s}"
                )


def piecewise_attn(
    query,  # (B, Sq, H, D)
    key,
    value,
    full_attn_spans: list[list[tuple[int, int]]],
    softmax_scale: float,
    attn_func,
    query_ranges: tuple[QueryRange, ...] | None = None,
):
    _check_homogeneous(full_attn_spans)
    spans = full_attn_spans[0]
    if query_ranges is None:
        query_len = query.shape[1]
        ranges = ((0, query_len, key.shape[1] - query_len),)
    else:
        ranges = tuple((r.local_start, r.local_end, r.global_start) for r in query_ranges)

    outputs = []
    covered = 0
    for local_start, local_end, global_start in ranges:
        query_len = local_end - local_start
        if local_start != covered or query_len < 0:
            raise ValueError("query_ranges must cover local query contiguously")
        for segment in build_segments(spans, global_start, query_len):
            q_start = local_start + segment.q_start - global_start
            q_end = local_start + segment.q_end - global_start
            outputs.append(
                attn_func(
                    query[:, q_start:q_end],
                    key[:, : segment.kv_end],
                    value[:, : segment.kv_end],
                    causal=(segment.mode == "causal"),
                    softmax_scale=softmax_scale,
                )
            )
        covered = local_end

    if covered != query.shape[1]:
        raise ValueError("query_ranges must cover the full local query")
    if not outputs:
        return torch.empty_like(query)
    if len(outputs) == 1:
        return outputs[0]
    return torch.cat(outputs, dim=1)


@dataclass(frozen=True, slots=True)
class PagedPiecewiseSegment:
    """One aligned segment across rows in a packed paged batch."""

    row_segments: tuple[Segment, ...]
    query_indices: torch.Tensor


@dataclass(frozen=True, slots=True)
class PagedPiecewisePlan:
    """Piecewise segment batches for one packed paged-attention batch."""

    spans: tuple[tuple[tuple[int, int], ...], ...]
    segments: tuple[PagedPiecewiseSegment, ...]
    num_query_tokens: int


PagedPiecewiseRunner = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, object],
    torch.Tensor,
]


def build_paged_piecewise_plan(
    full_attn_spans: Sequence[Sequence[tuple[int, int]]],
    query_offsets: Sequence[int],
    query_lens: Sequence[int],
    seq_lens: Sequence[int],
    *,
    device: torch.device | str | None = None,
) -> PagedPiecewisePlan:
    """Build packed indices for corresponding piecewise segments in each row."""

    row_count = len(full_attn_spans)
    if row_count == 0 or not (row_count == len(query_offsets) == len(query_lens) == len(seq_lens)):
        raise ValueError("Paged piecewise inputs must have one entry per row")

    spans = tuple(tuple(tuple(span) for span in row_spans) for row_spans in full_attn_spans)

    packed_offsets = [0]
    for row_spans, query_offset, query_len, seq_len in zip(spans, query_offsets, query_lens, seq_lens, strict=True):
        previous_end = 0
        for start, end in row_spans:
            if start < previous_end or not 0 <= start < end <= seq_len:
                raise ValueError("Paged piecewise spans must be sorted, non-overlapping, and within the sequence")
            previous_end = end
        if query_offset < 0 or query_len <= 0 or query_offset + query_len > seq_len:
            raise ValueError("Paged piecewise query range must be within the sequence")
        packed_offsets.append(packed_offsets[-1] + query_len)

    segments_by_row = tuple(
        tuple(build_segments(row_spans, query_offset, query_len))
        for row_spans, query_offset, query_len in zip(spans, query_offsets, query_lens, strict=True)
    )
    if len({len(row_segments) for row_segments in segments_by_row}) != 1:
        raise ValueError("Paged piecewise rows must produce aligned segments")

    packed_segments = []
    for row_segments in zip(*segments_by_row, strict=True):
        if len({segment.mode for segment in row_segments}) != 1:
            raise ValueError("Paged piecewise rows must use the same attention mode per segment")
        query_indices = []
        for row_index, (segment, query_offset) in enumerate(zip(row_segments, query_offsets, strict=True)):
            local_start = segment.q_start - query_offset
            local_end = segment.q_end - query_offset
            query_indices.extend(range(packed_offsets[row_index] + local_start, packed_offsets[row_index] + local_end))
        packed_segments.append(
            PagedPiecewiseSegment(
                row_segments=tuple(row_segments),
                query_indices=torch.tensor(query_indices, dtype=torch.long, device=device),
            )
        )

    return PagedPiecewisePlan(spans, tuple(packed_segments), packed_offsets[-1])


def run_paged_piecewise_plan(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: PagedPiecewisePlan,
    segment_metadata: Sequence[object],
    segment_runner: PagedPiecewiseRunner,
) -> torch.Tensor:
    """Run and scatter piecewise attention over packed Q/K/V write tensors."""

    if query.shape[0] != plan.num_query_tokens:
        raise ValueError(f"Paged piecewise plan has {plan.num_query_tokens} query tokens, got {query.shape[0]}")

    output = None
    for segment, metadata in zip(plan.segments, segment_metadata, strict=True):
        indices = segment.query_indices
        segment_output = segment_runner(
            query.index_select(0, indices),
            key.index_select(0, indices),
            value.index_select(0, indices),
            metadata,
        )
        if output is None:
            output = segment_output.new_empty((query.shape[0], *segment_output.shape[1:]))
        output.index_copy_(0, indices, segment_output)

    assert output is not None
    return output
