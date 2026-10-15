# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
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
    ranges: tuple[tuple[int, int, int], ...]
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
    query_range: tuple[int, int] | None
    # Local [start, end) range shared by every row when the batch can retain
    # its original [B, S] layout.  The packed query_indices remain available
    # for metadata construction and the heterogeneous fallback.
    local_query_range: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class PagedPiecewisePlan:
    """Piecewise segment batches for one packed paged-attention batch."""

    spans: tuple[tuple[tuple[int, int], ...], ...]
    segments: tuple[PagedPiecewiseSegment, ...]
    num_query_tokens: int
    segments_cover_query_contiguously: bool
    # (batch_size, query_len) when all rows have the same local segment
    # layout.  Native FIA still receives flattened row-major tokens, but the
    # runner can restore outputs with one batch concat instead of one indexed
    # scatter per segment.
    homogeneous_batch_shape: tuple[int, int] | None = None


PagedPiecewiseRunner = Callable[
    [torch.Tensor, torch.Tensor | None, torch.Tensor | None, object, torch.Tensor | None],
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

    spans: tuple[tuple[tuple[int, int], ...], ...] = tuple(
        tuple((span_start, span_end) for span_start, span_end in row_spans) for row_spans in full_attn_spans
    )

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
        query_indices: list[int] = []
        local_ranges: list[tuple[int, int]] = []
        for row_index, (segment, query_offset) in enumerate(zip(row_segments, query_offsets, strict=True)):
            local_start = segment.q_start - query_offset
            local_end = segment.q_end - query_offset
            local_ranges.append((local_start, local_end))
            query_indices.extend(range(packed_offsets[row_index] + local_start, packed_offsets[row_index] + local_end))
        query_range = None
        if query_indices and query_indices == list(range(query_indices[0], query_indices[-1] + 1)):
            query_range = (query_indices[0], query_indices[-1] + 1)
        local_query_range = local_ranges[0] if len(set(local_ranges)) == 1 else None
        packed_segments.append(
            PagedPiecewiseSegment(
                row_segments=tuple(row_segments),
                query_indices=torch.tensor(query_indices, dtype=torch.long, device=device),
                query_range=query_range,
                local_query_range=local_query_range,
            )
        )

    covered = 0
    for segment in packed_segments:
        if segment.query_range is None or segment.query_range[0] != covered:
            break
        covered = segment.query_range[1]
    segments_cover_query_contiguously = covered == packed_offsets[-1]
    homogeneous_batch_shape: tuple[int, int] | None = None
    if (
        row_count > 1
        and len(set(query_lens)) == 1
        and all(segment.local_query_range is not None for segment in packed_segments)
    ):
        # A common local range is sufficient even when rows have different
        # global offsets or KV lengths; those values remain row-specific in
        # native metadata.
        homogeneous_batch_shape = (row_count, int(query_lens[0]))
    return PagedPiecewisePlan(
        spans=spans,
        segments=tuple(packed_segments),
        num_query_tokens=packed_offsets[-1],
        segments_cover_query_contiguously=segments_cover_query_contiguously,
        homogeneous_batch_shape=homogeneous_batch_shape,
    )


def _run_homogeneous_paged_piecewise_plan(
    query: torch.Tensor,
    key: torch.Tensor | None,
    value: torch.Tensor | None,
    plan: PagedPiecewisePlan,
    segment_metadata: Sequence[object],
    segment_runner: PagedPiecewiseRunner,
    output_buffer: torch.Tensor | None,
) -> torch.Tensor:
    """Run a homogeneous batch while retaining the old [B, S] layout.

    The native FIA contract is flattened TND, so each segment is flattened
    only for the kernel call.  Results are reshaped back to [B, L] and
    concatenated along the sequence dimension, which avoids the per-segment
    ``index_copy_`` used by heterogeneous packed rows.
    """

    batch_shape = plan.homogeneous_batch_shape
    assert batch_shape is not None
    batch_size, query_len = batch_shape
    if query.shape[0] != batch_size * query_len:
        raise ValueError(
            f"Homogeneous paged piecewise plan expects {batch_size * query_len} query tokens, got {query.shape[0]}"
        )
    query_rows = query.reshape(batch_size, query_len, *query.shape[1:])
    key_rows = None if key is None else key.reshape(batch_size, query_len, *key.shape[1:])
    value_rows = None if value is None else value.reshape(batch_size, query_len, *value.shape[1:])

    segment_outputs: list[torch.Tensor] = []
    for segment, metadata in zip(plan.segments, segment_metadata, strict=True):
        local_range = segment.local_query_range
        if local_range is None:
            raise ValueError("Homogeneous paged piecewise execution requires a shared local segment range")
        start, end = local_range
        segment_query = query_rows[:, start:end].contiguous().reshape(-1, *query.shape[1:])
        segment_key = None if key_rows is None else key_rows[:, start:end].contiguous().reshape(-1, *key.shape[1:])
        segment_value = (
            None if value_rows is None else value_rows[:, start:end].contiguous().reshape(-1, *value.shape[1:])
        )
        segment_output = segment_runner(segment_query, segment_key, segment_value, metadata, None)
        expected_tokens = batch_size * (end - start)
        if segment_output.shape[0] != expected_tokens:
            raise ValueError(
                f"Paged piecewise runner returned {segment_output.shape[0]} tokens "
                f"for a {expected_tokens}-token homogeneous segment"
            )
        segment_outputs.append(segment_output.reshape(batch_size, end - start, *segment_output.shape[1:]))

    if not segment_outputs:
        result = torch.empty_like(query)
    elif len(segment_outputs) == 1:
        result = segment_outputs[0].reshape(plan.num_query_tokens, *segment_outputs[0].shape[2:])
    else:
        result_rows = torch.cat(segment_outputs, dim=1)
        result = result_rows.reshape(plan.num_query_tokens, *result_rows.shape[2:])
    if output_buffer is not None:
        if output_buffer.shape != result.shape:
            raise ValueError(
                f"Paged piecewise output buffer shape {tuple(output_buffer.shape)} does not match "
                f"homogeneous result shape {tuple(result.shape)}"
            )
        output_buffer.copy_(result)
        return output_buffer
    return result


def run_paged_piecewise_plan(
    query: torch.Tensor,
    key: torch.Tensor | None,
    value: torch.Tensor | None,
    plan: PagedPiecewisePlan,
    segment_metadata: Sequence[object],
    segment_runner: PagedPiecewiseRunner,
    output_buffer: torch.Tensor | None = None,
    *,
    use_homogeneous_batch: bool = False,
) -> torch.Tensor:
    """Run piecewise attention with view fast paths for contiguous segments.

    ``use_homogeneous_batch`` is an opt-in for native backends that can keep
    identical rows in the legacy batch layout.  The default remains the
    indexed packed path so existing GPU/heterogeneous contracts are unchanged.
    """

    if query.shape[0] != plan.num_query_tokens:
        raise ValueError(f"Paged piecewise plan has {plan.num_query_tokens} query tokens, got {query.shape[0]}")
    if output_buffer is not None and output_buffer.shape[0] != plan.num_query_tokens:
        raise ValueError(
            f"Paged piecewise output buffer has {output_buffer.shape[0]} tokens; expected {plan.num_query_tokens}"
        )

    if use_homogeneous_batch and plan.homogeneous_batch_shape is not None:
        return _run_homogeneous_paged_piecewise_plan(
            query,
            key,
            value,
            plan,
            segment_metadata,
            segment_runner,
            output_buffer,
        )

    output = output_buffer
    contiguous_outputs = []
    for segment, metadata in zip(plan.segments, segment_metadata, strict=True):
        indices = segment.query_indices
        segment_output_buffer = None
        if segment.query_range is None:
            segment_query = query.index_select(0, indices)
            segment_key = None if key is None else key.index_select(0, indices)
            segment_value = None if value is None else value.index_select(0, indices)
        else:
            start, end = segment.query_range
            segment_query = query[start:end]
            segment_key = None if key is None else key[start:end]
            segment_value = None if value is None else value[start:end]
            if output is not None:
                segment_output_buffer = output[start:end]
        segment_output = segment_runner(
            segment_query,
            segment_key,
            segment_value,
            metadata,
            segment_output_buffer,
        )
        if segment_output.shape[0] != indices.shape[0]:
            raise ValueError(
                f"Paged piecewise runner returned {segment_output.shape[0]} tokens "
                f"for a {indices.shape[0]}-token segment"
            )
        if segment_output_buffer is not None and segment_output is not segment_output_buffer:
            if segment_output.shape != segment_output_buffer.shape:
                raise ValueError(
                    f"Paged piecewise runner returned shape {tuple(segment_output.shape)} "
                    f"for output buffer shape {tuple(segment_output_buffer.shape)}"
                )
            segment_output_buffer.copy_(segment_output)
            segment_output = segment_output_buffer
        if plan.segments_cover_query_contiguously:
            if output_buffer is None:
                contiguous_outputs.append(segment_output)
            continue
        if output is None:
            output = segment_output.new_empty((query.shape[0], *segment_output.shape[1:]))
        if segment.query_range is None:
            output.index_copy_(0, indices, segment_output)
        else:
            start, end = segment.query_range
            output[start:end].copy_(segment_output)

    if output_buffer is not None and plan.segments_cover_query_contiguously:
        return output_buffer
    if contiguous_outputs:
        if len(contiguous_outputs) == 1:
            return contiguous_outputs[0]
        return torch.cat(contiguous_outputs, dim=0)
    assert output is not None
    return output
