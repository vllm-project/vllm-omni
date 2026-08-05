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
