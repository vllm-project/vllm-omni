# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Piecewise attention for mixed causal / full (bidirectional) masks.

Dispatches each segment as a separate attention call whose causal flag
follows FlashAttention's bottom-right convention (``K[:e]`` is attended by
``Q[s:e]``, with causal alignment anchored at the bottom-right corner).

Per segment:
  - causal segment ``[s, e)``: ``attn(Q[:, s:e], K[:, :e], V[:, :e], causal=True)``
  - full-attn span ``[a, e)``: ``attn(Q[:, a:e], K[:, :e], V[:, :e], causal=False)``
"""

from __future__ import annotations

from typing import Literal, NamedTuple


class Segment(NamedTuple):
    start: int
    end: int
    mode: Literal["causal", "full"]


def build_segments(full_attn_spans, query_offset, query_len):
    """
    full_attn_spans: list of (start, end) half-open spans in global coordinates
    query_offset: starting position of query in the global sequence
    query_len: length of the query

    return:
        List[Segment] in global coordinates, clipped to [query_offset, query_offset + query_len)
    """
    q_start = query_offset
    q_end = query_offset + query_len

    segs: list[Segment] = []
    cur = q_start

    for a, e in full_attn_spans:
        # clip span to query range
        a_clipped = max(a, q_start)
        e_clipped = min(e, q_end)
        if a_clipped >= e_clipped:
            continue

        if cur < a_clipped:
            segs.append(Segment(cur, a_clipped, "causal"))
        segs.append(Segment(a_clipped, e_clipped, "full"))
        cur = e_clipped

    if cur < q_end:
        segs.append(Segment(cur, q_end, "causal"))

    return segs


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
):
    B, Sq, H, D = query.shape
    _check_homogeneous(full_attn_spans)

    query_offset = key.shape[1] - Sq
    spans = full_attn_spans[0]
    out = query.new_zeros(B, Sq, H, D)

    for s, e, mode in build_segments(spans, query_offset, Sq):
        q_s = s - query_offset
        q_e = e - query_offset
        out_seg = attn_func(
            query[:, q_s:q_e],
            key[:, :e],
            value[:, :e],
            causal=(mode == "causal"),
            softmax_scale=softmax_scale,
        )
        out[:, q_s:q_e] = out_seg
    return out
