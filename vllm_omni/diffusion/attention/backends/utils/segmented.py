# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Segmented attention for mixed causal / bidirectional masks
(e.g. HunyuanImage3: causal outside image spans, bidirectional inside).

Targets FA-style bottom-right aligned causal masks: each segment is
dispatched as a separate attention call whose causal flag follows
FlashAttention's bottom-right convention (``K[:e]`` is attended by
``Q[s:e]``, with causal alignment anchored at the bottom-right corner).

Per segment:
  - text segment ``[s, e)``: ``attn(Q[:, s:e], K[:, :e], V[:, :e], causal=True)``
  - image span   ``[a, e)``: ``attn(Q[:, a:e], K[:, :e], V[:, :e], causal=False)``
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import torch


class Segment(NamedTuple):
    start: int
    end: int
    type: Literal["text", "image"]


def build_segments(spans, q_positions, total_len):
    """
    spans: list of (start, end) half-open image spans
    q_positions: optional list of absolute positions for query
    total_len: full sequence length

    return:
        List[Segment]
    """

    # ---- query range ----
    if q_positions is None:
        q_start, q_end = 0, total_len
    else:
        q_start = q_positions[0]
        q_end = q_positions[-1] + 1

    segs: list[Segment] = []
    cur = q_start

    for a, e in spans:
        if cur >= e:
            continue
        elif cur < a:
            segs.append(Segment(cur, min(a, q_end), "text"))
            cur = a

        if cur < q_end:
            assert a == cur and e <= q_end, f"span ({a}, {e}) must be within query range ({q_start}, {q_end})"
            segs.append(Segment(a, e, "image"))
            cur = e

        if cur >= q_end:
            break
    if cur < q_end:
        segs.append(Segment(cur, q_end, "text"))

    return segs


def _is_homogeneous_batch(
    image_spans: list[list[tuple[int, int]]],
    q_global_positions: torch.Tensor,
) -> bool:
    """
    True if all samples share identical image_spans and identical q_global_positions rows.
    HunyuanImage3's CFG batch (pos + neg prompt) satisfies this.
    """
    B = len(image_spans)
    if B <= 1:
        return True
    ref_spans = image_spans[0]
    for s in image_spans[1:]:
        if s != ref_spans:
            return False
    ref_pos = q_global_positions[0]
    for i in range(1, q_global_positions.shape[0]):
        if not torch.equal(q_global_positions[i], ref_pos):
            return False
    return True


def segmented_attn(
    query: torch.Tensor,  # (B, S*, H, D)
    key: torch.Tensor,
    value: torch.Tensor,
    image_spans: list[list[tuple[int, int]]],  # outer length B
    q_global_positions: torch.Tensor,  # (B, Sq)
    softmax_scale: float,
    attn_func,
) -> torch.Tensor:
    B, Sq, H, D = query.shape
    if not _is_homogeneous_batch(image_spans, q_global_positions):
        raise NotImplementedError(
            "segmented_attn requires a homogeneous batch (identical image_spans and q_global_positions across samples)."
        )
    sample_spans = image_spans[0]
    sample_q_positions = q_global_positions[0].tolist()
    query_offset = int(q_global_positions[0, 0].item())
    out = query.new_zeros(B, Sq, H, D)

    for s, e, type in build_segments(sample_spans, sample_q_positions, key.shape[1]):
        q_seg_start = s - query_offset
        q_seg_end = e - query_offset
        out_seg = attn_func(
            query[:, q_seg_start:q_seg_end],
            key[:, :e],
            value[:, :e],
            causal=(type == "text"),
            softmax_scale=softmax_scale,
        )
        out[:, q_seg_start:q_seg_end] = out_seg
    return out
