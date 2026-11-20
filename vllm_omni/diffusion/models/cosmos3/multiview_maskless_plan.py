# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Host-side planner for the maskless Cosmos3 multiview backend.

The maskless backend evaluates the multiview visibility rules as a few
*unmasked* variable-length attention passes whose partial softmax results are
merged by log-sum-exp.  Every (query, key) pair the predicate admits is covered
by exactly one pass, so the merge equals one softmax over the admitted key set:
the same attention the Triton backend computes with a mask.

The planner never re-derives geometry.  It evaluates the production predicate
on semantic runs, exactly as the sparse block-map builder does, and only
partitions that run-level truth table into rectangles:

1. ``same_view``: every GEN token of one ``(sample, view)`` group, control and
   target alike, with LiDAR as its own view.  Covers every query row.
2. ``cross_view_L<d>``: the deduplicated, time-windowed cross-view edges of the
   non-control tokens.  Views are split recursively into halves and attend
   L->R and R->L, so a directed pair of distinct views occurs exactly at their
   lowest common ancestor; all rectangles of one depth share one varlen call.
3. ``caption``: GEN tokens reading their camera's caption (LiDAR reads all).

A coverage check against the truth table is part of every build.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate

import torch

from .multiview_flex_attention import (
    MultiviewLayout,
    PaddedAttentionGeometry,
    _make_pair_allowed,
    _semantic_groups,
    build_multiview_flex_metadata,
)

#: Tensors per pass in :meth:`MultiviewMasklessPlan.flatten`: ``q_index``,
#: ``k_index``, ``cu_seqlens_q``, ``cu_seqlens_k`` and ``meta``.
TENSORS_PER_PASS = 5
#: ``meta`` is a CPU int64 vector so the attention op reads host integers
#: without a device synchronization and Dynamo never guards on prompt lengths.
META_MAX_SEQLEN_Q, META_MAX_SEQLEN_K, META_KEYS_FROM_UND, META_IDENTITY_Q, META_IDENTITY_K, META_GEN_TOKENS = range(6)
META_SIZE = 6
_INT32_LIMIT = 2**31

Segment = tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True, eq=False)
class MasklessPass:
    """One unmasked varlen attention call over gathered query/key rows.

    ``q_index`` gathers queries from the packed GEN stream; ``k_index`` gathers
    keys from the GEN stream, or from the UND stream when ``keys_from_und``.
    ``segments`` keeps the rectangles in run-table numbering for diagnostics.
    """

    name: str
    keys_from_und: bool
    q_index: torch.Tensor
    k_index: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    meta: torch.Tensor
    segments: tuple[Segment, ...]

    def tensors(self) -> list[torch.Tensor]:
        return [self.q_index, self.k_index, self.cu_seqlens_q, self.cu_seqlens_k, self.meta]


@dataclass(frozen=True, eq=False)
class MultiviewMasklessPlan:
    """Passes in merge order: ``same_view`` first, then cross-view levels, then ``caption``."""

    num_gen_tokens: int
    num_und_tokens: int
    num_levels: int
    passes: tuple[MasklessPass, ...]
    #: Token offsets into the ``[UND | GEN]`` key stream and lengths of each semantic run.
    run_starts: torch.Tensor
    run_lengths: torch.Tensor

    def flatten(self) -> list[torch.Tensor]:
        return [tensor for attention_pass in self.passes for tensor in attention_pass.tensors()]


def _sum_lengths(run_lengths: torch.Tensor, runs: Sequence[int]) -> int:
    return int(run_lengths[list(runs)].sum()) if runs else 0


def _expand_runs(run_starts: torch.Tensor, run_lengths: torch.Tensor, runs: Sequence[int], offset: int) -> torch.Tensor:
    """Token indices of ``runs`` in order, shifted by ``offset`` into the target stream."""
    if not runs:
        return torch.empty(0, dtype=torch.int64)
    selected = torch.tensor(list(runs), dtype=torch.int64)
    starts = run_starts[selected] - offset
    lengths = run_lengths[selected]
    total = int(lengths.sum())
    base = torch.repeat_interleave(starts - (torch.cumsum(lengths, 0) - lengths), lengths)
    return base + torch.arange(total, dtype=torch.int64)


def _cumulative_offsets(lengths: Sequence[int]) -> torch.Tensor:
    if sum(lengths) >= _INT32_LIMIT:
        raise ValueError("Cosmos3 maskless attention exceeds int32 cumulative sequence offsets.")
    return torch.tensor([0, *accumulate(lengths)], dtype=torch.int32)


def _materialize_pass(
    name: str,
    segments: Sequence[Segment],
    *,
    keys_from_und: bool,
    run_starts: torch.Tensor,
    run_lengths: torch.Tensor,
    num_und_tokens: int,
    num_gen_tokens: int,
    device: torch.device,
) -> MasklessPass:
    query_lengths = [_sum_lengths(run_lengths, q_runs) for q_runs, _ in segments]
    key_lengths = [_sum_lengths(run_lengths, k_runs) for _, k_runs in segments]
    if any(length == 0 for length in (*query_lengths, *key_lengths)):
        raise ValueError(f"Cosmos3 maskless pass {name!r} planned an empty varlen segment.")
    query_runs = [run for q_runs, _ in segments for run in q_runs]
    key_runs = [run for _, k_runs in segments for run in k_runs]
    if len(set(query_runs)) != len(query_runs):
        raise ValueError(f"Cosmos3 maskless pass {name!r} would attend a query more than once.")
    q_index = _expand_runs(run_starts, run_lengths, query_runs, num_und_tokens)
    k_index = _expand_runs(run_starts, run_lengths, key_runs, 0 if keys_from_und else num_und_tokens)
    key_stream_tokens = num_und_tokens if keys_from_und else num_gen_tokens
    identity_q = q_index.numel() == num_gen_tokens and bool(torch.equal(q_index, torch.arange(num_gen_tokens)))
    identity_k = k_index.numel() == key_stream_tokens and bool(torch.equal(k_index, torch.arange(key_stream_tokens)))
    meta = torch.zeros(META_SIZE, dtype=torch.int64)
    meta[META_MAX_SEQLEN_Q] = max(query_lengths)
    meta[META_MAX_SEQLEN_K] = max(key_lengths)
    meta[META_KEYS_FROM_UND] = int(keys_from_und)
    meta[META_IDENTITY_Q] = int(identity_q)
    meta[META_IDENTITY_K] = int(identity_k)
    meta[META_GEN_TOKENS] = num_gen_tokens
    k_index = k_index.to(device)
    if keys_from_und and k_index.numel() > 1:
        # The only plan shape that follows the prompt: keep it symbolic so new
        # prompts and both CFG branches reuse one compiled GEN layer.
        torch._dynamo.mark_dynamic(k_index, 0)
    return MasklessPass(
        name=name,
        keys_from_und=keys_from_und,
        q_index=q_index.to(device),
        k_index=k_index,
        cu_seqlens_q=_cumulative_offsets(query_lengths).to(device),
        cu_seqlens_k=_cumulative_offsets(key_lengths).to(device),
        meta=meta,
        segments=tuple((tuple(q_runs), tuple(k_runs)) for q_runs, k_runs in segments),
    )


def build_multiview_maskless_plan(
    layout: MultiviewLayout,
    num_und_tokens: int,
    device: torch.device | str,
) -> MultiviewMasklessPlan:
    """Partition the exact visibility predicate into unmasked varlen passes.

    All planning runs on the host; only the final index tensors move to
    ``device``.  The plan depends on the layout and the real caption lengths,
    so the caller caches it per request and CFG branch.
    """
    device = torch.device(device)
    num_gen_tokens = layout.gen_tokens
    geometry = PaddedAttentionGeometry(num_gen_tokens, num_gen_tokens, num_und_tokens, num_und_tokens)
    metadata = build_multiview_flex_metadata(layout, geometry, "cpu")
    key_vectors = metadata.key_vectors()
    _, run_starts = _semantic_groups(metadata.key_grouping_vectors())
    run_ends = torch.cat([run_starts[1:], torch.tensor([metadata.kv_len], dtype=run_starts.dtype)])
    run_lengths = run_ends - run_starts
    sample, _, view, is_control, is_und, timestamp = (vector[run_starts] for vector in key_vectors)

    gen_runs = torch.nonzero(~is_und).flatten()
    und_runs = torch.nonzero(is_und).flatten().tolist()
    q_row_of_run = torch.full((run_starts.numel(),), -1, dtype=torch.int64)
    q_row_of_run[gen_runs] = torch.arange(gen_runs.numel())
    pair_allowed = _make_pair_allowed(metadata.query_vectors(), key_vectors, layout.cross_view_past_window_seconds)
    allowed = pair_allowed((run_starts[gen_runs] - num_und_tokens)[:, None], run_starts[None, :])

    # Pass 1: same (sample, view) groups in packed first-appearance order.
    groups: dict[tuple[int, int], list[int]] = {}
    for run in gen_runs.tolist():
        groups.setdefault((int(sample[run]), int(view[run])), []).append(run)
    group_runs = list(groups.values())
    for runs in group_runs:
        if not bool(allowed[q_row_of_run[runs]][:, runs].all()):
            raise RuntimeError("Cosmos3 maskless planner expected every same-view pair to be visible.")
    same_view_segments: list[Segment] = [(tuple(runs), tuple(runs)) for runs in group_runs]

    # Last pass: caption access is a property of the query's view group.
    caption_segments: list[Segment] = []
    for runs in group_runs:
        reads = allowed[q_row_of_run[runs]][:, und_runs]
        if reads.numel() and not bool(torch.equal(reads, reads[:1].expand_as(reads))):
            raise RuntimeError("Cosmos3 maskless planner expected one caption access rule per view group.")
        keys = [und_runs[column] for column in torch.nonzero(reads[0]).flatten().tolist()] if reads.numel() else []
        if keys:
            caption_segments.append((tuple(runs), tuple(keys)))

    # Middle passes: whatever the predicate admits beyond same-view and captions
    # must be sensor-to-sensor edges between distinct views at admissible times.
    residual = allowed.clone()
    if und_runs:
        residual[:, und_runs] = False
    for runs in group_runs:
        residual[q_row_of_run[runs][:, None], torch.tensor(runs)[None, :]] = False
    if bool(residual[is_control[gen_runs]].any()) or bool(residual[:, is_control].any()):
        raise RuntimeError("Cosmos3 maskless planner found cross-view edges touching control tokens.")
    if bool((residual & (view[gen_runs][:, None] == view[None, :])).any()):
        raise RuntimeError("Cosmos3 maskless planner found a same-view edge outside the same-view pass.")

    view_order: dict[int, int] = {}
    for run in gen_runs.tolist():
        view_order.setdefault(int(view[run]), len(view_order))
    time_groups: dict[tuple[int, float], list[int]] = {}
    for run in gen_runs.tolist():
        if not bool(is_control[run]):
            time_groups.setdefault((int(sample[run]), float(timestamp[run])), []).append(run)

    levels: list[list[Segment]] = []

    def split(views: list[int], q_runs: list[int], k_runs: list[int], depth: int) -> None:
        if len(views) < 2:
            return
        middle = len(views) // 2
        left, right = views[:middle], views[middle:]
        for q_half, k_half in ((left, right), (right, left)):
            q_views, k_views = set(q_half), set(k_half)
            qs = tuple(run for run in q_runs if int(view[run]) in q_views)
            ks = tuple(run for run in k_runs if int(view[run]) in k_views)
            if qs and ks:
                while len(levels) <= depth:
                    levels.append([])
                levels[depth].append((qs, ks))
        split(left, q_runs, k_runs, depth + 1)
        split(right, q_runs, k_runs, depth + 1)

    for runs in time_groups.values():
        rows = residual[q_row_of_run[runs]]
        admissible = rows.any(0)
        if not bool(admissible.any()):
            continue
        # The window depends only on the query time, so rows of one time group
        # differ solely by excluding their own view.
        for run, row in zip(runs, rows, strict=True):
            if not bool(torch.equal(row, admissible & (view != view[run]))):
                raise RuntimeError("Cosmos3 maskless planner expected one cross-view key set per query time.")
        key_runs = torch.nonzero(admissible).flatten().tolist()
        views = sorted(
            {int(view[run]) for run in runs} | {int(view[run]) for run in key_runs}, key=view_order.__getitem__
        )
        split(views, runs, key_runs, 0)

    ordered: list[tuple[str, bool, list[Segment]]] = [("same_view", False, same_view_segments)]
    ordered.extend((f"cross_view_L{depth}", False, segments) for depth, segments in enumerate(levels))
    if caption_segments:
        ordered.append(("caption", True, caption_segments))

    # Coverage: every admitted pair exactly once, nothing else.
    count = torch.zeros_like(allowed, dtype=torch.int64)
    for _, _, segments in ordered:
        for q_runs, k_runs in segments:
            count[q_row_of_run[list(q_runs)][:, None], torch.tensor(k_runs)[None, :]] += 1
    if not torch.equal(count, allowed.to(torch.int64)):
        raise RuntimeError("Cosmos3 maskless plan does not cover the visibility predicate exactly once.")
    if levels and len(levels) > math.ceil(math.log2(len(group_runs))):
        raise RuntimeError("Cosmos3 maskless plan used more cross-view levels than the view hierarchy allows.")

    passes = tuple(
        _materialize_pass(
            name,
            segments,
            keys_from_und=keys_from_und,
            run_starts=run_starts,
            run_lengths=run_lengths,
            num_und_tokens=num_und_tokens,
            num_gen_tokens=num_gen_tokens,
            device=device,
        )
        for name, keys_from_und, segments in ordered
    )
    if passes[0].q_index.numel() != num_gen_tokens:
        raise RuntimeError("Cosmos3 maskless same-view pass must cover every GEN row.")
    return MultiviewMasklessPlan(
        num_gen_tokens=num_gen_tokens,
        num_und_tokens=num_und_tokens,
        num_levels=len(levels),
        passes=passes,
        run_starts=run_starts,
        run_lengths=run_lengths,
    )
