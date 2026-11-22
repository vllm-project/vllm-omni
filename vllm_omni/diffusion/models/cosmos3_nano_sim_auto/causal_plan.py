# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Causal replay visibility and maskless pass planning for Cosmos3-Nano-Sim-Auto.

Every GEN query of the current chunk attends one softmax over the union of the
keys the replay predicate admits from three streams: the current chunk, the
chunks retained in the KV ring (``HISTORY``) and the caption documents
(``UND``). The planner partitions that key set into unmasked variable-length
attention passes whose partial results the shared maskless executor merges by
log-sum-exp (:mod:`..cosmos3.multiview_maskless_attention`). Each admitted
(query, key) pair is covered by exactly one pass, so the merge equals the
masked softmax.

The predicate is a term-for-term port of the reference teacher-forcing replay
rules with the production policy folded in: causal control visibility, controls
reading strictly-past clean sensor history, chunk-causal clean passes, the
decomposed view scope with a 0.4 s past-only cross-view window, and
deduplicated cross-view keys. Planning is host-side; only index tensors move
to the device.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import accumulate

import torch

from vllm_omni.diffusion.models.cosmos3.multiview_maskless_plan import (
    META_GEN_TOKENS,
    META_IDENTITY_K,
    META_IDENTITY_Q,
    META_KEY_STREAM,
    META_MAX_SEQLEN_K,
    META_MAX_SEQLEN_Q,
    META_SIZE,
    MasklessPass,
)

from .layout import (
    CAPTION_SCOPE_ALL,
    CAPTION_SCOPE_SAME_VIEW,
    ROLE_CLEAN_TARGET,
    ROLE_CONTROL,
    ROLE_CURRENT_TARGET,
    ROLE_TARGET_CONDITION,
    ROLE_UND,
    CaptionMetadata,
    RingSlot,
    TokenMetadata,
)

STREAM_CURRENT = 0
STREAM_HISTORY = 1
STREAM_UND = 2
STREAM_NAMES = ("current", "history", "und")

#: Reference tolerances: the window and caption bounds compare float32 seconds.
WINDOW_EPS = 1e-4
CAPTION_EPS = 1e-4
_INT32_LIMIT = 2**31


@dataclass(frozen=True)
class RunTable:
    """Contiguous token runs of one stream that agree on every predicate field.

    ``start``/``length`` address tokens inside the stream (virtual slot
    coordinates for ``HISTORY``). UND runs are whole caption documents and
    carry the time bounds; sensor runs carry ``-inf``/``inf`` placeholders.
    """

    stream: int
    start: torch.Tensor
    length: torch.Tensor
    view_id: torch.Tensor
    frame_id: torch.Tensor
    timestamp: torch.Tensor
    role: torch.Tensor
    step: torch.Tensor
    caption_scope: torch.Tensor
    caption_start: torch.Tensor
    caption_end: torch.Tensor

    @property
    def num_runs(self) -> int:
        return int(self.start.numel())

    @property
    def num_tokens(self) -> int:
        return int(self.length.sum()) if self.num_runs else 0

    @property
    def is_und(self) -> torch.Tensor:
        return self.role == ROLE_UND


def _run_boundaries(*fields: torch.Tensor) -> torch.Tensor:
    """First-token indexes of maximal runs with identical field values."""
    length = fields[0].numel()
    if length == 0:
        return torch.empty(0, dtype=torch.int64)
    change = torch.zeros(length, dtype=torch.bool)
    change[0] = True
    for field in fields:
        change[1:] |= field[1:] != field[:-1]
    return torch.nonzero(change).flatten()


def sensor_runs(metadata: TokenMetadata, *, stream: int, base: int = 0) -> RunTable:
    """Group a sensor stream by (view, frame, role, step); timestamps follow from frame and item."""
    starts = _run_boundaries(metadata.view_id, metadata.frame_id, metadata.role, metadata.step, metadata.caption_scope)
    ends = torch.cat([starts[1:], torch.tensor([metadata.num_tokens], dtype=torch.int64)])
    inf = torch.full((starts.numel(),), math.inf, dtype=torch.float32)
    return RunTable(
        stream=stream,
        start=starts + base,
        length=ends - starts,
        view_id=metadata.view_id[starts],
        frame_id=metadata.frame_id[starts],
        timestamp=metadata.timestamp[starts],
        role=metadata.role[starts],
        step=metadata.step[starts],
        caption_scope=metadata.caption_scope[starts],
        caption_start=-inf,
        caption_end=inf,
    )


def history_runs(slots: Sequence[RingSlot]) -> RunTable:
    """Concatenate the retained slots' runs in virtual ring coordinates."""
    tables = [
        sensor_runs(slot.metadata, stream=STREAM_HISTORY, base=slot.base) for slot in slots if slot.metadata.num_tokens
    ]
    if not tables:
        return _empty_runs(STREAM_HISTORY)
    return _concat_runs(tables)


def caption_runs(captions: CaptionMetadata) -> RunTable:
    """One run per caption document."""
    starts = torch.tensor(captions.doc_offsets, dtype=torch.int64)
    ends = torch.cat([starts[1:], torch.tensor([captions.num_tokens], dtype=torch.int64)])
    count = starts.numel()
    return RunTable(
        stream=STREAM_UND,
        start=starts,
        length=ends - starts,
        view_id=captions.view_id[starts],
        frame_id=torch.full((count,), -1, dtype=torch.int64),
        timestamp=torch.full((count,), -1.0, dtype=torch.float32),
        role=torch.full((count,), ROLE_UND, dtype=torch.int64),
        step=torch.full((count,), -1, dtype=torch.int64),
        caption_scope=torch.zeros(count, dtype=torch.int64),
        caption_start=captions.start_seconds[starts],
        caption_end=captions.end_seconds[starts],
    )


def _empty_runs(stream: int) -> RunTable:
    empty_long = torch.empty(0, dtype=torch.int64)
    empty_float = torch.empty(0, dtype=torch.float32)
    return RunTable(
        stream,
        empty_long,
        empty_long,
        empty_long,
        empty_long,
        empty_float,
        empty_long,
        empty_long,
        empty_long,
        empty_float,
        empty_float,
    )


def _concat_runs(tables: Sequence[RunTable]) -> RunTable:
    stream = tables[0].stream
    if any(table.stream != stream for table in tables):
        raise ValueError("Cannot concatenate run tables of different streams.")
    return RunTable(
        stream=stream,
        **{
            name: torch.cat([getattr(table, name) for table in tables])
            for name in (
                "start",
                "length",
                "view_id",
                "frame_id",
                "timestamp",
                "role",
                "step",
                "caption_scope",
                "caption_start",
                "caption_end",
            )
        },
    )


def pair_allowed(q: RunTable, k: RunTable, *, window_seconds: float) -> torch.Tensor:
    """Replay visibility ``[Q, K]`` between current-chunk query runs and key runs of one stream.

    Production policy constants: ``control_visibility="causal"`` (controls at
    steps <= the query's), ``controls_read_strict_past_clean_rgb=True``,
    ``clean_pass_causality="chunk"``, decomposed scope with a past-only window
    and deduplicated cross-view keys (a key is counted once; same-view and
    windowed cross-view are disjoint by construction here).
    """
    if not (math.isfinite(window_seconds) and window_seconds >= 0):
        raise ValueError(f"window_seconds must be finite and non-negative, got {window_seconds}.")
    if q.num_runs == 0 or k.num_runs == 0:
        return torch.zeros(q.num_runs, k.num_runs, dtype=torch.bool)
    q_view, k_view = q.view_id[:, None], k.view_id[None, :]
    same_view = q_view == k_view
    same_frame = q.frame_id[:, None] == k.frame_id[None, :]
    gap = q.timestamp[:, None] - k.timestamp[None, :]
    within_window = (gap >= -WINDOW_EPS) & (gap <= window_seconds + WINDOW_EPS)
    in_scope = same_view | within_window
    q_step, k_step = q.step[:, None], k.step[None, :]
    q_role, k_role = q.role[:, None], k.role[None, :]

    q_is_target = (q_role == ROLE_CURRENT_TARGET) | (q_role == ROLE_CLEAN_TARGET)
    q_is_current = q_role == ROLE_CURRENT_TARGET
    q_is_control = q_role == ROLE_CONTROL
    q_is_condition = q_role == ROLE_TARGET_CONDITION
    q_is_condition_like = q_is_control | q_is_condition

    k_is_und = k_role == ROLE_UND
    k_is_control = k_role == ROLE_CONTROL
    k_is_condition = k_role == ROLE_TARGET_CONDITION
    k_is_current = k_role == ROLE_CURRENT_TARGET
    k_is_clean = k_role == ROLE_CLEAN_TARGET
    # Reference ``kv_is_noisy_rgb``/``kv_is_clean_rgb`` are role-based and apply to LiDAR too.
    k_is_noisy_sensor = k_is_current
    k_is_clean_sensor = k_is_condition | k_is_clean

    control_step_allowed = k_step <= q_step
    q_scope = q.caption_scope[:, None]
    caption_reaches_query = k_is_und & (
        (q_scope == CAPTION_SCOPE_ALL) | ((q_scope == CAPTION_SCOPE_SAME_VIEW) & same_view)
    )
    q_time = q.timestamp[:, None]
    caption_reaches_query &= (q_time >= k.caption_start[None, :] - CAPTION_EPS) & (
        q_time < k.caption_end[None, :] - CAPTION_EPS
    )

    target_to_und = q_is_target & caption_reaches_query
    condition_to_und = q_is_condition_like & caption_reaches_query
    control_to_control = q_is_control & k_is_control & same_view & control_step_allowed
    control_to_clean_history = q_is_control & k_is_clean_sensor & (k_step < q_step) & in_scope
    condition_to_itself = q_is_condition & k_is_condition & same_frame & same_view
    condition_to_control = q_is_condition & k_is_control & same_view & control_step_allowed
    # Condition K/V is reused across chunks, so it must not encode noisy targets of its own chunk.
    condition_to_noisy = q_is_condition & k_is_noisy_sensor & (k_step < q_step) & in_scope
    target_to_current = q_is_current & k_is_current & (k_step == q_step) & in_scope
    clean_step_allowed = (q_is_current & (k_step < q_step)) | (~q_is_current & (k_step <= q_step))
    target_to_clean = q_is_target & k_is_clean & clean_step_allowed & in_scope
    target_to_control = q_is_target & k_is_control & same_view & control_step_allowed
    target_to_condition = q_is_target & k_is_condition & in_scope & (k_step <= q_step)
    return (
        target_to_und
        | condition_to_und
        | control_to_control
        | control_to_clean_history
        | condition_to_itself
        | condition_to_control
        | condition_to_noisy
        | target_to_current
        | target_to_clean
        | target_to_control
        | target_to_condition
    )


Segment = tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True, eq=False)
class CausalMasklessPlan:
    """Passes in merge order; the first pass covers every GEN query row."""

    num_gen_tokens: int
    num_history_tokens: int
    num_und_tokens: int
    passes: tuple[MasklessPass, ...]
    query_runs: RunTable
    key_runs: tuple[RunTable, RunTable, RunTable]
    #: Run-level truth table ``[Q, K_current + K_history + K_und]`` (diagnostics and tests).
    allowed: torch.Tensor

    def flatten(self) -> list[torch.Tensor]:
        return [tensor for attention_pass in self.passes for tensor in attention_pass.tensors()]

    def with_history_offsets(self, slot_physical_bases: Sequence[int], frame_unit: int) -> CausalMasklessPlan:
        """Remap HISTORY key indexes from virtual ``slot * frame_unit + offset`` to physical pool slots.

        ``slot_physical_bases[s]`` is the first flat-pool slot of the page that
        holds ring slot ``s`` (``block_id * block_size``); with block == frame
        unit the remap is a per-slot additive shift.
        """
        if frame_unit < 1:
            raise ValueError(f"frame_unit must be positive, got {frame_unit}.")
        bases = torch.tensor(list(slot_physical_bases), dtype=torch.int64)
        passes = []
        for attention_pass in self.passes:
            if int(attention_pass.meta[META_KEY_STREAM]) != STREAM_HISTORY:
                passes.append(attention_pass)
                continue
            k_index = attention_pass.k_index
            slot = torch.div(k_index, frame_unit, rounding_mode="floor")
            if slot.numel() and int(slot.max()) >= bases.numel():
                raise ValueError("History key index addresses a ring slot without a physical page.")
            remapped = k_index - slot * frame_unit + bases.to(k_index.device)[slot]
            meta = attention_pass.meta.clone()
            meta[META_IDENTITY_K] = 0
            passes.append(replace(attention_pass, k_index=remapped, meta=meta))
        return replace(self, passes=tuple(passes))


def _expand(runs: RunTable, selected: Sequence[int]) -> torch.Tensor:
    if not selected:
        return torch.empty(0, dtype=torch.int64)
    index = torch.tensor(list(selected), dtype=torch.int64)
    starts, lengths = runs.start[index], runs.length[index]
    total = int(lengths.sum())
    base = torch.repeat_interleave(starts - (torch.cumsum(lengths, 0) - lengths), lengths)
    return base + torch.arange(total, dtype=torch.int64)


def _cumulative(lengths: Sequence[int]) -> torch.Tensor:
    if sum(lengths) >= _INT32_LIMIT:
        raise ValueError("Cosmos3 causal maskless attention exceeds int32 cumulative sequence offsets.")
    return torch.tensor([0, *accumulate(lengths)], dtype=torch.int32)


def _segments(mask: torch.Tensor) -> list[Segment]:
    """Group query runs with identical admitted key sets into ``(q_runs, k_runs)`` rectangles."""
    segments: list[Segment] = []
    if mask.numel() == 0 or not bool(mask.any()):
        return segments
    rows, inverse = torch.unique(mask, dim=0, return_inverse=True)
    for group in range(rows.shape[0]):
        keys = torch.nonzero(rows[group]).flatten().tolist()
        if not keys:
            continue
        queries = torch.nonzero(inverse == group).flatten().tolist()
        segments.append((tuple(queries), tuple(keys)))
    segments.sort(key=lambda segment: segment[0][0])
    return segments


def _materialize(
    name: str,
    stream: int,
    segments: Sequence[Segment],
    *,
    query_runs: RunTable,
    key_runs: RunTable,
    key_stream_tokens: int,
    num_gen_tokens: int,
    device: torch.device,
) -> MasklessPass:
    query_lengths = [int(query_runs.length[list(q_runs)].sum()) for q_runs, _ in segments]
    key_lengths = [int(key_runs.length[list(k_runs)].sum()) for _, k_runs in segments]
    if any(length == 0 for length in (*query_lengths, *key_lengths)):
        raise RuntimeError(f"Cosmos3 causal maskless pass {name!r} planned an empty varlen segment.")
    q_order = [run for q_runs, _ in segments for run in q_runs]
    k_order = [run for _, k_runs in segments for run in k_runs]
    if len(set(q_order)) != len(q_order):
        raise RuntimeError(f"Cosmos3 causal maskless pass {name!r} would attend a query more than once.")
    q_index = _expand(query_runs, q_order)
    k_index = _expand(key_runs, k_order)
    identity_q = q_index.numel() == num_gen_tokens and bool(torch.equal(q_index, torch.arange(num_gen_tokens)))
    identity_k = k_index.numel() == key_stream_tokens and bool(torch.equal(k_index, torch.arange(key_stream_tokens)))
    meta = torch.zeros(META_SIZE, dtype=torch.int64)
    meta[META_MAX_SEQLEN_Q] = max(query_lengths)
    meta[META_MAX_SEQLEN_K] = max(key_lengths)
    meta[META_KEY_STREAM] = stream
    meta[META_IDENTITY_Q] = int(identity_q)
    meta[META_IDENTITY_K] = int(identity_k)
    meta[META_GEN_TOKENS] = num_gen_tokens
    k_index = k_index.to(device)
    if stream != STREAM_CURRENT and k_index.numel() > 1:
        # Prompt lengths and ring population vary per request/chunk kind; keep
        # those shapes symbolic so the compiled GEN layers are reused.
        torch._dynamo.mark_dynamic(k_index, 0)
    return MasklessPass(
        name=name,
        keys_from_und=stream == STREAM_UND,
        q_index=q_index.to(device),
        k_index=k_index,
        cu_seqlens_q=_cumulative(query_lengths).to(device),
        cu_seqlens_k=_cumulative(key_lengths).to(device),
        meta=meta,
        segments=tuple(segments),
    )


def build_causal_maskless_plan(
    *,
    current: TokenMetadata,
    history: Sequence[RingSlot],
    captions: CaptionMetadata,
    window_seconds: float,
    device: torch.device | str = "cpu",
) -> CausalMasklessPlan:
    """Partition the replay predicate into unmasked varlen passes over three key streams.

    Pass order: ``same_view_current`` (covers every row; seeds the merge),
    ``cross_view_current``, ``same_view_history``, ``cross_view_history``,
    ``caption``. Within a pass, query runs with identical admitted key sets
    share one varlen segment. Categories are disjoint (same vs. different view,
    current vs. history stream, UND), so every admitted pair lands in exactly
    one pass; the build asserts this against the run-level truth table.
    """
    device = torch.device(device)
    if current.num_tokens == 0:
        raise ValueError("The current chunk has no GEN tokens.")
    query_runs = sensor_runs(current, stream=STREAM_CURRENT)
    key_tables = (query_runs, history_runs(history), caption_runs(captions))
    allowed_by_stream = [pair_allowed(query_runs, table, window_seconds=window_seconds) for table in key_tables]
    num_history_tokens = max((slot.base + slot.metadata.num_tokens for slot in history), default=0)
    stream_tokens = (current.num_tokens, num_history_tokens, captions.num_tokens)

    plan_spec: list[tuple[str, int, torch.Tensor]] = []
    for stream, prefix in ((STREAM_CURRENT, "current"), (STREAM_HISTORY, "history")):
        table = key_tables[stream]
        if table.num_runs == 0:
            continue
        same_view = query_runs.view_id[:, None] == table.view_id[None, :]
        plan_spec.append((f"same_view_{prefix}", stream, allowed_by_stream[stream] & same_view))
        plan_spec.append((f"cross_view_{prefix}", stream, allowed_by_stream[stream] & ~same_view))
    if key_tables[STREAM_UND].num_runs:
        plan_spec.append(("caption", STREAM_UND, allowed_by_stream[STREAM_UND]))

    # Every query attends itself through its own view, so the first pass seeds every row.
    first_rows = plan_spec[0][2].any(dim=1)
    if plan_spec[0][0] != "same_view_current" or not bool(first_rows.all()):
        raise RuntimeError("Cosmos3 causal maskless same-view pass must cover every GEN query run.")

    passes: list[MasklessPass] = []
    coverage = [torch.zeros_like(mask, dtype=torch.int64) for mask in allowed_by_stream]
    for name, stream, mask in plan_spec:
        segments = _segments(mask)
        if not segments:
            continue
        for q_runs, k_runs in segments:
            coverage[stream][torch.tensor(q_runs)[:, None], torch.tensor(k_runs)[None, :]] += 1
        passes.append(
            _materialize(
                name,
                stream,
                segments,
                query_runs=query_runs,
                key_runs=key_tables[stream],
                key_stream_tokens=stream_tokens[stream],
                num_gen_tokens=current.num_tokens,
                device=device,
            )
        )
    for stream, (count, mask) in enumerate(zip(coverage, allowed_by_stream, strict=True)):
        if not torch.equal(count, mask.to(torch.int64)):
            raise RuntimeError(
                f"Cosmos3 causal maskless plan does not cover the {STREAM_NAMES[stream]} predicate exactly once."
            )
    if passes[0].q_index.numel() != current.num_tokens:
        raise RuntimeError("Cosmos3 causal maskless first pass must cover every GEN row.")
    return CausalMasklessPlan(
        num_gen_tokens=current.num_tokens,
        num_history_tokens=num_history_tokens,
        num_und_tokens=captions.num_tokens,
        passes=tuple(passes),
        query_runs=query_runs,
        key_runs=key_tables,
        allowed=torch.cat(allowed_by_stream, dim=1),
    )


def dense_token_mask(plan: CausalMasklessPlan) -> torch.Tensor:
    """Expand the run-level truth table to tokens: ``[GEN, current + history + und]`` (tests)."""
    q_runs, (cur, hist, und) = plan.query_runs, plan.key_runs
    q_rows = torch.repeat_interleave(torch.arange(q_runs.num_runs), q_runs.length)
    columns: list[torch.Tensor] = []
    for table, extent in zip((cur, hist, und), (plan.num_gen_tokens, plan.num_history_tokens, plan.num_und_tokens)):
        column = torch.full((extent,), -1, dtype=torch.int64)
        for run in range(table.num_runs):
            start, length = int(table.start[run]), int(table.length[run])
            column[start : start + length] = run
        columns.append(column)
    allowed = plan.allowed
    offsets = (0, cur.num_runs, cur.num_runs + hist.num_runs)
    blocks = []
    for column, offset in zip(columns, offsets):
        block = torch.zeros(q_rows.numel(), column.numel(), dtype=torch.bool)
        real = column >= 0
        if bool(real.any()):
            block[:, real] = allowed[q_rows][:, column[real] + offset]
        blocks.append(block)
    return torch.cat(blocks, dim=1)
