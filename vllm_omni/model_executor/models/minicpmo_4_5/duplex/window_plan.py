# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Position and block accounting for the MiniCPM-o 4.5 duplex KV window.

MiniCPM-o's native duplex streaming has a bounded attention budget: once a
session's context passes the window, the oldest content has to go. The HF
reference implementation does that by trimming the cached K/V tensors and
re-RoPE-ing the survivors (``slide_trigger_seconds`` / ``slide_stride_seconds``
in the streaming processor, which is a *separate* window living in the audio
CNN). This module is the same rule, in the units a paged KV cache schedules in:
tokens, blocks and watermarks.

Why trimming and renumbering are one event, not two
---------------------------------------------------
The alternative is to keep every position and only mask the out-of-window
range, which is what vLLM's sliding-window cache types do. Neither fits a
duplex session:

* ``SlidingWindowSpec`` masks a *contiguous* tail window -- the kernel is handed
  ``window_size=(W-1, 0)`` -- so it cannot keep a head prefix attended while
  dropping what sits right after it.
* ``RSWASpec`` can, that being its whole shape, but it defines its sink as
  ``num_prompt_tokens``, which for a session that grows by appends is the first
  append rather than the head context the duplex protocol preserves. It also
  needs a backend that can express the mask: FA4's ``rswa_mask_mod``, Triton or
  FlexAttention. On a FlashAttention-3 Stage 0 the branch is skipped and the
  zeroed gap blocks are attended as ordinary context.

Renumbering removes the gap outright instead. The retained span is a contiguous
``[0, prefix + window)`` range, plain causal attention over it is what the model
was trained to do, and no backend has to know a window exists. The price is one
rotation of the retained keys, which is what :class:`PositionReanchor` plans.

Everything here is pure integer arithmetic so the invariants are testable
without a GPU, a checkpoint, or a vLLM engine. ``window_kv.py`` is the vLLM
tier that consumes it.

Paged KV Retention Semantics vs. Unpaged Reference
--------------------------------------------------
In the unpaged HuggingFace reference implementation (OpenBMB MiniCPM-o 4.5), KV
tensors are dense PyTorch tensors. The reference basic window drops whole units
immediately after the exact preserved prefix length:
`retained = [0, prefix_tokens) + [unit_end, total)`.

In vLLM, KV memory is paged into physical blocks of size `block_size` (e.g. 16 or 128).
To achieve zero-copy in-place KV reuse without allocating new blocks or shifting
physical slot rows across pages:
1. The sink region is aligned up to whole blocks:
   `sink_end = cdiv(prefix_tokens, block_size) * block_size`.
2. Any tokens in `[prefix_tokens, sink_end)` reside in the physical sink block and
   remain resident in-place (up to `block_size - 1` tokens).
3. The dropped interval is `[sink_end, moved_from)`.
4. The retained tail `[moved_from, total)` is translated down by `delta = moved_from - sink_end`,
   where `delta` is an exact multiple of `block_size`.

This guarantees that physical pages in the block table require no data movement or
reallocation, at the cost of retaining up to `block_size - 1` tokens of the oldest
unit in the sink page.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.v1.request import Request, StreamingUpdate

# The Stage-0 window policy owns interruption handoff. Managed sessions keep
# a settled input snapshot until their replacement claims it or the session
# closes. External replicas use the bounded, completed-unit offer below.
#: Maximum concurrent externally held completed-unit offers.
CARRIED_WINDOW_MAX_SESSIONS = 32
#: Seconds the scheduler offers an aborted request's window to the session's
#: next Stage-0 request. The next append normally follows within a second.
CARRIED_WINDOW_OFFER_TTL_S = 10.0


def align_up(value: int, unit: int) -> int:
    if unit <= 0:
        raise ValueError(f"unit must be positive, got {unit}")
    return -((-int(value)) // unit) * unit


def cdiv(value: int, divisor: int) -> int:
    return -((-int(value)) // int(divisor))


@dataclass(frozen=True)
class DuplexWindowGeometry:
    """Window policy for one duplex session, in tokens.

    Attributes:
        prefix_tokens: The head context that outlives every trim (system
            instructions, reference audio, suffix). It stays at position 0.
        window_tokens: Low watermark -- how much appended content survives a
            trim, so a trim lands the session on ``prefix_tokens +
            window_tokens``.
        block_size: KV cache block size.
        max_model_len: Position budget of the stage. A windowed session is not
            supposed to run against this; it bounds the degenerate case where
            the window is the whole context.
        sample_room: Tokens that must stay free after the prompt to decode.
        high_watermark_tokens: Trigger point -- a trim is planned once the
            projected content passes this. Defaults to the low watermark, i.e.
            hold the budget exactly; the reference implementation's
            trigger/stride pair is the same knob measured in seconds.
    """

    prefix_tokens: int
    window_tokens: int
    block_size: int
    max_model_len: int
    sample_room: int = 1
    high_watermark_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.prefix_tokens < 0 or self.window_tokens <= 0:
            raise ValueError(f"invalid window geometry: prefix={self.prefix_tokens} window={self.window_tokens}")
        if self.block_size <= 0 or self.max_model_len <= 0:
            raise ValueError(
                f"invalid window geometry: block_size={self.block_size} max_model_len={self.max_model_len}"
            )
        if self.sample_room < 0:
            raise ValueError(f"sample_room must be non-negative, got {self.sample_room}")
        if self.high_watermark_tokens is not None and self.high_watermark_tokens < self.window_tokens:
            raise ValueError(
                f"high_watermark_tokens {self.high_watermark_tokens} is below window_tokens {self.window_tokens}"
            )

    @property
    def trigger_tokens(self) -> int:
        """Content tokens at which a trim becomes due, before the prefix."""
        return self.window_tokens if self.high_watermark_tokens is None else self.high_watermark_tokens


def unit_starts(geometry: DuplexWindowGeometry, unit_tokens: list[int]) -> list[int]:
    """Absolute position of each unit's first token, after the fixed prefix.

    ``unit_tokens`` is one entry per accepted append: every processor chunk and
    internal closure token that append wrote into the KV cache. The positions
    are read after the last trim, so the caller owns the history this is derived
    from; a session that has already re-anchored reports shifted lengths.
    """
    starts: list[int] = []
    position = geometry.prefix_tokens
    for unit in unit_tokens:
        starts.append(position)
        position += int(unit)
    return starts


@dataclass(frozen=True)
class PositionReanchor:
    """A renumbering of the retained tail, and the rotation it implies.

    Attributes:
        delta: Positions subtracted from every retained token at or after
            ``moved_from``. A multiple of ``block_size``, which is what lets the
            tail keep the physical slots it is already written in: the block
            table loses its gap entries, so logical index ``j`` becomes
            ``j - delta / block_size``, and ``block_table[j']`` still names the
            page that holds the row.
        moved_from: First absolute position that moves.
        sink_blocks: Table prefix that stays where it is.
    """

    delta: int
    moved_from: int
    sink_blocks: int

    @property
    def sink_end(self) -> int:
        """Position the retained tail lands on, i.e. the end of the sink."""
        return self.moved_from - self.delta


def plan_position_reanchor(
    geometry: DuplexWindowGeometry,
    *,
    computed_tokens: int,
    pending_tokens: int,
    target_tokens: int | None = None,
    unit_tokens: list[int] | None = None,
) -> PositionReanchor | None:
    """Plan the trim that keeps the next append inside the window.

    Args:
        computed_tokens: Positions written so far, after the last trim.
        pending_tokens: What the next append is expected to add.
        target_tokens: Where the sequence should land; defaults to the sink plus
            the low-watermark window.
        unit_tokens: Per-append committed lengths. When given, the cut snaps down
            to a unit start, so a unit is either fully kept or fully gone --
            dropping half a unit would be a behaviour change against a window
            that drops whole ones.

    Returns ``None`` while the session is inside the trigger watermark.
    """
    projected = int(computed_tokens) + int(pending_tokens)
    if projected <= geometry.prefix_tokens + geometry.trigger_tokens:
        return None
    sink_end_block = cdiv(geometry.prefix_tokens, geometry.block_size)
    sink_end = sink_end_block * geometry.block_size
    target = geometry.prefix_tokens + geometry.window_tokens if target_tokens is None else int(target_tokens)
    target = max(sink_end + geometry.block_size, target)
    if target >= projected:
        # Nothing to reclaim: the caller must finish the session with
        # context_length_exceeded instead of moving the window start backwards.
        return None

    drop_needed = projected - target
    if unit_tokens:
        boundaries = unit_starts(geometry, unit_tokens)
        # To avoid splitting units in half, moved_from must land on an exact unit start
        # that is ALSO a page boundary (b % block_size == 0) and at or after sink_end.
        # We need moved_from >= sink_end + drop_needed so post-trim length <= target.
        min_cut = sink_end + drop_needed
        candidates = [b for b in boundaries if b >= min_cut and b % geometry.block_size == 0 and b < projected]
        if not candidates:
            # No page-aligned whole-unit boundary satisfies the drop budget without
            # slicing into a unit. Return None so caller routes to fallback.
            return None
        moved_from = min(candidates)
        delta = moved_from - sink_end
    else:
        delta = align_up(drop_needed, geometry.block_size)
        moved_from = sink_end + delta
        if moved_from >= projected:
            return None

    return PositionReanchor(
        delta=delta,
        moved_from=moved_from,
        sink_blocks=sink_end_block,
    )


def prepare_stage0_window(
    session: Request,
    update: StreamingUpdate,
    *,
    segment_output_ids: list[int],
    completed_terminator: int | None = None,
) -> bool:
    """Plan an official-style MiniCPM Stage-0 window at unit boundaries.

    vLLM owns a paged KV cache, so deleting a middle span and rotating the
    retained K tensors in place is not a safe model hook.  Instead, record
    completed unit lengths in the scheduler and request a full prompt
    replacement when a watermark fires.  The worker rebuilds matching
    embeddings from its unit history, which recomputes RoPE at the new
    contiguous positions.
    """
    info = getattr(update, "model_intermediate_buffer", None)
    if not isinstance(info, dict):
        return False
    duplex = info.get("duplex")
    if not isinstance(duplex, dict) or duplex.get("data_plane") is not True:
        return False
    runtime_config = duplex.get("runtime_config")
    runtime_config = runtime_config if isinstance(runtime_config, dict) else {}
    window = runtime_config.get("duplex_window_config")
    mode = window.get("sliding_window_mode", "off") if isinstance(window, dict) else "off"
    retains_units = mode in {"basic", "context", "evidence"}
    if not retains_units:
        # Without a window the worker keeps no unit history to carry.
        session._minicpmo45_carried_window = None

    try:
        seq = int(duplex.get("seq", 0) or 0)
        preserve_len = int(runtime_config.get("duplex_first_append_context_tokens", 0) or 0)
    except (TypeError, ValueError):
        return False
    if seq <= 1:
        return False

    # The sampled terminator is discarded by the normal session update and
    # re-injected at the head of this append, followed by </unit>.
    # generated_ids contains only the confirmed output span retained by
    # the streaming update, excluding its final uncomputed sample.
    generated_ids = segment_output_ids
    # Sampler-side history can contain speculative or stale async output.
    # The worker must rebuild from the same accepted ids used for lengths.
    duplex["stage0_window"] = {"completed_token_ids": list(generated_ids)}
    if completed_terminator is not None:
        duplex["stage0_window"]["completed_terminator_token_id"] = int(completed_terminator)
    # The worker drops its carried units at the first plan it gets, placed or
    # not, so the carry is taken here, before any return below, and none is
    # handed to this request afterwards.
    carry = getattr(session, "_minicpmo45_carried_window", None)
    session._minicpmo45_carried_window = None
    session._minicpmo45_window_planned = True
    # Boundary reconciliation is required even with an unbounded context:
    # async lookahead can overwrite the worker's pending terminator after
    # the scheduler accepted the segment stop. Window retention is optional.
    if not isinstance(window, dict) or (
        not retains_units
        and (
            not runtime_config.get("duplex_boundary_token_ids")
            or duplex.get("session_window_owned") is False
            or duplex.get("session_window_replay") is False
        )
    ):
        return False
    base_len = int(getattr(session, "num_prompt_tokens", 0) or 0) + len(generated_ids)
    boundary = base_len + 2
    # A legitimate open_start of 0 (an empty context prefix) must not fall
    # back to preserve_len, which would double-count the suffix below.
    recorded_open_start = getattr(session, "_minicpmo45_window_open_start", None)
    open_start = preserve_len if recorded_open_start is None else int(recorded_open_start)
    unit_len = boundary - open_start
    if unit_len <= 0:
        return False

    special_ids = {
        int(token_id)
        for token_id in runtime_config.get("duplex_window_special_token_ids", ())
        if isinstance(token_id, int)
    }
    units = list(getattr(session, "_minicpmo45_window_units", ()))
    units.append(
        {
            "length": unit_len,
            "generated_token_ids": [int(token_id) for token_id in generated_ids if token_id not in special_ids],
        }
    )
    projected_len = base_len + len(update.prompt_token_ids)
    carried_units: list[dict[str, Any]] = []
    if carry is not None:
        # This request replaced one a barge-in aborted. Its first plan puts
        # that request's completed units back ahead of this one's own, with
        # the text the context window had already moved out of them.
        carried_units = [dict(unit) for unit in carry.get("units", ())]
        units = [*carried_units, *units]
        projected_len += sum(int(unit["length"]) for unit in carried_units)
        if mode == "context":
            session._minicpmo45_window_previous_token_ids = list(carry.get("previous", ()))
    drop_count = 0
    dropped_len = 0
    selected = None
    retrieval_plan = None
    retrieved_units = None

    if mode == "evidence":
        from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_evidence import (
            EvidenceScorer,
            select_evidence_units,
        )

        previous = getattr(session, "model_intermediate_buffer", None) or {}
        previous_duplex = previous.get("duplex") or {}
        observation = previous_duplex.get("window_evidence")
        scorer = getattr(session, "_minicpmo45_evidence_scorer", None)
        if scorer is None:
            scorer = session._minicpmo45_evidence_scorer = EvidenceScorer()
        if window.get("evidence_query_retrieval"):
            from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_retrieval import (
                evidence_score,
                plan_retrieval,
            )

            if carried_units:
                # Input sequence numbers are fence-local. Carried units precede
                # the new fence, and its former host archive has been released.
                session._minicpmo45_memory_offset = len(carried_units)
                for index, unit in enumerate(carried_units, 1):
                    unit.update(append_seq=index, evidence_score=0.0, salience=(0.0, 0.0))
            offset = getattr(session, "_minicpmo45_memory_offset", 0)
            frame, energy, onset = observation if observation is not None else (b"", 0.0, 0.0)
            salience = scorer.observe(frame, energy)
            costs = previous_duplex.get("window_modalities", (0, 0))
            units[-1].update(
                append_seq=offset + seq - 1,
                salience=salience,
                evidence_score=evidence_score(*salience),
                audio_peak=energy,
                boundary_onset=onset,
                modality_costs=costs if unit_len >= 3 + sum(costs) else (0, 0),
            )
            retrieved_units, archive, retrieval_plan = plan_retrieval(
                units,
                getattr(session, "_minicpmo45_memory_archive", {}),
                projected_len=projected_len,
                high=int(window["basic_window_high_tokens"]),
                low=int(window["basic_window_low_tokens"]),
                history_budget=int(window["evidence_history_tokens"]),
                relevance=[(offset + index, v, a) for index, v, a in duplex.get("memory_relevance", ())],
                restore_opening=not offset and duplex.get("memory_restore_opening") is True,
            )
            session._minicpmo45_memory_archive = archive
            dropped_len = sum(unit["length"] for unit in units) - sum(unit["length"] for unit in retrieved_units)
        elif observation is not None:
            route, units[-1]["evidence_score"] = scorer.score(*observation)
            if (costs := previous_duplex.get("window_modalities")) and route != "av":
                vision, audio = costs
                if unit_len >= 3 + vision + audio:
                    units[-1]["remove_media"] = (1 + vision, audio) if route == "vision" else (1, vision)
        if retrieved_units is None and projected_len > int(window["basic_window_high_tokens"]):
            unit_tokens = sum(unit["length"] for unit in units)
            fixed = projected_len - unit_tokens
            budget = max(0, int(window["basic_window_low_tokens"]) - fixed)
            selected = select_evidence_units(units, budget, int(window["evidence_history_tokens"]))
            drop_count = len(units) - len(selected)
            dropped_len = unit_tokens - sum(unit["length"] for _, unit in selected)
    elif mode == "basic":
        try:
            high = int(window.get("basic_window_high_tokens", 8000))
            low = int(window.get("basic_window_low_tokens", 6000))
        except (TypeError, ValueError):
            return False
        if projected_len > high:
            while units and projected_len - dropped_len > low:
                dropped_len += int(units[drop_count]["length"])
                drop_count += 1
                if drop_count >= len(units):
                    break
    elif mode == "context":
        try:
            max_units = int(window.get("context_max_units", 24))
            previous_max = int(window.get("context_previous_max_tokens", 500))
        except (TypeError, ValueError):
            return False
        drop_count = max(0, len(units) - max_units)
        dropped_len = sum(int(unit["length"]) for unit in units[:drop_count])
        previous = list(getattr(session, "_minicpmo45_window_previous_token_ids", ()))
        for unit in units[:drop_count]:
            previous.extend(unit["generated_token_ids"])
        if len(previous) > previous_max:
            previous = previous[-previous_max:]
        marker = [
            int(token_id)
            for token_id in runtime_config.get("duplex_window_previous_marker_token_ids", ())
            if isinstance(token_id, int)
        ]
        previous_with_marker = marker + previous if previous else []
        old_previous_len = int(getattr(session, "_minicpmo45_window_previous_len", 0) or 0)
        session._minicpmo45_window_previous_token_ids = previous
        session._minicpmo45_window_previous_len = len(previous_with_marker)
        projected_len += len(previous_with_marker) - old_previous_len

    retained_units = units[drop_count:] if selected is None else [unit for _, unit in selected]
    if retrieved_units is not None:
        retained_units = retrieved_units
    session._minicpmo45_window_units = retained_units
    if dropped_len == 0 and not carried_units and retrieval_plan is None:
        session._minicpmo45_window_open_start = boundary
        return False

    replacement_len = projected_len - dropped_len
    if replacement_len <= 0:
        return False
    scheduler_token_id = runtime_config.get("duplex_scheduler_token_id", 0)
    try:
        scheduler_token_id = max(0, int(scheduler_token_id))
    except (TypeError, ValueError):
        scheduler_token_id = 0
    update.prompt_token_ids = [scheduler_token_id] * replacement_len
    meta = info.setdefault("meta", {})
    if isinstance(meta, dict):
        meta["replace_streaming_prompt"] = True
    previous_ids = list(getattr(session, "_minicpmo45_window_previous_token_ids", ()))
    # The worker embeds the marker ids the window plan carries: it must not
    # re-tokenize the marker with its own tokenizer, which could differ
    # from the ids this length is computed from.
    marker_ids = [
        int(token_id)
        for token_id in runtime_config.get("duplex_window_previous_marker_token_ids", ())
        if isinstance(token_id, int)
    ]
    duplex["stage0_window"] = {
        "completed_token_ids": list(generated_ids),
        "replace": True,
        "mode": mode,
        "drop_units": drop_count,
        "dropped_tokens": dropped_len,
        "previous_token_ids": previous_ids,
        "previous_marker_token_ids": marker_ids,
        "replacement_prompt_len": replacement_len,
    }
    if selected is not None:
        duplex["stage0_window"]["evidence_keep"] = [
            (index, *(units[index]["remove_media"] if unit["length"] != units[index]["length"] else (0, 0)))
            for index, unit in selected
        ]
    if retrieval_plan is not None:
        duplex["stage0_window"].update(retrieval_plan)
        if retrieval_plan["retrieval_query"] and window.get("evidence_local_prefill") and not carried_units:
            # Only the contiguous unchanged suffix has a single RoPE displacement.
            end = preserve_len + sum(unit["length"] for unit in units)
            old_start = end
            last = len(units) - 1
            for source, _, removed, length in reversed(retrieval_plan["retrieval_keep"]):
                if last < 0 or source != last or removed or length != units[last]["length"]:
                    break
                old_start -= length
                last -= 1
            new_start = old_start - dropped_len
            keep = min(base_len, session.num_computed_tokens) - old_start
            if end == boundary and keep >= 512 and new_start > preserve_len and new_start + keep < replacement_len:
                duplex["stage0_window"]["local_prefill"] = (preserve_len, old_start, new_start, keep)
    if carried_units:
        duplex["stage0_window"]["carried_units"] = len(carried_units)
        duplex["stage0_window"]["carried_unit_lengths"] = [int(unit["length"]) for unit in carried_units]
    if completed_terminator is not None:
        duplex["stage0_window"]["completed_terminator_token_id"] = int(completed_terminator)
    recorded_prefix_len = runtime_config.get("duplex_window_prefix_tokens")
    prefix_len = preserve_len if recorded_prefix_len is None else int(recorded_prefix_len)
    suffix_len = len(runtime_config.get("duplex_window_suffix_token_ids", ()) or ())
    previous_len = int(getattr(session, "_minicpmo45_window_previous_len", 0) or 0)
    new_preserve_len = prefix_len + previous_len + suffix_len if mode == "context" else preserve_len
    session._minicpmo45_window_open_start = new_preserve_len + sum(int(unit["length"]) for unit in retained_units)
    return True
