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
"""

from __future__ import annotations

from dataclasses import dataclass


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

    @property
    def rope_angle_shift_tokens(self) -> int:
        """Signed position offset to rotate retained keys by: ``-delta``.

        RoPE composes, so a token that moves from ``p`` to ``p - delta`` needs
        its cached key rotated by ``-delta * inv_freq`` and nothing else: the
        sink did not move, and a uniform translation of the tail leaves every
        within-tail relative distance untouched.
        """
        return -self.delta


def reanchor_positions(position: int, plan: PositionReanchor) -> int:
    """Map one absolute position through a re-anchor.

    The sink keeps its positions, everything from ``moved_from`` translates by
    ``-delta``, and the span in between ceases to exist.
    """
    if int(position) != position:
        raise ValueError(f"position must be an integer, got {position!r}")
    position = int(position)
    if plan.sink_end <= position < plan.moved_from:
        raise ValueError(
            f"position {position} was dropped by reanchor plan moved_from={plan.moved_from} delta={plan.delta}"
        )
    if position < plan.moved_from:
        return position
    return position - plan.delta


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
    target = geometry.prefix_tokens + geometry.window_tokens if target_tokens is None else int(target_tokens)
    target = max(geometry.prefix_tokens + geometry.block_size, target)
    if target >= projected:
        # Nothing to reclaim: the caller must finish the session with
        # context_length_exceeded instead of moving the window start backwards.
        return None
    boundaries = unit_starts(geometry, unit_tokens) if unit_tokens else None
    moved_from = align_up(projected - target, geometry.block_size)
    if boundaries:
        # ``align_up`` keeps the shift a whole number of pages, which is what
        # lets the retained rows stay where they physically are.
        aligned = [align_up(b, geometry.block_size) for b in boundaries if b >= geometry.prefix_tokens]
        candidates = [b for b in aligned if b <= moved_from and b > geometry.prefix_tokens]
        if candidates:
            moved_from = max(candidates)
    sink_end_block = cdiv(geometry.prefix_tokens, geometry.block_size)
    if moved_from <= sink_end_block * geometry.block_size:
        return None
    delta = moved_from - sink_end_block * geometry.block_size
    return PositionReanchor(
        delta=delta,
        moved_from=moved_from,
        sink_blocks=sink_end_block,
    )
