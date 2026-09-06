# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from dataclasses import dataclass, field
from typing import ClassVar

logger = logging.getLogger(__name__)


def max_ic_for_chunk_size(chunk_size: int) -> int:
    """Largest power of 2 strictly less than chunk_size."""
    if chunk_size <= 2:
        return 1
    return 1 << ((chunk_size - 1).bit_length() - 1)


def compute_dynamic_initial_chunk_size(
    active_requests: int,
    max_num_seqs: int,
    max_ic: int,
) -> int:
    """Select IC from power-of-2 steps [2, 4, ..., max_ic] based on load factor.

    - Low load: small IC (faster TTFA).
    - High load: large IC (amortise decode cost).
    """
    steps: list[int] = []
    v = 2
    while v <= max_ic:
        steps.append(v)
        v <<= 1
    if not steps:
        return max(1, max_ic)
    if max_num_seqs <= 0:
        return steps[0]
    load_factor = min(active_requests / max_num_seqs, 1.0)
    idx = int(round(load_factor * (len(steps) - 1)))
    return steps[idx]


def parse_chunk_ramp(cfg: dict, steady: int | None = None) -> list[int] | None:
    """Parse ``codec_chunk_ramp`` from connector extra config.

    Accepts a list of positive ints or a comma-separated string.
    Returns ``None`` when the key is absent or invalid (ramp disabled).
    Requires >= 2 entries; all entries must be > 0.

    When ``steady`` is provided, warns if ``ramp[-1] != steady`` (the last
    ramp entry should match ``codec_chunk_frames`` to avoid reintroducing
    the cliff the ramp is meant to eliminate).
    """
    raw = cfg.get("codec_chunk_ramp")
    if raw is None:
        return None
    try:
        if isinstance(raw, str):
            raw = [int(x.strip()) for x in raw.split(",")]
        ramp = [int(x) for x in raw]
    except (TypeError, ValueError):
        logger.warning("codec_chunk_ramp parse error for %r; disabling ramp.", raw)
        return None
    if len(ramp) < 2:
        logger.warning("codec_chunk_ramp needs >= 2 entries, got %d; disabling.", len(ramp))
        return None
    if any(x <= 0 for x in ramp):
        logger.warning("codec_chunk_ramp entries must be positive; disabling.")
        return None
    if steady is not None and ramp[-1] != steady:
        logger.warning(
            "codec_chunk_ramp[-1]=%d != codec_chunk_frames=%d; this reintroduces the chunk-size cliff at index %d.",
            ramp[-1],
            steady,
            len(ramp) - 1,
        )
    return ramp


def ramp_chunk_size(index: int, ramp: list[int], steady: int) -> int:
    """Return the target chunk size for chunk ``index`` under the ramp schedule.

    Indices within the ramp table use ``ramp[index]``; indices past the table
    fall back to ``steady`` (typically ``codec_chunk_frames``).
    """
    if index < len(ramp):
        return ramp[index]
    return steady


def ramp_cumulative(index: int, ramp: list[int], steady: int) -> int:
    """Cumulative frame count through chunk ``index`` (inclusive).

    O(len(ramp)) closed form: sum the ramp table once, then add
    ``(index + 1 - len(ramp)) * steady`` for indices past the table.
    """
    ramp_len = len(ramp)
    if index < ramp_len:
        return sum(ramp[: index + 1])
    return sum(ramp) + (index + 1 - ramp_len) * steady


def compute_ramp_emit(
    length: int,
    chunk_index: int,
    ramp: list[int],
    steady: int,
    finished: bool,
) -> tuple[bool, int]:
    """Decide whether to emit a chunk under the ramp schedule.

    Uses cumulative thresholds to determine emission boundaries.  Each chunk
    ``i`` covers frames ``[ramp_cumulative(i-1), ramp_cumulative(i))``.

    Returns:
        ``(emit, context_length)``

        - ``emit=False, context_length=0``: not enough frames yet, hold.
        - ``emit=True, context_length>0``: emit with this many new frames.
        - ``emit=True, context_length=0``: finished with no new frames
            (caller should emit an empty-finished sentinel).
    """
    threshold = ramp_cumulative(chunk_index, ramp, steady)
    prev_threshold = ramp_cumulative(chunk_index - 1, ramp, steady) if chunk_index > 0 else 0

    if not finished and length < threshold:
        return False, 0

    if finished and length <= prev_threshold:
        return True, 0

    return True, length - prev_threshold


@dataclass
class AdaptiveChunkController:
    """Buffer-feedback controller for adaptive chunk sizing.

    Replaces the fixed ramp table with a greedy buffer-feedback rule.
    Chunk 0 uses dynamic IC; chunk 1+ sizes are decided per-chunk from
    live buffer-depth signals.

    State is per-segment: the controller is discarded and rebuilt at
    segment boundaries via ``_adaptive_states.pop()`` in the adapter.
    """

    frame_time_ewma_ms: float = 0.0
    first_emit_time: float = 0.0
    last_emit_time: float = 0.0
    frames_emitted_this_segment: int = 0
    steady_state_reached: bool = False
    accumulated_at_last_emit: int = 0
    last_target: int = 0
    # Per-request underrun telemetry, mirroring stream_client.py's playback
    # cursor model so server-side numbers are comparable to client-reported
    # AUDIO_UNDERRUN. Collected in record_emit; logged once at request end via
    # log_summary(). total_gap_s accumulates only gaps > 5ms (matching
    # stream_client's silence-insert threshold); per_chunk_gap_ms records every
    # chunk's gap (including 0) for the full trajectory.
    chunk_sizes: list = field(default_factory=list)
    per_chunk_gap_ms: list = field(default_factory=list)
    playback_cursor_s: float = 0.0
    total_gap_s: float = 0.0
    # Ramp-rate tuning, sourced from connector config (codec_chunk_ramp_divisor
    # / codec_chunk_ramp_delta_min) so deploys can tune the ramp without code
    # changes. Defaults preserve the original 15/2 behavior.
    ramp_divisor: int = 15
    ramp_delta_min: int = 2

    EWMA_ALPHA: ClassVar[float] = 0.3
    FRAME_DURATION_MS: ClassVar[float] = 80.0

    def compute_next_chunk_size(
        self,
        now: float,
        min_frames: int,
        max_frames: int,
        safety_margin_ms: float,
    ) -> int:
        if self.steady_state_reached:
            return max_frames

        emitted_ms = self.frames_emitted_this_segment * self.FRAME_DURATION_MS
        elapsed_ms = (now - self.first_emit_time) * 1000.0
        buffer_ms = emitted_ms - elapsed_ms

        # Hysteresis: only check when EWMA has a valid measurement.
        if self.frame_time_ewma_ms > 0:
            steady_gen_ms = max_frames * self.frame_time_ewma_ms
            if buffer_ms >= steady_gen_ms + safety_margin_ms:
                self.steady_state_reached = True
                return max_frames

        # Greedy: select the largest n where buffer can absorb
        # n * frame_time + safety_margin.
        available_ms = max(0.0, buffer_ms - safety_margin_ms)
        if self.frame_time_ewma_ms > 0:
            max_n = int(available_ms / self.frame_time_ewma_ms)
        else:
            max_n = min_frames
        greedy_result = max(min_frames, min(max_n, max_frames))

        # Ramp floor: chunk size grows by an EWMA-scaled delta per emit.
        # Unconditional (never gated on buffer sign): in the saturated
        # regime (per-stream RTF > 1) buffer is structurally negative and
        # shrinking chunks would multiply Code2Wav decode cost -- each chunk
        # re-decodes the left_context, cost ~ (72+n)/n, so n=2 costs ~10x
        # n=25. Growing toward max minimizes decode passes, the only lever
        # pre-W5. delta scales with per-frame time: slow hardware (NPU,
        # ewma~150) ramps fast (delta=10, ~3 emits to max); fast hardware
        # (NVIDIA, ewma~16) ramps gently (delta=2) for lower TTFP. This
        # cross-hardware self-tuning is why the adaptive controller exists.
        # last_target is set to the controller's target_size in record_emit
        # (on actual emit), not here -- compute_next_chunk_size may be called
        # multiple times per emit as frames accumulate one-by-one.
        # WARNING: do NOT gate ramp_floor on buffer>0. That collapses to
        # greedy=min_frames under load and triggers a vocoder-cost death
        # spiral (see #4855 linyueqian 2026-07-19 comment + W5 coupling).
        # Downward chunk-size adaptation is deferred until W5 (streaming
        # decoder state) lands -- see PR description.
        if self.frame_time_ewma_ms > 0:
            ramp_delta = max(self.ramp_delta_min, int(self.frame_time_ewma_ms / self.ramp_divisor))
        else:
            ramp_delta = self.ramp_delta_min
        ramp_floor = self.last_target + ramp_delta
        result = max(ramp_floor, greedy_result)
        result = min(result, max_frames)

        return result

    def record_emit(
        self,
        now: float,
        n_frames: int,
        accumulated: int,
        target_size: int,
    ) -> None:
        if self.frames_emitted_this_segment > 0:
            dt_ms = (now - self.last_emit_time) * 1000.0
            measured_ms = dt_ms / n_frames if n_frames > 0 else self.frame_time_ewma_ms
            if self.frame_time_ewma_ms == 0:
                self.frame_time_ewma_ms = measured_ms
            else:
                self.frame_time_ewma_ms = (
                    self.EWMA_ALPHA * measured_ms + (1 - self.EWMA_ALPHA) * self.frame_time_ewma_ms
                )
        # Per-chunk underrun telemetry: mirror stream_client.py's playback
        # cursor so server-side total_gap is comparable to the client's
        # AUDIO_UNDERRUN. arrival_s = wall since first emit; gap_s = how late
        # this chunk is vs where playback had reached.
        arrival_s = now - self.first_emit_time
        audio_dur_s = n_frames * self.FRAME_DURATION_MS / 1000.0
        gap_s = max(0.0, arrival_s - self.playback_cursor_s)
        if gap_s > 0.005:
            self.total_gap_s += gap_s
        self.per_chunk_gap_ms.append(gap_s * 1000.0)
        self.chunk_sizes.append(n_frames)
        self.playback_cursor_s = max(arrival_s, self.playback_cursor_s) + audio_dur_s
        self.last_emit_time = now
        self.frames_emitted_this_segment += n_frames
        self.accumulated_at_last_emit = accumulated
        # last_target tracks the controller's intended target_size, NOT the
        # actual n_frames shipped. With the B1 fix (compute_adaptive_emit
        # returns new_frames), n_frames can overshoot target_size during a
        # latency spike. Using n_frames here would ratchet ramp_floor to the
        # spike size and lock the controller toward max_frames prematurely.
        self.last_target = target_size

    def log_summary(self, request_id: str) -> None:
        """Emit one INFO line at segment/request end.

        Mirrors stream_client.py's underrun model so total_gap_s is
        comparable to the client-reported AUDIO_UNDERRUN. Fires on
        ``finished`` which is ``is_segment_finished or is_finished`` (see
        chunk_transfer_adapter), so a resumable /v1/realtime session emits
        one line per segment. INFO level so it is visible by default
        without any env var or dedicated logger. Only emits fields vLLM
        stats do NOT already report: chunk-size trajectory, per-chunk
        underrun, and total underrun. (pure/wall/ewma were dropped: vLLM
        already reports audio_duration, e2e/stage wall, and
        tpot/time_per_output_unit respectively; a trailing ewma also could
        not reflect its intra-request evolution.)
        """
        logger.info(
            "[ADAPTIVE] req=%s done: chunks=%s gaps_ms=%s total_gap=%.3fs",
            request_id,
            self.chunk_sizes,
            [round(g, 1) for g in self.per_chunk_gap_ms],
            self.total_gap_s,
        )


def compute_adaptive_emit(
    length: int,
    accumulated_at_last_emit: int,
    target_size: int,
    finished: bool,
) -> tuple[bool, int]:
    """Decide whether to emit under the adaptive controller.

    Same return contract as ``compute_ramp_emit``:

    - ``(False, 0)``: hold.
    - ``(True, >0)``: emit with this many new frames.
    - ``(True, 0)``: finished with no new frames (empty-finished sentinel).
    """
    new_frames = length - accumulated_at_last_emit

    if finished and new_frames <= 0:
        return True, 0

    # When finished, flush ALL remaining frames to avoid data loss.
    # Truncating to target_size would silently drop new_frames - target_size frames.
    if finished:
        return True, new_frames

    if not finished and new_frames < target_size:
        return False, 0

    # Ship ALL accumulated frames, not just target_size. When new_frames
    # exceeds target_size (e.g. a scheduler stall dropped target below the
    # already-queued frames), returning target_size would cause the caller's
    # [-context_length:] slice to take the newest frames, silently skipping
    # the surplus. Shipping new_frames keeps the slice contiguous with the
    # previous emit and preserves Code2Wav's per-request decoder cache.
    return True, new_frames


def parse_adaptive_config(
    cfg: dict,
) -> tuple[bool, int, float, int, int]:
    """Parse adaptive chunk settings from connector extra config.

    The adaptive max is ``codec_chunk_frames`` (applied at the call site
    as ``max_frames``); there is no separate max config -- capping below
    steady would undermine the controller's ramp-to-large design (more
    Code2Wav re-decode tax) and is a saturated-regime death-spiral risk,
    so it is intentionally not exposed.

    Returns:
        ``(enabled, min_frames, safety_margin_ms, ramp_divisor, ramp_delta_min)``
    """
    enabled = bool(cfg.get("codec_chunk_adaptive", False))
    min_frames = int(cfg.get("codec_chunk_min_frames", 2))
    margin_ms = float(cfg.get("codec_chunk_safety_margin_ms", 50.0))
    ramp_divisor = int(cfg.get("codec_chunk_ramp_divisor", 15))
    ramp_delta_min = int(cfg.get("codec_chunk_ramp_delta_min", 2))
    if min_frames < 1:
        logger.warning("codec_chunk_min_frames must be >= 1, clamping to 1.")
        min_frames = 1
    if ramp_divisor < 1:
        logger.warning("codec_chunk_ramp_divisor must be >= 1, clamping to 1.")
        ramp_divisor = 1
    if ramp_delta_min < 1:
        logger.warning("codec_chunk_ramp_delta_min must be >= 1, clamping to 1.")
        ramp_delta_min = 1
    return enabled, min_frames, margin_ms, ramp_divisor, ramp_delta_min
