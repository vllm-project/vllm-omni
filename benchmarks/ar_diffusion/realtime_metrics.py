# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metric definitions for realtime AR-Diffusion session benchmarks.

Pure functions over recorded chunk events: no engine, no model, no device.
Everything here is derived from a :class:`WorkloadProfile` so swapping the
pipeline under test requires no change in this module.

Two load modes produce different metrics, and conflating them produces a
number no deployment can use:

``saturating``
    Each session issues its next tick as soon as the previous one returns.
    Measures the ceiling -- aggregate generated FPS and frames per GPU-second.
    Deadlines do not exist in this mode, so continuity metrics are ``None``.

``paced``
    Sessions consume at ``target_fps``, so a chunk is due once per release
    period. Measures continuity -- CPR, TTFC and worst-case chunk latency.
    SLO is only defined in this mode.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LoadMode(str, Enum):
    SATURATING = "saturating"
    PACED = "paced"


@dataclass(frozen=True)
class WorkloadProfile:
    """Serving-relevant shape of the pipeline under test.

    ``frames_per_chunk`` is the number of raw frames one committed chunk
    delivers, i.e. the model's latent frames per block multiplied by the VAE
    temporal factor. It comes from the pipeline capability rather than from a
    command-line default so the harness stays model-neutral.
    """

    frames_per_chunk: int
    target_fps: float
    buffer_chunks: int = 1
    resident_bytes_per_session: int | None = None
    frames_per_first_chunk: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.frames_per_chunk, bool) or not isinstance(self.frames_per_chunk, int):
            raise TypeError("frames_per_chunk must be an integer.")
        if self.frames_per_chunk <= 0:
            raise ValueError("frames_per_chunk must be positive.")
        if not isinstance(self.target_fps, (int, float)) or isinstance(self.target_fps, bool):
            raise TypeError("target_fps must be a number.")
        if not math.isfinite(self.target_fps) or self.target_fps <= 0:
            raise ValueError("target_fps must be a positive finite number.")
        if isinstance(self.buffer_chunks, bool) or not isinstance(self.buffer_chunks, int):
            raise TypeError("buffer_chunks must be an integer.")
        if self.buffer_chunks < 1:
            raise ValueError("buffer_chunks must be at least 1.")
        if self.resident_bytes_per_session is not None and self.resident_bytes_per_session < 0:
            raise ValueError("resident_bytes_per_session must be non-negative.")
        first = self.frames_per_chunk if self.frames_per_first_chunk is None else self.frames_per_first_chunk
        if isinstance(first, bool) or not isinstance(first, int) or first <= 0:
            raise ValueError("frames_per_first_chunk must be a positive integer.")
        object.__setattr__(self, "frames_per_first_chunk", first)

    @property
    def release_period_s(self) -> float:
        """Wall time one steady-state chunk is worth at the declared playout rate.

        Steady state, not the first chunk: a causal decoder expands its first
        latent frame to a single raw frame and every later one to the full
        temporal factor, so the opening chunk delivers less video than those
        that follow. The playout grid is paced by the steady-state chunk, and
        the shorter opening is accounted for in the cumulative video time that
        :func:`chunk_deadlines` walks.
        """
        return self.frames_per_chunk / self.target_fps

    def frames_at(self, chunk_index: int) -> int:
        """Raw frames chunk ``chunk_index`` delivers."""
        if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
            raise ValueError("chunk_index must be a non-negative integer.")
        assert self.frames_per_first_chunk is not None  # resolved in __post_init__
        return self.frames_per_first_chunk if chunk_index == 0 else self.frames_per_chunk

    def cumulative_frames(self, chunk_count: int) -> int:
        """Raw frames delivered by chunks ``0 .. chunk_count - 1``."""
        if chunk_count <= 0:
            return 0
        assert self.frames_per_first_chunk is not None
        return self.frames_per_first_chunk + self.frames_per_chunk * (chunk_count - 1)

    @property
    def is_causal(self) -> bool:
        """Whether the opening chunk delivers fewer frames than later ones."""
        return self.frames_per_first_chunk != self.frames_per_chunk


@dataclass(frozen=True)
class ChunkEvent:
    """One committed chunk, timed against a common monotonic clock."""

    session_id: str
    chunk_index: int
    t_submit: float
    t_ready: float
    frames: int | None = None
    generate_s: float | None = None
    decode_s: float | None = None
    overlap_s: float | None = None
    outstanding_generations: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("session_id must be a non-empty string.")
        if isinstance(self.chunk_index, bool) or not isinstance(self.chunk_index, int) or self.chunk_index < 0:
            raise ValueError("chunk_index must be a non-negative integer.")
        if self.t_ready < self.t_submit:
            raise ValueError("t_ready must not precede t_submit.")
        if self.frames is not None and (isinstance(self.frames, bool) or not isinstance(self.frames, int)):
            raise TypeError("frames must be an integer when provided.")
        if self.frames is not None and self.frames <= 0:
            raise ValueError("frames must be positive when provided.")
        for name in ("generate_s", "decode_s", "overlap_s"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative.")

    @property
    def latency_s(self) -> float:
        return self.t_ready - self.t_submit

    def to_dict(self, *, deadline: float | None = None) -> dict[str, Any]:
        record: dict[str, Any] = {
            "session_id": self.session_id,
            "chunk_index": self.chunk_index,
            "t_submit": self.t_submit,
            "t_ready": self.t_ready,
            "latency_s": self.latency_s,
        }
        for name in ("frames", "generate_s", "decode_s", "overlap_s", "outstanding_generations"):
            value = getattr(self, name)
            if value is not None:
                record[name] = value
        if deadline is not None:
            record["deadline"] = deadline
            record["met_deadline"] = self.t_ready <= deadline
        return record


@dataclass(frozen=True)
class SessionRecord:
    """Every committed chunk of one session, plus when the session started."""

    session_id: str
    t_start: float
    events: tuple[ChunkEvent, ...]
    lost_reason: str | None = None
    resident_decoder_bytes: int | None = None

    def __post_init__(self) -> None:
        indices = [event.chunk_index for event in self.events]
        if indices != sorted(indices) or len(set(indices)) != len(indices):
            raise ValueError("events must be unique and ordered by chunk_index.")
        if any(event.session_id != self.session_id for event in self.events):
            raise ValueError("every event must belong to this session.")


def _optional_sum(values: Iterable[float | None]) -> float | None:
    """Sum values, or ``None`` when nothing reported a value."""
    collected = [value for value in values if value is not None]
    return sum(collected) if collected else None


def percentile(values: Sequence[float], fraction: float) -> float | None:
    """Nearest-rank percentile; ``None`` for an empty sample.

    Nearest-rank rather than an interpolating estimator because these samples
    are small and a reported P99 should be a latency that actually occurred.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1].")
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def chunk_deadlines(record: SessionRecord, profile: WorkloadProfile) -> dict[int, float]:
    """Playout deadline per chunk, or ``{}`` when playback never starts.

    Playback begins once ``buffer_chunks`` chunks are buffered. From that
    instant the player consumes continuously, so chunk ``i`` starts playing
    only after chunks ``0 .. i - 1`` have played, and must be ready by then::

        deadline(i) = t_ready(buffer_chunks - 1) + cumulative_frames(i) / target_fps

    Walking cumulative frames rather than multiplying a constant period is what
    lets a deeper buffer grant real slack, and what keeps a causal decoder's
    shorter opening chunk from being credited with a full period of video.
    Chunks inside the prebuffer have no deadline: their cost is start latency,
    which TTFC already reports.
    """
    if len(record.events) < profile.buffer_chunks:
        return {}
    anchor = record.events[profile.buffer_chunks - 1]
    return {
        event.chunk_index: anchor.t_ready + profile.cumulative_frames(event.chunk_index) / profile.target_fps
        for event in record.events[profile.buffer_chunks :]
    }


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    chunks: int
    frames: int
    ttfc_s: float | None
    latency_p50_s: float | None
    latency_p95_s: float | None
    latency_p99_s: float | None
    latency_max_s: float | None
    rtf: float | None
    deadline_chunks: int
    deadline_misses: int
    continuous_play_ratio: float | None
    lost_reason: str | None
    generate_s_total: float | None = None
    decode_s_total: float | None = None
    decode_share: float | None = None
    overlap_s_total: float | None = None
    overlap_efficiency: float | None = None
    peak_outstanding_generations: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def summarize_session(
    record: SessionRecord,
    profile: WorkloadProfile,
    *,
    mode: LoadMode,
) -> SessionSummary:
    """Reduce one session's chunk events to its reported metrics."""
    events = record.events
    latencies = [event.latency_s for event in events]
    ttfc = events[0].t_ready - record.t_start if events else None
    # Prefer what the decoder actually delivered. Falling back on the profile
    # is an assumption about the decoder's temporal geometry; a measured count
    # is not, which matters precisely because that geometry is not declared
    # anywhere the runtime can read.
    frames = sum(event.frames if event.frames is not None else profile.frames_at(event.chunk_index) for event in events)

    rtf: float | None = None
    if events:
        # Steady-state generation cost against the video time produced. Measured
        # from the first submit so queueing before the session starts ticking is
        # not charged to the model.
        wall = events[-1].t_ready - events[0].t_submit
        video_seconds = frames / profile.target_fps
        rtf = wall / video_seconds if video_seconds > 0 else None

    deadlines = chunk_deadlines(record, profile) if mode is LoadMode.PACED else {}
    misses = sum(
        1 for event in events if event.chunk_index in deadlines and event.t_ready > deadlines[event.chunk_index]
    )
    cpr = (len(deadlines) - misses) / len(deadlines) if deadlines else None

    generate_total = _optional_sum(event.generate_s for event in events)
    decode_total = _optional_sum(event.decode_s for event in events)
    overlap_total = _optional_sum(event.overlap_s for event in events)
    outstanding = [e.outstanding_generations for e in events if e.outstanding_generations is not None]
    latency_total = sum(latencies)
    return SessionSummary(
        session_id=record.session_id,
        chunks=len(events),
        frames=frames,
        ttfc_s=ttfc,
        latency_p50_s=percentile(latencies, 0.50),
        latency_p95_s=percentile(latencies, 0.95),
        latency_p99_s=percentile(latencies, 0.99),
        latency_max_s=max(latencies) if latencies else None,
        rtf=rtf,
        deadline_chunks=len(deadlines),
        deadline_misses=misses,
        continuous_play_ratio=cpr,
        lost_reason=record.lost_reason,
        generate_s_total=generate_total,
        decode_s_total=decode_total,
        # The fraction of chunk wall time spent decoding. With generation and
        # decode serialized this is the headroom overlapping them can recover;
        # once they overlap it is what the measurement has to show shrinking.
        decode_share=(decode_total / latency_total if decode_total is not None and latency_total > 0 else None),
        overlap_s_total=overlap_total,
        # Fraction of generate time hidden behind decode. 0 when the two are
        # serialized, approaching 1 as generation is fully covered.
        overlap_efficiency=(
            overlap_total / generate_total
            if overlap_total is not None and generate_total is not None and generate_total > 0
            else None
        ),
        peak_outstanding_generations=max(outstanding) if outstanding else None,
    )


@dataclass(frozen=True)
class RunSummary:
    """Aggregate view of one benchmark run, plus the per-session detail."""

    mode: LoadMode
    profile: WorkloadProfile
    num_gpus: int
    wall_s: float
    sessions: tuple[SessionSummary, ...]
    sessions_lost: int
    generated_fps: float | None
    frames_per_gpu_second: float | None
    worst_case_chunk_latency_s: float | None
    continuous_play_ratio: float | None
    mean_chunk_latency_s: float | None
    rtf_spread: float | None
    peak_concurrent_sessions: int
    decode_share: float | None = None
    overlap_efficiency: float | None = None
    resident_decoder_bytes_per_session: int | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "profile": {
                "frames_per_chunk": self.profile.frames_per_chunk,
                "frames_per_first_chunk": self.profile.frames_per_first_chunk,
                "target_fps": self.profile.target_fps,
                "buffer_chunks": self.profile.buffer_chunks,
                "release_period_s": self.profile.release_period_s,
                "resident_bytes_per_session": self.profile.resident_bytes_per_session,
            },
            "num_gpus": self.num_gpus,
            "wall_s": self.wall_s,
            "sessions_run": len(self.sessions),
            "sessions_lost": self.sessions_lost,
            "peak_concurrent_sessions": self.peak_concurrent_sessions,
            "generated_fps": self.generated_fps,
            "frames_per_gpu_second": self.frames_per_gpu_second,
            "mean_chunk_latency_s": self.mean_chunk_latency_s,
            "worst_case_chunk_latency_s": self.worst_case_chunk_latency_s,
            "continuous_play_ratio": self.continuous_play_ratio,
            "rtf_spread": self.rtf_spread,
            "decode_share": self.decode_share,
            "overlap_efficiency": self.overlap_efficiency,
            "resident_decoder_bytes_per_session": self.resident_decoder_bytes_per_session,
            "notes": list(self.notes),
            "per_session": [summary.to_dict() for summary in self.sessions],
        }


def summarize_run(
    records: Iterable[SessionRecord],
    profile: WorkloadProfile,
    *,
    mode: LoadMode,
    wall_s: float,
    num_gpus: int = 1,
    peak_concurrent_sessions: int | None = None,
    resident_decoder_bytes_per_session: int | None = None,
    notes: Sequence[str] = (),
) -> RunSummary:
    """Aggregate session records into the reportable run summary."""
    if num_gpus < 1:
        raise ValueError("num_gpus must be at least 1.")
    if wall_s < 0:
        raise ValueError("wall_s must be non-negative.")

    ordered = list(records)
    summaries = tuple(summarize_session(record, profile, mode=mode) for record in ordered)
    total_frames = sum(summary.frames for summary in summaries)
    latencies = [event.latency_s for record in ordered for event in record.events]

    # Macro-average over sessions: one lagging session is not diluted by a
    # neighbour that produced many more chunks.
    per_session_cpr = [s.continuous_play_ratio for s in summaries if s.continuous_play_ratio is not None]
    rtfs = [s.rtf for s in summaries if s.rtf is not None]
    decode_shares = [s.decode_share for s in summaries if s.decode_share is not None]
    overlaps = [s.overlap_efficiency for s in summaries if s.overlap_efficiency is not None]

    return RunSummary(
        mode=mode,
        profile=profile,
        num_gpus=num_gpus,
        wall_s=wall_s,
        sessions=summaries,
        sessions_lost=sum(1 for s in summaries if s.lost_reason is not None),
        generated_fps=total_frames / wall_s if wall_s > 0 else None,
        frames_per_gpu_second=total_frames / (wall_s * num_gpus) if wall_s > 0 else None,
        worst_case_chunk_latency_s=max(latencies) if latencies else None,
        continuous_play_ratio=statistics.fmean(per_session_cpr) if per_session_cpr else None,
        mean_chunk_latency_s=statistics.fmean(latencies) if latencies else None,
        rtf_spread=max(rtfs) - min(rtfs) if len(rtfs) > 1 else None,
        peak_concurrent_sessions=(peak_concurrent_sessions if peak_concurrent_sessions is not None else len(summaries)),
        decode_share=statistics.fmean(decode_shares) if decode_shares else None,
        overlap_efficiency=statistics.fmean(overlaps) if overlaps else None,
        resident_decoder_bytes_per_session=resident_decoder_bytes_per_session,
        notes=tuple(notes),
    )


def compare_runs(baseline: RunSummary, candidate: RunSummary) -> dict[str, Any]:
    """Derive the per-tick switching cost between two runs.

    Aggregate generated FPS is expected to stay flat when concurrency of state
    rises but concurrency of execution does not: ticks serialize, so the same
    device produces the same frames. A measurable drop is per-tick switching
    cost -- state bind/unbind, KV pool paging, conditioning rebuild -- and that
    quantity is the baseline any cross-session batching work has to beat.
    """
    delta: dict[str, Any] = {
        "baseline_sessions": len(baseline.sessions),
        "candidate_sessions": len(candidate.sessions),
    }
    if baseline.generated_fps and candidate.generated_fps is not None:
        delta["generated_fps_ratio"] = candidate.generated_fps / baseline.generated_fps
    if baseline.mean_chunk_latency_s is not None and candidate.mean_chunk_latency_s is not None:
        delta["mean_chunk_latency_delta_s"] = candidate.mean_chunk_latency_s - baseline.mean_chunk_latency_s
        sessions = len(candidate.sessions) or 1
        # With serialized ticks a candidate chunk waits behind the other
        # resident sessions, so the expected latency is sessions x baseline.
        # Anything beyond that expectation is switching cost, not queueing.
        expected = baseline.mean_chunk_latency_s * sessions
        delta["per_tick_switching_cost_s"] = candidate.mean_chunk_latency_s - expected
    return delta


def load_profile_from_spec(
    spec: Mapping[str, Any],
    *,
    target_fps: float,
    buffer_chunks: int = 1,
    causal: bool = True,
) -> WorkloadProfile:
    """Build a profile from a pipeline's declared AR-Diffusion KV spec.

    ``frames_per_block`` is in latent frames. A causal video decoder expands
    the very first latent frame of a session to one raw frame and every later
    one to the full temporal factor, so an opening block of ``n`` latent frames
    delivers ``(n - 1) * factor + 1`` raw frames while every later block
    delivers ``n * factor``. Pass ``causal=False`` for a decoder that expands
    every latent frame identically.
    """
    frames_per_block = spec.get("frames_per_block")
    temporal_factor = spec.get("vae_temporal_factor", 1)
    if isinstance(frames_per_block, bool) or not isinstance(frames_per_block, int) or frames_per_block <= 0:
        raise ValueError("spec must declare a positive integer frames_per_block.")
    if isinstance(temporal_factor, bool) or not isinstance(temporal_factor, int) or temporal_factor <= 0:
        raise ValueError("spec vae_temporal_factor must be a positive integer.")
    steady = frames_per_block * temporal_factor
    first = (frames_per_block - 1) * temporal_factor + 1 if causal else steady
    return WorkloadProfile(
        frames_per_chunk=steady,
        frames_per_first_chunk=first,
        target_fps=target_fps,
        buffer_chunks=buffer_chunks,
        resident_bytes_per_session=spec.get("resident_bytes_per_session"),
    )
