# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Workload construction and metric math for the LingBot-World realtime benchmark.

Nothing here imports torch, vllm, or websockets, so the arithmetic that decides
what a benchmark number *means* is unit-testable without a GPU or a server.

Two model properties drive every formula in this module, both read from the
checkpoint's ``transformer/config.json`` rather than assumed:

``num_frames_per_block = 3``
    One AR block commits three latent frames, so one streamed chunk is three
    latent frames and a request's ``camera_action_script`` carries exactly one
    three-entry action list per chunk.

``sliding_window_num_frames = 18`` with ``sink_size = 9``
    Self-attention sees nine sink frames plus a nine-frame recent window, so the
    key count a block attends over keeps growing until eighteen latent frames of
    history exist -- six chunks. Chunk cost is therefore *not* stationary before
    chunk six, and a mean taken over a short rollout reports a number no steady
    session will ever produce. :data:`DEFAULT_WARMUP_CHUNKS` exists for that
    reason and is not a round-number guess.

The causal VAE expands the first latent frame to one pixel frame and every later
latent frame to four, so chunk zero carries nine pixel frames and every chunk
after it carries twelve. Video-duration math has to respect that asymmetry or
real-time factor comes out wrong on short rollouts.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# transformer.config.num_frames_per_block: latent frames committed per AR block.
FRAMES_PER_BLOCK = 3
# vae_scale_factor_temporal: latent frames -> pixel frames, after the first.
TEMPORAL_COMPRESSION = 4
# (sink_size 9 + recent window 9) / FRAMES_PER_BLOCK: chunks until the attention
# window saturates and per-chunk cost becomes stationary.
DEFAULT_WARMUP_CHUNKS = 6

DEFAULT_MODEL = "robbyant/lingbot-world-v2-14b-causal-fast-diffusers"
DEFAULT_PROMPT = "The camera moves slowly forward through the scene."
# The AR KV cache geometry is fixed at load time, so a request must ask for
# exactly the resolution the deploy config declares.
DEFAULT_WIDTH = 832
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 16
DEFAULT_FLOW_SHIFT = 5.0
DEFAULT_SEED = 42

_HOLD = "hold"
_ACTION_KEYS = ("w", "a", "s", "d")


def num_frames_for_chunks(num_chunks: int) -> int:
    """Pixel-frame count whose rollout is exactly ``num_chunks`` chunks.

    Inverse of :func:`chunks_for_num_frames`, which is the server's own
    ``((num_frames - 1) // 4 + 1) // 3``.
    """
    if num_chunks < 1:
        raise ValueError("num_chunks must be at least 1.")
    return (num_chunks * FRAMES_PER_BLOCK - 1) * TEMPORAL_COMPRESSION + 1


def chunks_for_num_frames(num_frames: int) -> int:
    """Chunks the server generates for ``num_frames``, mirroring its own formula."""
    if num_frames < 1:
        raise ValueError("num_frames must be at least 1.")
    return ((num_frames - 1) // TEMPORAL_COMPRESSION + 1) // FRAMES_PER_BLOCK


def pixel_frames_in_chunk(chunk_index: int) -> int:
    """Pixel frames carried by one chunk.

    The causal decoder expands the session's opening latent frame once and every
    later latent frame four times, so chunk zero is short by three frames.
    """
    if chunk_index < 0:
        raise ValueError("chunk_index must be non-negative.")
    if chunk_index == 0:
        return (FRAMES_PER_BLOCK - 1) * TEMPORAL_COMPRESSION + 1
    return FRAMES_PER_BLOCK * TEMPORAL_COMPRESSION


def camera_actions_for_chunk(pattern: str, chunk_index: int) -> list[list[str]]:
    """One three-entry action list for ``chunk_index``.

    Camera actions select camera embeddings; they change what the world does,
    not how much arithmetic it costs. The pattern therefore exists to keep a
    benchmark rollout representative rather than to sweep a cost dimension.
    """
    if chunk_index < 0:
        raise ValueError("chunk_index must be non-negative.")
    if pattern == "forward":
        return [["w"] for _ in range(FRAMES_PER_BLOCK)]
    if pattern == _HOLD:
        return [[] for _ in range(FRAMES_PER_BLOCK)]
    if pattern == "orbit":
        key = _ACTION_KEYS[chunk_index % len(_ACTION_KEYS)]
        return [[key] for _ in range(FRAMES_PER_BLOCK)]
    raise ValueError(f"Unknown camera pattern {pattern!r}; expected one of {sorted(CAMERA_PATTERNS)}.")


CAMERA_PATTERNS = ("forward", "orbit", _HOLD)


def build_camera_script(num_chunks: int, pattern: str) -> list[list[list[str]]]:
    """Full ``camera_action_script``: one action list per generated chunk."""
    if num_chunks < 1:
        raise ValueError("num_chunks must be at least 1.")
    return [camera_actions_for_chunk(pattern, index) for index in range(num_chunks)]


def _validate_camera_script(script: Sequence[Any]) -> list[list[list[str]]]:
    validated: list[list[list[str]]] = []
    for chunk_index, actions in enumerate(script):
        if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes)):
            raise ValueError(f"camera_action_script[{chunk_index}] must be a list of three action lists.")
        if len(actions) != FRAMES_PER_BLOCK:
            raise ValueError(
                f"camera_action_script[{chunk_index}] must hold exactly {FRAMES_PER_BLOCK} action lists "
                f"(got {len(actions)}); one AR block is {FRAMES_PER_BLOCK} latent frames."
            )
        frames: list[list[str]] = []
        for frame_index, frame in enumerate(actions):
            if not isinstance(frame, Sequence) or isinstance(frame, (str, bytes)):
                raise ValueError(f"camera_action_script[{chunk_index}][{frame_index}] must be a list of strings.")
            if any(not isinstance(action, str) for action in frame):
                raise ValueError(f"camera_action_script[{chunk_index}][{frame_index}] must contain only strings.")
            frames.append([str(action) for action in frame])
        validated.append(frames)
    if not validated:
        raise ValueError("camera_action_script must not be empty.")
    return validated


@dataclass(frozen=True)
class PromptUpdate:
    """A mid-rollout ``session.interaction`` scheduled on a chunk boundary."""

    after_chunk: int
    prompt: str
    transition_chunks: int = 3

    def __post_init__(self) -> None:
        if self.after_chunk < 0:
            raise ValueError("PromptUpdate.after_chunk must be non-negative.")
        if not self.prompt.strip():
            raise ValueError("PromptUpdate.prompt must contain non-whitespace text.")
        if self.transition_chunks < 0:
            raise ValueError("PromptUpdate.transition_chunks must be non-negative.")

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": "session.interaction",
            "interaction": {
                "event_id": f"chunk-{self.after_chunk}",
                "event": {"prompt": self.prompt},
                "transition_chunks": self.transition_chunks,
            },
        }


@dataclass(frozen=True)
class Workload:
    """One rollout: what the benchmark asks the server to generate."""

    prompt: str
    image_reference: str
    camera_script: list[list[list[str]]]
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    fps: int = DEFAULT_FPS
    seed: int = DEFAULT_SEED
    flow_shift: float = DEFAULT_FLOW_SHIFT
    negative_prompt: str | None = None
    prompt_updates: tuple[PromptUpdate, ...] = ()
    extra_params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("Workload.prompt must contain non-whitespace text.")
        if not self.image_reference:
            raise ValueError("Workload.image_reference is required; LingBot-World is image-conditioned.")
        if self.width <= 0 or self.height <= 0 or self.width % 16 or self.height % 16:
            raise ValueError("Workload width/height must be positive multiples of 16.")
        if self.fps <= 0:
            raise ValueError("Workload.fps must be positive.")
        object.__setattr__(self, "camera_script", _validate_camera_script(self.camera_script))
        seen_update_chunks: set[int] = set()
        for update in self.prompt_updates:
            if update.after_chunk in seen_update_chunks:
                raise ValueError(f"Duplicate prompt update after chunk {update.after_chunk}; event IDs must be unique.")
            seen_update_chunks.add(update.after_chunk)
            if update.after_chunk >= self.num_chunks:
                raise ValueError(
                    f"Prompt update after chunk {update.after_chunk} never fires in a {self.num_chunks}-chunk rollout."
                )

    @property
    def num_chunks(self) -> int:
        return len(self.camera_script)

    @property
    def num_frames(self) -> int:
        return num_frames_for_chunks(self.num_chunks)

    @property
    def video_seconds(self) -> float:
        return self.num_frames / self.fps

    def session_start_payload(self, model: str) -> dict[str, Any]:
        """The ``session.start`` event for ``WS /v1/realtime/video``."""
        extra_params: dict[str, Any] = {
            "flow_shift": self.flow_shift,
            "camera_action_script": self.camera_script,
        }
        extra_params.update(self.extra_params)
        payload: dict[str, Any] = {
            "type": "session.start",
            "model": model,
            "prompt": self.prompt,
            "image_reference": {"image_url": self.image_reference},
            "width": self.width,
            "height": self.height,
            "num_frames": self.num_frames,
            "fps": self.fps,
            "seed": self.seed,
            "extra_params": extra_params,
        }
        if self.negative_prompt is not None:
            payload["negative_prompt"] = self.negative_prompt
        return payload

    def describe(self) -> dict[str, Any]:
        return {
            "num_chunks": self.num_chunks,
            "num_frames": self.num_frames,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "seed": self.seed,
            "flow_shift": self.flow_shift,
            "video_seconds": self.video_seconds,
            "prompt_updates": [
                {
                    "after_chunk": update.after_chunk,
                    "transition_chunks": update.transition_chunks,
                }
                for update in self.prompt_updates
            ],
        }


def default_workload(
    *,
    image_reference: str,
    num_chunks: int,
    camera_pattern: str = "forward",
    prompt: str = DEFAULT_PROMPT,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: int = DEFAULT_FPS,
    seed: int = DEFAULT_SEED,
    flow_shift: float = DEFAULT_FLOW_SHIFT,
    negative_prompt: str | None = None,
) -> Workload:
    """Built-in rollout: a fixed prompt and a generated camera script."""
    return Workload(
        prompt=prompt,
        image_reference=image_reference,
        camera_script=build_camera_script(num_chunks, camera_pattern),
        width=width,
        height=height,
        fps=fps,
        seed=seed,
        flow_shift=flow_shift,
        negative_prompt=negative_prompt,
    )


def load_workload(
    path: str | Path,
    *,
    image_resolver: Any = None,
    overrides: Mapping[str, Any] | None = None,
) -> Workload:
    """Read a rollout spec from JSON.

    ``image_resolver`` turns the spec's ``image`` field into an ``http(s)`` or
    ``data:`` URL; the caller owns that because inlining a local file is I/O.
    Keys in ``overrides`` that are not ``None`` win over the file, so CLI flags
    stay meaningful next to ``--workload``.
    """
    document = json.loads(Path(path).read_text())
    if not isinstance(document, dict):
        raise ValueError("A workload file must contain a JSON object.")

    merged = dict(document)
    for key, value in (overrides or {}).items():
        if value is not None:
            merged[key] = value

    camera_script = merged.get("camera_action_script")
    if camera_script is None:
        num_chunks = merged.get("num_chunks")
        if num_chunks is None:
            raise ValueError("A workload file needs either camera_action_script or num_chunks.")
        camera_script = build_camera_script(int(num_chunks), str(merged.get("camera_pattern", "forward")))

    image = merged.get("image") or merged.get("image_reference")
    if image is None:
        raise ValueError("A workload file needs an image or image_reference field.")
    image_reference = image_resolver(str(image)) if image_resolver is not None else str(image)

    updates_field = merged.get("prompt_updates") or ()
    prompt_updates = tuple(
        PromptUpdate(
            after_chunk=int(entry["after_chunk"]),
            prompt=str(entry["prompt"]),
            transition_chunks=int(entry.get("transition_chunks", 3)),
        )
        for entry in updates_field
    )

    extra_params = merged.get("extra_params") or {}
    if not isinstance(extra_params, Mapping):
        raise ValueError("Workload extra_params must be a JSON object.")

    return Workload(
        prompt=str(merged.get("prompt", DEFAULT_PROMPT)),
        image_reference=image_reference,
        camera_script=camera_script,
        width=int(merged.get("width", DEFAULT_WIDTH)),
        height=int(merged.get("height", DEFAULT_HEIGHT)),
        fps=int(merged.get("fps", DEFAULT_FPS)),
        seed=int(merged.get("seed", DEFAULT_SEED)),
        flow_shift=float(merged.get("flow_shift", DEFAULT_FLOW_SHIFT)),
        negative_prompt=merged.get("negative_prompt"),
        prompt_updates=prompt_updates,
        extra_params=dict(extra_params),
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


@dataclass
class ChunkRecord:
    """One media chunk as the client observed it."""

    index: int
    arrival_s: float
    """Seconds from ``session.start`` being sent to this chunk's bytes arriving."""
    inter_arrival_s: float
    """Seconds since the previous chunk; for chunk zero this equals ``arrival_s``."""
    byte_length: int
    num_frames: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "arrival_s": self.arrival_s,
            "inter_arrival_ms": self.inter_arrival_s * 1000.0,
            "byte_length": self.byte_length,
            "num_frames": self.num_frames,
        }


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linearly interpolated percentile, so a 10-sample run still reports a p95.

    Implemented here rather than pulled from numpy to keep this module importable
    in a bare environment; the benchmark's own numbers must not depend on which
    optional package happens to be installed.
    """
    if not values:
        raise ValueError("percentile() needs at least one value.")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("percentile fraction must be within [0, 1].")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stddev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    """Millisecond summary of a set of inter-arrival intervals."""
    if not values:
        return {}
    milliseconds = [value * 1000.0 for value in values]
    return {
        "count": float(len(milliseconds)),
        "mean_ms": sum(milliseconds) / len(milliseconds),
        "median_ms": percentile(milliseconds, 0.5),
        "std_ms": _stddev(milliseconds),
        "min_ms": min(milliseconds),
        "p90_ms": percentile(milliseconds, 0.90),
        "p95_ms": percentile(milliseconds, 0.95),
        "p99_ms": percentile(milliseconds, 0.99),
        "max_ms": max(milliseconds),
    }


@dataclass(frozen=True)
class PlaybackResult:
    """What a viewer would have seen, given a start buffer and a playback rate."""

    underrun_count: int
    total_stall_s: float
    playback_span_s: float
    first_underrun_chunk: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "underrun_count": self.underrun_count,
            "total_stall_ms": self.total_stall_s * 1000.0,
            "playback_span_s": self.playback_span_s,
            "stall_ratio": (self.total_stall_s / self.playback_span_s) if self.playback_span_s > 0 else 0.0,
            "first_underrun_chunk": self.first_underrun_chunk,
        }


def simulate_playback(
    records: Sequence[ChunkRecord],
    *,
    fps: float,
    buffer_chunks: int = 1,
) -> PlaybackResult:
    """Replay arrivals against a wall clock and count stalls.

    Playback starts when ``buffer_chunks`` chunks have arrived and then consumes
    video at ``fps``. A chunk that has not arrived by the moment the player needs
    it is an underrun, and the player resumes only when the bytes land -- which
    is what a viewer experiences, and what a single p99 number cannot tell you:
    the same p99 is harmless behind a three-chunk buffer and fatal behind none.
    """
    if fps <= 0:
        raise ValueError("fps must be positive.")
    if buffer_chunks < 1:
        raise ValueError("buffer_chunks must be at least 1.")
    if not records:
        return PlaybackResult(0, 0.0, 0.0, None)
    if buffer_chunks > len(records):
        raise ValueError(f"buffer_chunks={buffer_chunks} exceeds the {len(records)} chunks received.")

    ordered = sorted(records, key=lambda record: record.index)
    start = ordered[buffer_chunks - 1].arrival_s
    cursor = start
    underruns = 0
    total_stall = 0.0
    first_underrun: int | None = None

    for record in ordered:
        if record.arrival_s > cursor:
            stall = record.arrival_s - cursor
            underruns += 1
            total_stall += stall
            if first_underrun is None:
                first_underrun = record.index
            cursor = record.arrival_s
        cursor += record.num_frames / fps

    return PlaybackResult(
        underrun_count=underruns,
        total_stall_s=total_stall,
        playback_span_s=cursor - start,
        first_underrun_chunk=first_underrun,
    )


def steady_chunk_deadline_ms(fps: float) -> float:
    """Wall-clock budget a steady chunk must meet to hold real time.

    ``fps`` here is the rate the frames are *intended to be played at*, which the
    checkpoint does not declare: nothing in ``model_index.json``, the scheduler
    config, or the transformer config carries a frame rate, and the pipeline
    never reads one. It is a property of how the output is consumed, so both
    this deadline and real-time factor scale linearly with the choice and the
    caller has to state it rather than inherit it from a muxing default.
    """
    if fps <= 0:
        raise ValueError("fps must be positive.")
    return pixel_frames_in_chunk(1) / fps * 1000.0


def compute_metrics(
    records: Sequence[ChunkRecord],
    *,
    fps: float,
    warmup_chunks: int = DEFAULT_WARMUP_CHUNKS,
    slo_ms: float | None = None,
    playback_buffer_chunks: int = 1,
    ttfc_s: float | None = None,
    session_wall_s: float | None = None,
) -> dict[str, Any]:
    """Summarize one session.

    ``warmup_chunks`` are excluded from the steady-state block because the
    attention window has not saturated before then (see the module docstring).
    The terminal chunk also stays out of steady metrics because it skips
    next-chunk preparation. All chunks stay in the playback simulation, which is
    what a session really costs end to end.
    """
    if not records:
        raise ValueError("compute_metrics() needs at least one chunk record.")
    if warmup_chunks < 0:
        raise ValueError("warmup_chunks must be non-negative.")

    ordered = sorted(records, key=lambda record: record.index)
    deadline_ms = slo_ms if slo_ms is not None else steady_chunk_deadline_ms(fps)

    all_intervals = [record.inter_arrival_s for record in ordered[1:]]
    # The terminal chunk skips next-chunk preparation and is not representative
    # of sustained generation. Keep it in end-to-end and playback metrics.
    steady = [record for record in ordered[:-1] if record.index >= warmup_chunks]
    steady_intervals = [record.inter_arrival_s for record in steady if record.index > 0]

    total_frames = sum(record.num_frames for record in ordered)
    video_seconds = total_frames / fps
    wall_seconds = session_wall_s if session_wall_s is not None else ordered[-1].arrival_s

    metrics: dict[str, Any] = {
        "chunks_received": len(ordered),
        "warmup_chunks": warmup_chunks,
        "steady_chunks": len(steady_intervals),
        "steady_state_reached": bool(steady_intervals),
        "total_frames": total_frames,
        "total_bytes": sum(record.byte_length for record in ordered),
        "video_seconds": video_seconds,
        "wall_seconds": wall_seconds,
        # RTF is processing time over content duration, as vLLM-Omni computes it
        # (`patch.py`: generation_s / video_duration; `compute_audio_rtf`: latency
        # / audio_duration, "SLO red line < 1"). LOWER IS BETTER and below 1.0 is
        # real time. RTFX is the reciprocal the repo reports under its own name,
        # kept here because "fraction of real time achieved" reads better on a
        # dashboard -- but the two must never be confused for one another.
        "video_rtf": (wall_seconds / video_seconds) if video_seconds > 0 else 0.0,
        "video_rtfx": (video_seconds / wall_seconds) if wall_seconds > 0 else 0.0,
        "frames_per_second": (total_frames / wall_seconds) if wall_seconds > 0 else 0.0,
        "chunk_deadline_ms": deadline_ms,
        "interval_all": _latency_summary(all_intervals),
        "interval_steady": _latency_summary(steady_intervals),
    }
    if ttfc_s is not None:
        metrics["ttfc_ms"] = ttfc_s * 1000.0

    if steady_intervals:
        steady_frames = sum(record.num_frames for record in steady if record.index > 0)
        steady_wall = sum(steady_intervals)
        steady_video = steady_frames / fps
        metrics["steady_video_rtf"] = (steady_wall / steady_video) if steady_video > 0 else 0.0
        metrics["steady_video_rtfx"] = (steady_video / steady_wall) if steady_wall > 0 else 0.0
        within = sum(1 for value in steady_intervals if value * 1000.0 <= deadline_ms)
        metrics["slo_attainment"] = within / len(steady_intervals)
        metrics["slo_violations"] = len(steady_intervals) - within

    metrics["playback"] = simulate_playback(
        ordered,
        fps=fps,
        buffer_chunks=playback_buffer_chunks,
    ).to_dict()
    metrics["playback"]["buffer_chunks"] = playback_buffer_chunks
    return metrics


def aggregate_metrics(sessions: Iterable[Mapping[str, Any]], *, fps: float) -> dict[str, Any]:
    """Combine per-session metrics across sequential repeats.

    Steady intervals are pooled rather than averaging per-session means, so a
    session that happened to produce fewer steady chunks does not carry the same
    weight as a full one.
    """
    entries = list(sessions)
    if not entries:
        raise ValueError("aggregate_metrics() needs at least one session.")
    if len(entries) == 1:
        single = dict(entries[0])
        single["sessions"] = 1
        return single

    deadline_ms = entries[0]["chunk_deadline_ms"]
    if any(entry["chunk_deadline_ms"] != deadline_ms for entry in entries):
        raise ValueError("Cannot aggregate sessions with different chunk deadlines.")
    ttfcs = [float(entry["ttfc_ms"]) for entry in entries if entry.get("ttfc_ms") is not None]
    # Lower is better, so "max" is the worst session, not the best.
    rtfs = [float(entry["video_rtf"]) for entry in entries]
    steady_means = [
        float(entry["interval_steady"]["mean_ms"]) for entry in entries if entry.get("interval_steady", {}).get("count")
    ]
    steady_count = sum(entry.get("interval_steady", {}).get("count", 0) for entry in entries)
    steady_total_ms = sum(
        entry["interval_steady"]["mean_ms"] * entry["interval_steady"]["count"]
        for entry in entries
        if entry.get("interval_steady", {}).get("count")
    )
    return {
        "sessions": len(entries),
        "chunks_received": sum(int(entry["chunks_received"]) for entry in entries),
        "total_frames": sum(int(entry["total_frames"]) for entry in entries),
        "video_seconds": sum(float(entry["video_seconds"]) for entry in entries),
        "wall_seconds": sum(float(entry["wall_seconds"]) for entry in entries),
        "video_rtf_mean": sum(rtfs) / len(rtfs),
        "video_rtf_best": min(rtfs),
        "video_rtf_worst": max(rtfs),
        "ttfc_ms_mean": (sum(ttfcs) / len(ttfcs)) if ttfcs else None,
        "ttfc_ms_max": max(ttfcs) if ttfcs else None,
        "steady_interval_ms_mean": (steady_total_ms / steady_count) if steady_count else None,
        "steady_interval_ms_spread": (max(steady_means) - min(steady_means)) if steady_means else None,
        "underrun_count": sum(int(entry["playback"]["underrun_count"]) for entry in entries),
        "chunk_deadline_ms": deadline_ms,
    }
