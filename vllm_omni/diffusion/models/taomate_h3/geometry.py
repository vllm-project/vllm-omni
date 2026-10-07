# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""TaoMate-H3 streaming geometry.

TaoMate-H3 generates one five-second MiniMax-H3 T2VA request at a time and
splits it into four causal phases of (2, 2, 2, 1) native 17-frame groups. The
first request additionally owns the five-frame / two-latent affine prefix of a
standalone H3 request; later requests ("canonical continuations") generate only
the steady 119 frames and inherit the previous request's tail as a transport
prefix. Audio latents run at 40 Hz and their phase boundaries are rounded on
the global frame timeline, so a continuation request has 198 or 199 audio
latents per channel rather than a request-local 207.

Everything in this module is host-side integer bookkeeping shared by the
denoise loop, the audio teacher and the streaming decoder. It mirrors the
validated TaoMate-H3 direct-streaming contract and must not drift from it: the
LoRA was distilled for exactly this phase geometry and RoPE timeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

VIDEO_FPS = 24
AUDIO_LATENT_RATE = 40
AUDIO_SAMPLE_RATE = 32000
AUDIO_SAMPLES_PER_LATENT = AUDIO_SAMPLE_RATE // AUDIO_LATENT_RATE  # 800
VIDEO_PREFIX_FRAMES = 5
VIDEO_PREFIX_LATENTS = 2
VIDEO_GROUP_FRAMES = 17
VIDEO_GROUP_LATENTS = 5
REQUEST_SECONDS = 5
PHASE_GROUP_COUNTS = (2, 2, 2, 1)
# Official standalone request geometry (37 video latents, 207 audio latents).
REQUEST_VIDEO_LATENTS = 37
REQUEST_AUDIO_LATENTS = 207
REQUEST_NATIVE_FRAMES = 124
STEADY_NATIVE_FRAMES = REQUEST_NATIVE_FRAMES - VIDEO_PREFIX_FRAMES  # 119
SUPPORTED_SHORT_EDGES = (480, 768, 1088)
CANVAS_MULTIPLE = 32
# RoPE temporal spacing of H3 video latents: the first latent of every group of
# five spans one native frame, the other four span four frames each, all
# rescaled by 5/3 so that one 40 Hz audio latent is one RoPE unit.
_TEMPORAL_WEIGHTS = (1, 4, 4, 4, 4)
_FRAME_RESCALE = Fraction(5, 3)


def _round_half_even(value: Fraction) -> int:
    quotient, remainder = divmod(value.numerator, value.denominator)
    doubled = remainder * 2
    if doubled < value.denominator:
        return quotient
    if doubled > value.denominator:
        return quotient + 1
    return quotient + (quotient & 1)


def audio_latent_boundary(frame: int) -> int:
    """40 Hz audio latent index at a 24 fps frame boundary (round half even)."""
    return _round_half_even(Fraction(frame * AUDIO_LATENT_RATE, VIDEO_FPS))


def video_temporal_position(latent_index: int) -> Fraction:
    """Absolute H3 RoPE time coordinate of a video latent before the text origin offset."""
    if isinstance(latent_index, bool) or not isinstance(latent_index, int):
        raise TypeError("latent_index must be an integer")
    if latent_index < 0:
        raise ValueError("latent_index must be non-negative")
    return sum(
        (_FRAME_RESCALE * _TEMPORAL_WEIGHTS[index % len(_TEMPORAL_WEIGHTS)] for index in range(latent_index)),
        start=Fraction(0),
    )


def video_temporal_positions(start: int, count: int) -> tuple[Fraction, ...]:
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    return tuple(video_temporal_position(index) for index in range(start, start + count))


@dataclass(frozen=True)
class StreamPhase:
    """One causal chunk of a request: a contiguous frame / latent span."""

    index: int
    group_count: int
    frame_start: int
    frame_stop: int
    video_latent_start: int
    video_latent_stop: int
    audio_latent_start: int
    audio_latent_stop: int

    @property
    def frame_count(self) -> int:
        return self.frame_stop - self.frame_start

    @property
    def video_latent_count(self) -> int:
        return self.video_latent_stop - self.video_latent_start

    @property
    def audio_latent_count(self) -> int:
        return self.audio_latent_stop - self.audio_latent_start


@dataclass(frozen=True)
class StreamPlan:
    """The phases of one request, in request-local frame / latent coordinates."""

    native_frame_count: int
    phases: tuple[StreamPhase, ...]

    @property
    def video_latent_count(self) -> int:
        return self.phases[-1].video_latent_stop

    @property
    def audio_latent_count(self) -> int:
        return self.phases[-1].audio_latent_stop


def direct_5s_plan() -> StreamPlan:
    """The four-phase plan of the first (standalone) five-second request."""
    phases: list[StreamPhase] = []
    frame_stop = 0
    video_stop = 0
    for index, groups in enumerate(PHASE_GROUP_COUNTS):
        frame_start = frame_stop
        video_start = video_stop
        frame_stop += groups * VIDEO_GROUP_FRAMES
        video_stop += groups * VIDEO_GROUP_LATENTS
        if index == 0:
            frame_stop += VIDEO_PREFIX_FRAMES
            video_stop += VIDEO_PREFIX_LATENTS
        phases.append(
            StreamPhase(
                index=index,
                group_count=groups,
                frame_start=frame_start,
                frame_stop=frame_stop,
                video_latent_start=video_start,
                video_latent_stop=video_stop,
                audio_latent_start=audio_latent_boundary(frame_start),
                audio_latent_stop=audio_latent_boundary(frame_stop),
            )
        )
    plan = StreamPlan(native_frame_count=REQUEST_NATIVE_FRAMES, phases=tuple(phases))
    if plan.video_latent_count != REQUEST_VIDEO_LATENTS or plan.audio_latent_count != REQUEST_AUDIO_LATENTS:
        raise RuntimeError("TaoMate-H3 direct plan does not match the official H3 request geometry")
    return plan


def canonical_continuation_plan(base_plan: StreamPlan, *, request_index: int) -> StreamPlan:
    """The steady-state plan of request ``request_index > 0``.

    A continuation advances seven native 17-frame groups (35 video latents)
    without regenerating the affine prefix. Audio boundaries are rounded on the
    global frame timeline, so the per-request audio count is 198 or 199.
    """
    if isinstance(request_index, bool) or not isinstance(request_index, int):
        raise TypeError("request_index must be an integer")
    if request_index <= 0:
        raise ValueError("canonical continuation requires request_index > 0")
    steady_frame_count = base_plan.native_frame_count - VIDEO_PREFIX_FRAMES
    global_frame_start = base_plan.native_frame_count + (request_index - 1) * steady_frame_count
    global_audio_start = audio_latent_boundary(global_frame_start)
    phases: list[StreamPhase] = []
    frame_stop = 0
    video_stop = 0
    for index, base_phase in enumerate(base_plan.phases):
        frame_start = frame_stop
        video_start = video_stop
        frame_stop += base_phase.group_count * VIDEO_GROUP_FRAMES
        video_stop += base_phase.group_count * VIDEO_GROUP_LATENTS
        phases.append(
            StreamPhase(
                index=index,
                group_count=base_phase.group_count,
                frame_start=frame_start,
                frame_stop=frame_stop,
                video_latent_start=video_start,
                video_latent_stop=video_stop,
                audio_latent_start=audio_latent_boundary(global_frame_start + frame_start) - global_audio_start,
                audio_latent_stop=audio_latent_boundary(global_frame_start + frame_stop) - global_audio_start,
            )
        )
    return StreamPlan(native_frame_count=steady_frame_count, phases=tuple(phases))


def request_plan(request_index: int) -> StreamPlan:
    """The plan of request ``request_index`` of a session."""
    base = direct_5s_plan()
    if request_index == 0:
        return base
    return canonical_continuation_plan(base, request_index=request_index)


@dataclass(frozen=True)
class CanvasGeometry:
    """Latent canvas of one output resolution."""

    height: int
    width: int

    def __post_init__(self) -> None:
        if isinstance(self.width, bool) or isinstance(self.height, bool):
            raise ValueError("width and height must be integers")
        if self.width <= 0 or self.height <= 0 or self.width % CANVAS_MULTIPLE or self.height % CANVAS_MULTIPLE:
            raise ValueError("TaoMate-H3 width and height must be positive multiples of 32")
        if min(self.width, self.height) not in SUPPORTED_SHORT_EDGES:
            raise ValueError("TaoMate-H3 requires a 480-, 768-, or 1088-pixel short edge")

    @property
    def latent_h(self) -> int:
        return self.height // 16

    @property
    def latent_w(self) -> int:
        return self.width // 16

    @property
    def frame_rows(self) -> int:
        """Packed DiT rows per video latent frame ((H/16/2) * (W/16/2))."""
        return (self.latent_h // 2) * (self.latent_w // 2)


def phases_for_frames(num_frames: int) -> int:
    """Number of phases (chunks) needed to publish at least ``num_frames`` frames."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    requests = 1
    published = REQUEST_NATIVE_FRAMES
    while published < num_frames:
        requests += 1
        published += STEADY_NATIVE_FRAMES
    return requests * len(PHASE_GROUP_COUNTS)


__all__ = [
    "AUDIO_LATENT_RATE",
    "AUDIO_SAMPLES_PER_LATENT",
    "AUDIO_SAMPLE_RATE",
    "CanvasGeometry",
    "PHASE_GROUP_COUNTS",
    "REQUEST_AUDIO_LATENTS",
    "REQUEST_NATIVE_FRAMES",
    "REQUEST_VIDEO_LATENTS",
    "STEADY_NATIVE_FRAMES",
    "StreamPhase",
    "StreamPlan",
    "VIDEO_FPS",
    "audio_latent_boundary",
    "canonical_continuation_plan",
    "direct_5s_plan",
    "phases_for_frames",
    "request_plan",
    "video_temporal_position",
    "video_temporal_positions",
]
