# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact CFR clocks for linear frame resampling with a held final sample."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

_MAX_OUTPUT_FRAMES = 4096


def _positive_fraction(value: Fraction | int, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (Fraction, int)):
        raise TypeError(f"{name} must be a Fraction or integer")
    value = Fraction(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True, init=False)
class FrameInterpolationPlan:
    """Map N source exposure intervals onto an explicit CFR output clock.

    Duration is the sum of frame exposure intervals, not the last frame's
    presentation timestamp. Source positions use ``j * N / M``; positions
    beyond the final source sample hold that sample. This resampling grid
    need not retain every original sample position.
    """

    source_frames: int
    source_fps: Fraction
    output_fps: Fraction
    output_duration: Fraction

    def __init__(
        self,
        source_frames: int,
        source_fps: Fraction | int,
        output_fps: Fraction | int,
        output_duration: Fraction | int,
    ) -> None:
        if isinstance(source_frames, bool) or not isinstance(source_frames, int):
            raise TypeError("source_frames must be an integer")
        if source_frames < 2:
            raise ValueError("source_frames must be at least 2")
        object.__setattr__(self, "source_frames", source_frames)
        object.__setattr__(self, "source_fps", _positive_fraction(source_fps, "source_fps"))
        object.__setattr__(self, "output_fps", _positive_fraction(output_fps, "output_fps"))
        object.__setattr__(self, "output_duration", _positive_fraction(output_duration, "output_duration"))
        frames = self.output_fps * self.output_duration
        if frames.denominator != 1:
            raise ValueError("output_fps * output_duration must be an integer frame count")
        if frames.numerator < source_frames:
            raise ValueError("output frame count must be at least source_frames; downsampling is unsupported")
        if frames.numerator > _MAX_OUTPUT_FRAMES:
            raise ValueError(f"resampling supports at most {_MAX_OUTPUT_FRAMES} output frames")

    @property
    def output_frames(self) -> int:
        return (self.output_fps * self.output_duration).numerator

    @property
    def source_duration(self) -> Fraction:
        return self.source_frames / self.source_fps

    def source_position(self, slot: int) -> tuple[int, int, float]:
        """Return adjacent source indices and the RIFE pair timestep."""
        if isinstance(slot, bool) or not isinstance(slot, int):
            raise TypeError("slot must be an integer")
        frames = self.output_frames
        if not 0 <= slot < frames:
            raise IndexError(f"output slot must be in [0, {frames})")
        left, remainder = divmod(slot * self.source_frames, frames)
        right = left if remainder == 0 else min(left + 1, self.source_frames - 1)
        return left, right, 0.0 if left == right else remainder / frames
