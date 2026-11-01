# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chunk boundary and overlap arithmetic for MiniMax Music 3.

The AR stage emits fixed 200-frame windows that overlap by half. The acoustic
stage decodes each window with the previous window's latent as a prompt, then
crops the overlap back off so the concatenated waveform is seamless. All of
that is pure integer arithmetic and lives here so it can be unit tested
without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

from .constants import (
    AR_CHUNK_FRAMES,
    AR_CHUNK_HOP_FRAMES,
    BLEND_MEL_FRAMES,
    DAV_HOP_SAMPLES,
    HOP_MEL_FRAMES,
)


@dataclass(frozen=True)
class ChunkWindow:
    """One acoustic window, in AR frame coordinates."""

    index: int
    start: int
    end: int
    is_first: bool
    is_last: bool

    @property
    def length(self) -> int:
        return self.end - self.start


def chunk_windows(frames: int) -> list[ChunkWindow]:
    """Return the fixed-size windows covering ``frames``, overlapping by half.

    A song shorter than one window produces a single short window; otherwise
    windows advance by ``AR_CHUNK_HOP_FRAMES`` and the final one is truncated
    at the end of the song.
    """
    frames = int(frames)
    if frames <= 0:
        return []
    if frames <= AR_CHUNK_FRAMES:
        return [ChunkWindow(0, 0, frames, True, True)]

    windows: list[ChunkWindow] = []
    index = 0
    start = 0
    while start < frames:
        end = min(start + AR_CHUNK_FRAMES, frames)
        windows.append(
            ChunkWindow(
                index=index,
                start=start,
                end=end,
                is_first=index == 0,
                is_last=end >= frames,
            )
        )
        if end >= frames:
            break
        index += 1
        start += AR_CHUNK_HOP_FRAMES
    return windows


def overlap_mel_length() -> int:
    """Latent frames carried forward as the next window's flow-matching prompt."""
    return HOP_MEL_FRAMES // 2


def crop_sample_bounds(window: ChunkWindow) -> tuple[int, int]:
    """Return ``(left, right)`` waveform samples to trim from a decoded window.

    The first window keeps its head and the last keeps its tail; every other
    window is cropped on both sides so that concatenating the survivors
    reconstructs the song exactly once.
    """
    left = 0 if window.is_first else BLEND_MEL_FRAMES * DAV_HOP_SAMPLES
    right = 0 if window.is_last else (HOP_MEL_FRAMES - BLEND_MEL_FRAMES) * DAV_HOP_SAMPLES
    return left, right


__all__ = [
    "ChunkWindow",
    "chunk_windows",
    "crop_sample_bounds",
    "overlap_mel_length",
]
