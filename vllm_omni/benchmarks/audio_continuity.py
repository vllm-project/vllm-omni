# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Streaming-audio continuity metrics for the TTS bench.

Captures the failure mode where ``RTF_p50 < 1`` (server keeps up with realtime
in aggregate) but per-stream chunk arrival is bursty enough that listeners
hear audible gaps.  Companion to ``audio_ttfp`` / ``audio_rtf``.

The math is purposefully simple so the helper has no ``vllm`` dependency and
can be unit-tested in isolation: given the wall-clock arrival timeline of
audio chunks and the PCM format, simulate a player that consumes at realtime
rate from the first chunk onward, and surface the worst-case deficit.

A deficit > ``threshold_s`` at any point counts as an audible underrun.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContinuityStats:
    """Result of a single-request continuity analysis.

    Attributes:
        max_underrun_s: Worst-case wall-clock seconds the player was starved.
        underrun_event_count: Inter-chunk intervals during which the buffer
            went negative (one per gap, not per starved millisecond).
        is_continuous: ``max_underrun_s <= threshold_s`` for the request.
    """

    max_underrun_s: float
    underrun_event_count: int
    is_continuous: bool


def compute_continuity_stats(
    chunk_arrival_times_s: list[float],
    chunk_bytes: list[int],
    sample_rate: int,
    sample_width: int = 2,
    channels: int = 1,
    threshold_s: float = 0.1,
) -> ContinuityStats:
    """Compute the continuity stats for one streaming response.

    Args:
        chunk_arrival_times_s: Wall-clock seconds (since request start) at
            which each non-empty audio chunk's *last byte* arrived. Must be
            monotonically non-decreasing.
        chunk_bytes: Byte count of each chunk in the same order.
        sample_rate: PCM sample rate (Hz). 24 000 for Qwen3-TTS / VoxCPM2.
        sample_width: Bytes per sample (2 = s16le).
        channels: PCM channel count (1 = mono).
        threshold_s: Underrun budget. The default 0.1 s matches the
            commonly-cited "audible gap" threshold for streaming TTS.

    Returns:
        A :class:`ContinuityStats` summarising the worst-case deficit and
        whether it stayed under the threshold.
    """
    n = len(chunk_arrival_times_s)
    if n == 0 or n != len(chunk_bytes):
        return ContinuityStats(0.0, 0, True)

    bytes_per_s = sample_rate * sample_width * channels
    if bytes_per_s <= 0:
        return ContinuityStats(0.0, 0, True)

    t0 = chunk_arrival_times_s[0]
    received_before = 0
    max_underrun_s = 0.0
    event_count = 0
    for i in range(n):
        if i > 0:
            played_bytes = (chunk_arrival_times_s[i] - t0) * bytes_per_s
            deficit_bytes = played_bytes - received_before
            if deficit_bytes > 0:
                deficit_s = deficit_bytes / bytes_per_s
                if deficit_s > max_underrun_s:
                    max_underrun_s = deficit_s
                event_count += 1
        received_before += chunk_bytes[i]

    return ContinuityStats(
        max_underrun_s=max_underrun_s,
        underrun_event_count=event_count,
        is_continuous=max_underrun_s <= threshold_s,
    )
