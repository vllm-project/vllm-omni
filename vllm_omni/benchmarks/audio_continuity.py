# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Streaming-audio continuity metrics for speech, chat and duplex benchmarks.

Captures the failure mode where ``RTF_p50 < 1`` (server keeps up with realtime
in aggregate) but per-stream chunk arrival is bursty enough that listeners
hear audible gaps.  Companion to ``audio_ttfp`` / ``audio_rtf``.

The math is purposefully simple so the helper has no ``vllm`` dependency and
can be unit-tested in isolation: given the wall-clock arrival timeline of
audio chunks and the PCM format, simulate a player that consumes at realtime
rate from the first chunk onward, and measure individual stalls and their sum.

Continuity remains a pass/fail check on total stall time, not the largest stall.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict

import pybase64 as base64


class ContinuitySample(TypedDict):
    """One measured response; byte counts have already been decoded to PCM."""

    audio_underrun_s: float
    audio_max_single_underrun_s: float
    total_underrun_s: float
    audio_continuity_ok: bool
    audio_underrun_event_count: int
    audio_duration_s: float
    audio_underrun_ratio: float
    audio_underrun_events_per_minute: float
    audio_chunk_count: int
    audio_continuity_threshold_s: float


@dataclass(frozen=True)
class ContinuityStats:
    """Result of a single-request continuity analysis.

    Attributes:
        max_underrun_s: Legacy maximum cumulative deficit, equal to total stalls.
        max_single_underrun_s: Largest individual playback stall in seconds.
        underrun_event_count: Number of actual new playback stalls.
        is_continuous: Total stall time stays within the continuity budget.
        total_underrun_s: Sum of playback stalls, excluding first-packet wait.
        audio_duration_s: Total playable audio duration, including the last chunk.
        underrun_ratio: Total stall time / (audio duration + total stall time).
        underrun_events_per_minute: Stall count per minute of audio content.
    """

    max_underrun_s: float
    underrun_event_count: int
    is_continuous: bool
    max_single_underrun_s: float = 0.0
    total_underrun_s: float = 0.0
    audio_duration_s: float = 0.0
    underrun_ratio: float = 0.0
    underrun_events_per_minute: float = 0.0


def audio_continuity_threshold_s() -> float:
    """Shared benchmark threshold; invalid environment values use 100 ms."""
    try:
        value = float(os.environ.get("VLLM_OMNI_BENCH_AUDIO_CONTINUITY_THRESHOLD_S", "0.1"))
    except ValueError:
        return 0.1
    return max(value, 0.0) if math.isfinite(value) else 0.1


def continuity_sample(
    arrivals_s: Sequence[float], durations_s: Sequence[float], *, threshold_s: float
) -> ContinuitySample | None:
    """Measure one playable stream, excluding first-packet wait and trailing silence.

    A stalled player resumes when the next chunk arrives; buffered chunks play
    in order. ``audio_max_single_underrun_s`` is the largest single stall.
    ``audio_underrun_s`` keeps its legacy cumulative-deficit meaning and equals
    ``total_underrun_s``. Existing dashboards therefore keep comparable values.
    No audio or an invalid timeline is unmeasured, never a passing sample.
    """
    if not arrivals_s or len(arrivals_s) != len(durations_s):
        return None
    if not math.isfinite(threshold_s) or threshold_s < 0:
        return None
    if any(not math.isfinite(t) for t in arrivals_s):
        return None
    if any(not math.isfinite(d) or d <= 0 for d in durations_s):
        return None
    if any(b < a for a, b in zip(arrivals_s, arrivals_s[1:])):
        return None
    playback_end_s = 0.0
    total_s = 0.0
    worst_s = 0.0
    events = 0
    for index, (arrival, duration) in enumerate(zip(arrivals_s, durations_s, strict=True)):
        relative_arrival_s = arrival - arrivals_s[0]
        gap_s = max(0.0, relative_arrival_s - playback_end_s)
        if index and gap_s > 1e-9:
            worst_s = max(worst_s, gap_s)
            total_s += gap_s
            events += 1
        playback_end_s = max(relative_arrival_s, playback_end_s) + duration
    audio_duration_s = sum(durations_s)
    playback_duration_s = audio_duration_s + total_s
    events_per_minute = events * 60.0 / audio_duration_s
    if not all(math.isfinite(value) for value in (playback_end_s, playback_duration_s, events_per_minute)):
        return None
    return {
        "audio_underrun_s": total_s,
        "audio_max_single_underrun_s": worst_s,
        "total_underrun_s": total_s,
        "audio_continuity_ok": total_s <= threshold_s + 1e-9,
        "audio_underrun_event_count": events,
        "audio_duration_s": audio_duration_s,
        "audio_underrun_ratio": total_s / playback_duration_s,
        "audio_underrun_events_per_minute": events_per_minute,
        "audio_chunk_count": len(arrivals_s),
        "audio_continuity_threshold_s": threshold_s,
    }


def collect_duplex_audio_continuity(
    events: Sequence[Mapping[str, object]], received_at_s: Sequence[float], *, after_s: float = 0.0
) -> dict[str, dict[str, object]]:
    """Measure completed PCM16 replies independently using client receive times.

    LISTEN-only, cancelled/failed/incomplete replies and malformed audio are
    excluded. No intervals cross response boundaries; cancellation never adds
    a synthetic silence interval. Late packets after response.done are ignored.
    """
    timelines: dict[str, tuple[list[float], list[float]]] = {}
    invalid: set[str] = set()
    done: set[str] = set()
    completed: set[str] = set()
    rate = 24_000
    encoding = "pcm16"
    for event, arrival in zip(events, received_at_s, strict=True):
        session = event.get("session")
        if event.get("type") in {"session.created", "session.updated", "session.resumed"} and isinstance(session, dict):
            encoding = session.get("output_audio_format", encoding)
            audio_config = session.get("audio")
            output_config = audio_config.get("output") if isinstance(audio_config, dict) else None
            if isinstance(output_config, dict):
                rate = output_config.get("sample_rate_hz", rate)
        response = event.get("response")
        rid = event.get("response_id") or (response.get("id") if isinstance(response, dict) else None)
        if not isinstance(rid, str) or arrival < after_s:
            continue
        kind = event.get("type")
        if kind == "response.done":
            done.add(rid)
            status = response.get("status", event.get("status")) if isinstance(response, dict) else event.get("status")
            if status in (None, "completed") and rid not in invalid:
                completed.add(rid)
            else:
                invalid.add(rid)
            continue
        if kind in {"response.cancelled", "response.canceled", "response.failed"}:
            invalid.add(rid)
            done.add(rid)
        if kind not in {"response.output_audio.delta", "response.audio.delta"} or rid in done:
            continue
        encoded = event.get("delta") or event.get("audio")
        if not encoded:
            continue
        event_rate = event.get("sample_rate_hz", rate)
        if encoding != "pcm16" or not isinstance(event_rate, int) or isinstance(event_rate, bool) or event_rate <= 0:
            invalid.add(rid)
            continue
        rate = event_rate
        try:
            pcm = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError):
            invalid.add(rid)
            continue
        if len(pcm) % 2:
            invalid.add(rid)
            continue
        if pcm:
            times, durations = timelines.setdefault(rid, ([], []))
            times.append(arrival)
            durations.append(len(pcm) / (2 * rate))
    result = {}
    threshold = audio_continuity_threshold_s()
    for rid, (times, durations) in timelines.items():
        if rid not in completed or rid in invalid:
            continue
        sample = continuity_sample(times, durations, threshold_s=threshold)
        if sample is not None:
            result[rid] = {"response_id": rid, **sample}
    return result


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
        threshold_s: Total underrun budget per response (default 0.1 s).

    Returns:
        A :class:`ContinuityStats` with maximum and total stalls, event count,
        ratio and frequency. Continuity compares total stalls to the threshold.
    """
    n = len(chunk_arrival_times_s)
    if n == 0 or n != len(chunk_bytes):
        return ContinuityStats(0.0, 0, True)

    bytes_per_s = sample_rate * sample_width * channels
    if bytes_per_s <= 0:
        return ContinuityStats(0.0, 0, True)

    sample = continuity_sample(
        chunk_arrival_times_s, [size / bytes_per_s for size in chunk_bytes], threshold_s=threshold_s
    )
    if sample is None:
        # Keep the existing defensive return contract for byte-based callers.
        return ContinuityStats(0.0, 0, True)
    return ContinuityStats(
        max_underrun_s=sample["audio_underrun_s"],
        max_single_underrun_s=sample["audio_max_single_underrun_s"],
        underrun_event_count=sample["audio_underrun_event_count"],
        is_continuous=sample["audio_continuity_ok"],
        total_underrun_s=sample["total_underrun_s"],
        audio_duration_s=sample["audio_duration_s"],
        underrun_ratio=sample["audio_underrun_ratio"],
        underrun_events_per_minute=sample["audio_underrun_events_per_minute"],
    )
