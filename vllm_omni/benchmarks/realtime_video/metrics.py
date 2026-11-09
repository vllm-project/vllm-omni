# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Public raw-measurement validation shared by bench serve and CI assertions."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any

from .workload import pixel_frames_in_chunk


class RegressionError(ValueError):
    """Missing evidence, invalid calibration, or an observed regression."""


def number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RegressionError(f"{label} must be finite: {value!r}")
    if value < 0 or (positive and value == 0):
        raise RegressionError(f"{label} must be {'positive' if positive else 'nonnegative'}")
    return float(value)


def performance_metrics(document: Mapping[str, Any], *, num_chunks: int, min_steady: int) -> dict[str, float]:
    """Recompute the population from every session; reject partial/missing data.

    P99 is always reported, but is eligible for a gate only with >=1000 steady
    observations. Warmup and terminal-chunk policy are owned by workload.py.
    """
    if any(
        isinstance(row, Mapping) and row.get("success") is False
        for key in ("sessions", "warmups")
        if isinstance(document.get(key), list)
        for row in document[key]
    ):
        raise RegressionError("Failed realtime-video session or warmup is not a complete measurement")
    sessions = document.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise RegressionError("No measured sessions")
    intervals: list[float] = []
    ttfc: list[float] = []
    generation_fps: list[float] = []
    stalls: list[float] = []
    startup_stalls: list[float] = []
    steady_stalls: list[float] = []
    for session in sessions:
        chunks = session.get("chunks", [])
        if len(chunks) != num_chunks or [row.get("index") for row in chunks] != list(range(num_chunks)):
            raise RegressionError("Incomplete or out-of-order chunk population")
        metrics = session["metrics"]
        if not metrics.get("steady_state_reached"):
            raise RegressionError("Attention window never reached steady state")
        # The records are kept in their native schema; use the same selector
        # as compute_metrics rather than reconstructing a different population.
        from .workload import ChunkRecord, compute_metrics

        records = [
            ChunkRecord(
                index=row["index"],
                arrival_s=number(row["arrival_s"], "arrival", positive=True),
                inter_arrival_s=number(row["inter_arrival_ms"], "interval", positive=True) / 1000,
                byte_length=row["byte_length"],
                num_frames=row["num_frames"],
            )
            for row in chunks
        ]
        for index, record in enumerate(records):
            previous = records[index - 1].arrival_s if index else 0
            if not math.isclose(record.arrival_s - previous, record.inter_arrival_s, abs_tol=1e-6):
                raise RegressionError("Raw arrivals and intervals disagree")
            if record.byte_length <= 0 or record.num_frames != pixel_frames_in_chunk(index):
                raise RegressionError("Missing media or incorrect frame coverage")
        config = document["config"]
        recomputed = compute_metrics(
            records,
            fps=config["target_fps"],
            warmup_chunks=config["warmup_chunks"],
            slo_ms=config["chunk_slo_ms"],
            playback_buffer_chunks=config["playback_buffer_chunks"],
            ttfc_s=records[0].arrival_s,
            session_wall_s=number(metrics["wall_seconds"], "wall time", positive=True),
        )
        if recomputed["wall_seconds"] < records[-1].arrival_s:
            raise RegressionError("Session wall time precedes its last chunk")
        metrics = recomputed
        start = max(1, config["warmup_chunks"])
        intervals.extend(record.inter_arrival_s * 1000 for record in records[start:-1])
        ttfc.append(number(metrics["ttfc_ms"], "TTFC", positive=True))
        generation_fps.append(number(metrics["total_frames"] / metrics["wall_seconds"], "generated FPS", positive=True))
        stalls.append(number(metrics["playback"]["total_stall_ms"], "playback stall"))
        startup_stalls.append(number(metrics["playback"]["startup_stall_ms"], "startup playback stall"))
        steady_stalls.append(number(metrics["playback"]["steady_stall_ms"], "steady playback stall"))
    if len(intervals) < min_steady:
        raise RegressionError(f"Only {len(intervals)} steady samples; require {min_steady}")
    from .workload import percentile

    return {
        "mean_ms": statistics.mean(intervals),
        "p50_ms": percentile(intervals, 0.50),
        "p95_ms": percentile(intervals, 0.95),
        "p99_ms": percentile(intervals, 0.99),
        "ttfc_ms": statistics.mean(ttfc),
        "generated_fps": min(generation_fps),
        "stall_ms": max(stalls),
        "startup_stall_ms": max(startup_stalls),
        "steady_stall_ms": max(steady_stalls),
        "steady_samples": float(len(intervals)),
    }


def measured_metrics(document: dict, settings: dict) -> dict[str, float]:
    sessions = document.get("sessions", [])
    count = settings.get("measured_sessions", 1)
    ids = [session.get("request_id") for session in sessions]
    if (
        len(sessions) != count
        or [session.get("session_index") for session in sessions] != list(range(count))
        or any(not isinstance(identity, str) or not identity for identity in ids)
        or len(set(ids)) != count
    ):
        raise RegressionError("Missing or duplicated measured sessions in an independent start")
    if any(document["config"].get(key) != settings[key] for key in ("warmup_chunks", "target_fps")):
        raise RegressionError("Benchmark measurement policy differs from the profile")
    if document["config"].get("playback_buffer_chunks") != 1:
        raise RegressionError("LingBot regression requires the pinned one-chunk playback buffer")
    warmups = document.get("warmups", [])
    if (
        document["config"].get("warmup_sessions") != 1
        or document["config"].get("warmup_num_chunks") != settings["warmup_num_chunks"]
        or len(warmups) != 1
        or warmups[0].get("measured") is not False
        or not isinstance(warmups[0].get("request_id"), str)
        or not warmups[0]["request_id"]
        or warmups[0]["request_id"] in ids
    ):
        raise RegressionError("Missing or mixed warmup session")
    chunks = warmups[0].get("chunks", [])
    if (
        len(chunks) != settings["warmup_num_chunks"]
        or [row.get("index") for row in chunks] != list(range(settings["warmup_num_chunks"]))
        or any(
            row.get("byte_length", 0) <= 0 or row.get("num_frames") != pixel_frames_in_chunk(i)
            for i, row in enumerate(chunks)
        )
    ):
        raise RegressionError("Incomplete warmup chunk population")
    return performance_metrics(document, num_chunks=settings["num_chunks"], min_steady=settings["min_steady_samples"])
