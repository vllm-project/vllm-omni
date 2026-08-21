# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-session driver for realtime AR-Diffusion benchmarks.

The driver owns arrivals, pacing and recording. It talks to sessions through
:class:`BenchmarkSession`, a two-method view of the realtime session API, so
the whole load model is exercisable on CPU against a fake session and needs no
engine, device or checkpoint.

Load parameters, and why only one of them is an axis:

``session arrival rate``
    The valid open-loop parameter. Sessions arrive and depart over time.

``per-session tick rate``
    Not an axis. Ticks of one session are strictly ordered and state-dependent:
    tick ``k + 1`` reads the state committed by tick ``k`` and cannot be issued
    before the previous chunk returns. Raising it only grows a queue.

``concurrently active sessions``
    Derived state, not an input. Reported alongside every measurement.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from benchmarks.ar_diffusion.realtime_metrics import (
    ChunkEvent,
    LoadMode,
    RunSummary,
    SessionRecord,
    WorkloadProfile,
    summarize_run,
)


class BenchmarkSession(Protocol):
    """The part of a realtime session the driver needs."""

    async def next_chunk(self) -> Any: ...

    async def close(self) -> None: ...


SessionFactory = Callable[[str], Awaitable[BenchmarkSession]]


@dataclass(frozen=True)
class ArrivalPlan:
    """When each session starts, relative to the run start."""

    offsets_s: tuple[float, ...]

    def __post_init__(self) -> None:
        if any(offset < 0 for offset in self.offsets_s):
            raise ValueError("arrival offsets must be non-negative.")
        if list(self.offsets_s) != sorted(self.offsets_s):
            raise ValueError("arrival offsets must be non-decreasing.")

    def __len__(self) -> int:
        return len(self.offsets_s)


def burst_arrivals(num_sessions: int) -> ArrivalPlan:
    """All sessions start together: the worst case for admission and memory."""
    if num_sessions < 1:
        raise ValueError("num_sessions must be at least 1.")
    return ArrivalPlan(tuple(0.0 for _ in range(num_sessions)))


def poisson_arrivals(num_sessions: int, *, rate_per_s: float, seed: int = 0) -> ArrivalPlan:
    """Poisson session arrivals; deterministic for a given seed."""
    if num_sessions < 1:
        raise ValueError("num_sessions must be at least 1.")
    if rate_per_s <= 0:
        raise ValueError("rate_per_s must be positive.")
    rng = random.Random(seed)
    offsets: list[float] = []
    clock = 0.0
    for _ in range(num_sessions):
        offsets.append(clock)
        clock += rng.expovariate(rate_per_s)
    return ArrivalPlan(tuple(offsets))


class _ConcurrencyGauge:
    """Tracks how many sessions were ticking at once."""

    def __init__(self) -> None:
        self.current = 0
        self.peak = 0

    def enter(self) -> None:
        self.current += 1
        self.peak = max(self.peak, self.current)

    def exit(self) -> None:
        self.current -= 1


@dataclass
class BenchmarkConfig:
    profile: WorkloadProfile
    mode: LoadMode
    chunks_per_session: int
    arrivals: ArrivalPlan
    num_gpus: int = 1
    events_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.chunks_per_session < 1:
            raise ValueError("chunks_per_session must be at least 1.")


async def _drive_session(
    session_id: str,
    *,
    config: BenchmarkConfig,
    factory: SessionFactory,
    start_at: float,
    clock: Callable[[], float],
    sleep: Callable[[float], Awaitable[None]],
    gauge: _ConcurrencyGauge,
) -> SessionRecord:
    """Run one session to ``chunks_per_session`` chunks, or until it is lost."""
    delay = start_at - clock()
    if delay > 0:
        await sleep(delay)

    t_start = clock()
    session = await factory(session_id)
    events: list[ChunkEvent] = []
    decoder_bytes: list[int] = []
    lost_reason: str | None = None
    period = config.profile.release_period_s
    gauge.enter()
    try:
        for chunk_index in range(config.chunks_per_session):
            if config.mode is LoadMode.PACED and events:
                # A paced client consumes one chunk per release period, so the
                # next tick is not issued before the previous chunk's playout
                # would have finished. Falling behind never pushes the client
                # ahead: the due time is anchored to the playout grid.
                due = t_start + chunk_index * period
                lag = due - clock()
                if lag > 0:
                    await sleep(lag)
            t_submit = clock()
            try:
                await session.next_chunk()
            except Exception as exc:  # noqa: BLE001 - a lost session is a result, not a crash
                lost_reason = f"{type(exc).__name__}: {exc}"
                break
            # A session that decodes its own chunks reports the generate/decode
            # split and the frames it actually delivered; one that returns
            # latents leaves these None and the profile supplies the geometry.
            timing = getattr(session, "last", None)
            events.append(
                ChunkEvent(
                    session_id=session_id,
                    chunk_index=chunk_index,
                    t_submit=t_submit,
                    t_ready=clock(),
                    frames=getattr(timing, "frames", None),
                    generate_s=getattr(timing, "generate_s", None),
                    decode_s=getattr(timing, "decode_s", None),
                    overlap_s=getattr(timing, "overlap_s", None),
                    outstanding_generations=getattr(timing, "outstanding_generations", None),
                )
            )
            resident = getattr(timing, "resident_decoder_bytes", None)
            if resident is not None:
                decoder_bytes.append(resident)
    finally:
        gauge.exit()
        with contextlib.suppress(Exception):
            await session.close()

    return SessionRecord(
        session_id=session_id,
        t_start=t_start,
        events=tuple(events),
        lost_reason=lost_reason,
        # Peak rather than last: the cache is bounded, so these agree in steady
        # state, and a peak is what admission has to reserve against.
        resident_decoder_bytes=max(decoder_bytes) if decoder_bytes else None,
    )


async def run_benchmark(
    config: BenchmarkConfig,
    factory: SessionFactory,
    *,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    notes: Sequence[str] = (),
) -> RunSummary:
    """Run every session concurrently and reduce the result to a summary.

    ``clock`` and ``sleep`` are injectable so tests can drive the whole load
    model on a virtual clock without spending wall time.
    """
    resolved_clock = clock or time.perf_counter
    resolved_sleep = sleep or asyncio.sleep
    gauge = _ConcurrencyGauge()

    t0 = resolved_clock()
    tasks = [
        asyncio.create_task(
            _drive_session(
                f"bench-{index}",
                config=config,
                factory=factory,
                start_at=t0 + offset,
                clock=resolved_clock,
                sleep=resolved_sleep,
                gauge=gauge,
            )
        )
        for index, offset in enumerate(config.arrivals.offsets_s)
    ]
    records = await asyncio.gather(*tasks)
    wall = resolved_clock() - t0
    reported = [r.resident_decoder_bytes for r in records if r.resident_decoder_bytes is not None]
    resident_decoder_bytes = max(reported) if reported else None

    if config.events_dir is not None:
        write_events(records, config)

    return summarize_run(
        records,
        config.profile,
        mode=config.mode,
        wall_s=wall,
        num_gpus=config.num_gpus,
        peak_concurrent_sessions=gauge.peak,
        resident_decoder_bytes_per_session=resident_decoder_bytes,
        notes=notes,
    )


def write_events(records: Sequence[SessionRecord], config: BenchmarkConfig) -> None:
    """Write one JSONL of chunk events per session."""
    if config.events_dir is None:
        raise ValueError("config.events_dir must be set to write events.")
    from benchmarks.ar_diffusion.realtime_metrics import chunk_deadlines

    config.events_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        deadlines = chunk_deadlines(record, config.profile) if config.mode is LoadMode.PACED else {}
        path = config.events_dir / f"{record.session_id}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for event in record.events:
                handle.write(json.dumps(event.to_dict(deadline=deadlines.get(event.chunk_index))) + "\n")
