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
from collections.abc import Awaitable, Callable, Mapping, Sequence
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


def _stage_durations(output: Any) -> dict[str, float]:
    """Read the pipeline profiler's per-stage seconds off one tick output.

    Deliberately total, never raising: stage timings are diagnostic, and a
    pipeline that reports none, or reports them in a shape this harness does
    not recognise, must still produce a benchmark result. What it must not do
    is silently contribute a zero, which is why an unrecognised shape yields
    an empty mapping and is counted as uninstrumented rather than as a tick
    whose stages took no time.
    """
    durations = getattr(output, "stage_durations", None)
    if not isinstance(durations, Mapping):
        return {}
    return {
        str(name): float(seconds)
        for name, seconds in durations.items()
        if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and seconds >= 0
    }


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
    lost_reason: str | None = None
    gauge.enter()
    try:
        for chunk_index in range(config.chunks_per_session):
            if config.mode is LoadMode.PACED and events:
                # Pace against the video actually delivered before this
                # chunk.  A causal decoder's opening chunk can be shorter than
                # steady state (LingBot/Wan: 9 frames, then 12), so multiplying
                # every index by the steady period delays chunk 1 by 3 frames
                # and manufactures a deadline miss.  Anchor to the first real
                # submit so session construction time is not part of playout.
                due = (
                    events[0].t_submit
                    + config.profile.cumulative_frames(chunk_index) / config.profile.target_fps
                )
                lag = due - clock()
                if lag > 0:
                    await sleep(lag)
            t_submit = clock()
            try:
                output = await session.next_chunk()
            except Exception as exc:  # noqa: BLE001 - a lost session is a result, not a crash
                lost_reason = f"{type(exc).__name__}: {exc}"
                break
            events.append(
                ChunkEvent(
                    session_id=session_id,
                    chunk_index=chunk_index,
                    t_submit=t_submit,
                    t_ready=clock(),
                    stage_durations=_stage_durations(output),
                )
            )
    finally:
        gauge.exit()
        with contextlib.suppress(Exception):
            await session.close()

    return SessionRecord(
        session_id=session_id,
        t_start=t_start,
        events=tuple(events),
        lost_reason=lost_reason,
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

    if config.events_dir is not None:
        write_events(records, config)

    return summarize_run(
        records,
        config.profile,
        mode=config.mode,
        wall_s=wall,
        num_gpus=config.num_gpus,
        peak_concurrent_sessions=gauge.peak,
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
