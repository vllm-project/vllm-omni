#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Event:
    mode: str
    stream: str
    event: str
    t_s: float
    data: dict[str, Any]


@dataclass(frozen=True)
class SimulationResult:
    mode: str
    client_events: list[Event]
    server_events: list[Event]
    total_time_s: float
    idle_time_s: float
    executed_rows: int
    underrun_rows: int

    @property
    def effective_control_idle_ratio(self) -> float:
        return self.idle_time_s / self.total_time_s if self.total_time_s > 0 else 0.0


def simulate_sync(
    *,
    num_chunks: int,
    action_horizon: int,
    chunk_duration_s: float,
    forward_latency_s: float,
) -> SimulationResult:
    t_s = 0.0
    client_events: list[Event] = []
    server_events: list[Event] = []
    idle_time_s = 0.0
    executed_rows = 0

    for chunk_index in range(1, num_chunks + 1):
        client_events.append(_event("sync", "client", "observation_sent", t_s, chunk_index=chunk_index))
        server_events.append(_event("sync", "server", "forward_started", t_s, chunk_index=chunk_index))
        idle_time_s += forward_latency_s
        t_s += forward_latency_s
        server_events.append(_event("sync", "server", "action_chunk_ready", t_s, chunk_index=chunk_index))
        client_events.append(_event("sync", "client", "action_chunk_received", t_s, chunk_index=chunk_index))
        client_events.append(_event("sync", "client", "chunk_execution_started", t_s, chunk_index=chunk_index))
        t_s += chunk_duration_s
        executed_rows += action_horizon
        client_events.append(_event("sync", "client", "chunk_execution_finished", t_s, chunk_index=chunk_index))

    return SimulationResult(
        mode="sync",
        client_events=client_events,
        server_events=server_events,
        total_time_s=t_s,
        idle_time_s=idle_time_s,
        executed_rows=executed_rows,
        underrun_rows=0,
    )


def simulate_async(
    *,
    num_chunks: int,
    action_horizon: int,
    chunk_duration_s: float,
    forward_latency_s: float,
) -> SimulationResult:
    client_events: list[Event] = []
    server_events: list[Event] = []
    ready_at: dict[int, float] = {}

    t_s = 0.0
    client_events.append(_event("async", "client", "observation_sent", t_s, observation_index=1))

    server_events.append(_event("async", "server", "forward_started", t_s, chunk_index=1, prefix="O1_real"))
    ready_at[1] = t_s + forward_latency_s
    server_events.append(_event("async", "server", "action_chunk_ready", ready_at[1], chunk_index=1))

    server_events.append(
        _event("async", "server", "forward_started", ready_at[1], chunk_index=2, prefix="O1_real,O2_sim")
    )
    ready_at[2] = ready_at[1] + forward_latency_s
    server_events.append(_event("async", "server", "action_chunk_ready", ready_at[2], chunk_index=2))

    idle_time_s = 0.0
    executed_rows = 0
    underrun_rows = 0

    for chunk_index in range(1, num_chunks + 1):
        chunk_ready_s = ready_at.get(chunk_index)
        if chunk_ready_s is None:
            chunk_ready_s = t_s + 2.0 * forward_latency_s
            ready_at[chunk_index] = chunk_ready_s
            server_events.append(
                _event("async", "server", "action_chunk_ready", chunk_ready_s, chunk_index=chunk_index)
            )

        if chunk_ready_s > t_s:
            idle_time_s += chunk_ready_s - t_s
            underrun_rows += int(round((chunk_ready_s - t_s) / chunk_duration_s * action_horizon))
            t_s = chunk_ready_s

        client_events.append(_event("async", "client", "action_chunk_received", t_s, chunk_index=chunk_index))
        client_events.append(_event("async", "client", "chunk_execution_started", t_s, chunk_index=chunk_index))
        t_s += chunk_duration_s
        executed_rows += action_horizon
        client_events.append(_event("async", "client", "chunk_execution_finished", t_s, chunk_index=chunk_index))

        observation_index = chunk_index + 1
        if observation_index <= num_chunks:
            client_events.append(_event("async", "client", "observation_sent", t_s, observation_index=observation_index))
            refresh_done_s = t_s + forward_latency_s
            next_chunk = chunk_index + 2
            if next_chunk <= num_chunks:
                server_events.append(
                    _event(
                        "async",
                        "server",
                        "forward_started",
                        t_s,
                        chunk_index=chunk_index,
                        prefix=f"O1_real..O{observation_index}_real",
                        send_action=False,
                    )
                )
                server_events.append(
                    _event(
                        "async",
                        "server",
                        "forward_started",
                        refresh_done_s,
                        chunk_index=next_chunk,
                        prefix=f"O1_real..O{observation_index}_real,O{observation_index + 1}_sim",
                        send_action=True,
                    )
                )
                ready_at[next_chunk] = refresh_done_s + forward_latency_s
                server_events.append(
                    _event("async", "server", "action_chunk_ready", ready_at[next_chunk], chunk_index=next_chunk)
                )

    return SimulationResult(
        mode="async",
        client_events=client_events,
        server_events=server_events,
        total_time_s=t_s,
        idle_time_s=idle_time_s,
        executed_rows=executed_rows,
        underrun_rows=underrun_rows,
    )


def _event(mode: str, stream: str, event: str, t_s: float, **data: Any) -> Event:
    return Event(mode=mode, stream=stream, event=event, t_s=round(t_s, 6), data=data)


def write_jsonl(path: Path, events: list[Event]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for event in sorted(events, key=lambda item: item.t_s):
            f.write(json.dumps(asdict(event), sort_keys=True) + "\n")


def summarize(sync_result: SimulationResult, async_result: SimulationResult, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "config": {
            "num_chunks": args.num_chunks,
            "action_horizon": args.action_horizon,
            "control_hz": args.control_hz,
            "chunk_duration_s": args.action_horizon / args.control_hz,
            "forward_latency_s": args.forward_latency_s,
        },
        "sync": _summary(sync_result),
        "async": _summary(async_result),
        "gain": {
            "idle_time_reduction_s": sync_result.idle_time_s - async_result.idle_time_s,
            "idle_time_reduction_ratio": (
                (sync_result.idle_time_s - async_result.idle_time_s) / sync_result.idle_time_s
                if sync_result.idle_time_s > 0
                else 0.0
            ),
        },
    }


def _summary(result: SimulationResult) -> dict[str, Any]:
    return {
        "total_time_s": result.total_time_s,
        "idle_time_s": result.idle_time_s,
        "effective_control_idle_ratio": result.effective_control_idle_ratio,
        "executed_rows": result.executed_rows,
        "underrun_rows": result.underrun_rows,
    }


def write_result_table(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "| Mode | Total time (s) | Idle time (s) | Idle ratio | Executed rows | Underrun rows |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("sync", "async"):
        item = summary[mode]
        lines.append(
            "| {mode} | {total:.3f} | {idle:.3f} | {ratio:.3f} | {executed} | {underrun} |".format(
                mode=mode,
                total=item["total_time_s"],
                idle=item["idle_time_s"],
                ratio=item["effective_control_idle_ratio"],
                executed=item["executed_rows"],
                underrun=item["underrun_rows"],
            )
        )
    gain = summary["gain"]
    lines.extend(
        [
            "",
            f"Idle-time reduction: {gain['idle_time_reduction_s']:.3f}s "
            f"({gain['idle_time_reduction_ratio']:.1%}).",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU mock benchmark for DreamZero W8 async control metrics.")
    parser.add_argument("--num-chunks", type=int, default=6)
    parser.add_argument("--forward-latency-s", type=float, default=0.9)
    parser.add_argument("--action-horizon", type=int, default=24)
    parser.add_argument("--control-hz", type=float, default=15.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dreamzero_async/mock"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chunk_duration_s = args.action_horizon / args.control_hz
    sync_result = simulate_sync(
        num_chunks=args.num_chunks,
        action_horizon=args.action_horizon,
        chunk_duration_s=chunk_duration_s,
        forward_latency_s=args.forward_latency_s,
    )
    async_result = simulate_async(
        num_chunks=args.num_chunks,
        action_horizon=args.action_horizon,
        chunk_duration_s=chunk_duration_s,
        forward_latency_s=args.forward_latency_s,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "client_events.jsonl", sync_result.client_events + async_result.client_events)
    write_jsonl(args.output_dir / "server_events.jsonl", sync_result.server_events + async_result.server_events)
    summary = summarize(sync_result, async_result, args)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_result_table(args.output_dir / "result_table.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
