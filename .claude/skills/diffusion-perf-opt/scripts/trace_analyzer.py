#!/usr/bin/env python3
"""Summarize torch profiler Chrome traces for vLLM Omni diffusion optimization."""

from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path
from typing import Any

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
CPU_CATS = {"python_function", "user_annotation", "cpu_op", "cuda_runtime", "cuda_driver"}


def open_trace(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("rt")


def event_name(event: dict[str, Any]) -> str:
    return str(event.get("name", ""))


def summarize_trace(path: Path, min_gap_us: float, topn: int) -> None:
    with open_trace(path) as f:
        data = json.load(f)

    events = data if isinstance(data, list) else data.get("traceEvents", [])
    gpu: list[tuple[float, float, float, str, str, Any, Any]] = []
    cpu: list[tuple[float, float, float, str, str, Any, Any]] = []
    by_gpu_name: dict[str, list[float]] = collections.defaultdict(lambda: [0, 0.0, 0.0])
    by_name: dict[str, list[float]] = collections.defaultdict(lambda: [0, 0.0, 0.0])
    nccl: dict[tuple[str, str], list[float]] = collections.defaultdict(lambda: [0, 0.0, 0.0])

    for event in events:
        dur = event.get("dur")
        ts = event.get("ts")
        if dur is None or ts is None or dur <= 0:
            continue
        cat = str(event.get("cat", ""))
        name = event_name(event)
        row = (float(ts), float(ts + dur), float(dur), name, cat, event.get("pid"), event.get("tid"))
        stat = by_name[name]
        stat[0] += 1
        stat[1] += dur
        stat[2] = max(stat[2], dur)
        if cat in GPU_CATS:
            gpu.append(row)
            gstat = by_gpu_name[name]
            gstat[0] += 1
            gstat[1] += dur
            gstat[2] = max(gstat[2], dur)
        elif cat in CPU_CATS:
            cpu.append(row)
        if "nccl" in name.lower():
            nstat = nccl[(cat, name)]
            nstat[0] += 1
            nstat[1] += dur
            nstat[2] = max(nstat[2], dur)

    print(f"\n== {path}")
    print(f"events={len(events)} gpu_events={len(gpu)} cpu_events={len(cpu)}")
    if not gpu:
        print("No GPU events found.")
        return

    gpu.sort()
    merged: list[list[Any]] = []
    for start, end, dur, name, cat, pid, tid in gpu:
        if not merged or start > merged[-1][1]:
            merged.append([start, end, [(start, end, dur, name, cat, pid, tid)]])
        else:
            merged[-1][1] = max(merged[-1][1], end)
            merged[-1][2].append((start, end, dur, name, cat, pid, tid))

    span = max(end for _, end, *_ in gpu) - min(start for start, *_ in gpu)
    busy = sum(end - start for start, end, _ in merged)
    idle = span - busy
    print(
        f"gpu_span_s={span / 1e6:.3f} "
        f"busy_union_s={busy / 1e6:.3f} "
        f"idle_union_s={idle / 1e6:.3f} "
        f"idle_pct={idle / span * 100:.2f}"
    )

    interesting_cpu = [
        row
        for row in cpu
        if row[2] >= 1000
        and (
            row[4] in {"python_function", "user_annotation"}
            or "cudaStreamSynchronize" in row[3]
            or "cudaDeviceSynchronize" in row[3]
            or "cudaLaunch" in row[3]
            or "cudaMemcpy" in row[3]
        )
    ]

    gaps = []
    for idx in range(1, len(merged)):
        gap_start = merged[idx - 1][1]
        gap_end = merged[idx][0]
        gap_dur = gap_end - gap_start
        if gap_dur >= min_gap_us:
            prev_event = max(merged[idx - 1][2], key=lambda x: x[1])
            next_event = min(merged[idx][2], key=lambda x: x[0])
            mid = (gap_start + gap_end) / 2
            containers = [row for row in interesting_cpu if row[0] <= mid <= row[1]]
            containers = sorted(containers, key=lambda x: x[2])[:8]
            gaps.append((gap_dur, gap_start, gap_end, prev_event, next_event, containers))

    print(f"gaps_ge_{min_gap_us / 1000:.3f}ms count={len(gaps)} sum_s={sum(g[0] for g in gaps) / 1e6:.3f}")
    for gap_dur, gap_start, gap_end, prev_event, next_event, containers in sorted(gaps, reverse=True)[:topn]:
        print(f"\nGAP {gap_dur / 1000:.3f} ms ts={gap_start:.0f}->{gap_end:.0f}")
        print(f"  prev {prev_event[4]} {prev_event[2] / 1000:.3f} ms {prev_event[3][:160]}")
        print(f"  next {next_event[4]} {next_event[2] / 1000:.3f} ms {next_event[3][:160]}")
        for row in containers:
            print(f"  in   {row[4]} {row[2] / 1000:.3f} ms {row[3][:180]}")

    print("\nTop GPU/operator events by total duration:")
    for name, (count, total, max_dur) in sorted(by_gpu_name.items(), key=lambda kv: kv[1][1], reverse=True)[:topn]:
        print(f"  {int(count):8d} total={total / 1e6:9.3f}s max={max_dur / 1000:9.3f}ms {name[:180]}")

    print("\nTop NCCL-like events by category:")
    for (cat, name), (count, total, max_dur) in sorted(nccl.items(), key=lambda kv: kv[1][1], reverse=True)[:topn]:
        print(f"  {int(count):8d} total={total / 1e6:9.3f}s max={max_dur / 1000:9.3f}ms cat={cat} {name[:160]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path, help="trace.json or trace.json.gz files")
    parser.add_argument("--min-gap-ms", type=float, default=5.0, help="minimum GPU idle gap to print")
    parser.add_argument("--topn", type=int, default=20, help="number of gaps/hotspots to print")
    args = parser.parse_args()

    for trace in args.traces:
        summarize_trace(trace, args.min_gap_ms * 1000.0, args.topn)


if __name__ == "__main__":
    main()
