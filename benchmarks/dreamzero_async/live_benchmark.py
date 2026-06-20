#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.dreamzero_async.compare_replays import summarize as summarize_pair
from benchmarks.dreamzero_async.compare_replays import write_result_table as write_pair_table


SYNC_CLIENT = REPO_ROOT / "examples" / "online_serving" / "dreamzero" / "openpi_client.py"
ASYNC_CLIENT = REPO_ROOT / "examples" / "online_serving" / "dreamzero" / "async_client.py"
DEFAULT_VIDEO_DIR = REPO_ROOT / "outputs" / "dreamzero" / "assets"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_command(command: list[str], *, cwd: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    started_s = time.monotonic()
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    finished_s = time.monotonic()
    _write_json(
        output_dir / "command.json",
        {
            "command": command,
            "cwd": str(cwd),
            "returncode": proc.returncode,
            "elapsed_s": round(finished_s - started_s, 6),
        },
    )
    (output_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(command)}")


def _sync_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    return [
        args.python,
        str(SYNC_CLIENT),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--video-dir",
        str(args.video_dir),
        "--num-chunks",
        str(args.num_chunks),
        "--control-hz",
        str(args.control_hz),
        "--output-dir",
        str(output_dir),
    ]


def _async_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    return [
        args.python,
        str(ASYNC_CLIENT),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--video-dir",
        str(args.video_dir),
        "--num-chunks",
        str(args.num_chunks),
        "--control-hz",
        str(args.control_hz),
        "--bootstrap-timeout-s",
        str(args.bootstrap_timeout_s),
        "--chunk-timeout-s",
        str(args.chunk_timeout_s),
        "--output-dir",
        str(output_dir),
    ]


def _run_pair(args: argparse.Namespace, *, run_dir: Path, run_index: int, warmup: bool) -> dict[str, Any]:
    sync_dir = run_dir / "sync_openpi"
    async_dir = run_dir / "dreamzero_async"
    if args.order == "sync-first":
        modes = ("sync", "async")
    else:
        modes = ("async", "sync")

    for mode in modes:
        if mode == "sync":
            _run_command(_sync_command(args, sync_dir), cwd=REPO_ROOT, output_dir=run_dir / "logs" / "sync_openpi")
        else:
            _run_command(_async_command(args, async_dir), cwd=REPO_ROOT, output_dir=run_dir / "logs" / "dreamzero_async")

    pair_summary = summarize_pair(
        _load_json(sync_dir / "summary.json"),
        _load_json(async_dir / "summary.json"),
        control_hz=args.control_hz,
    )
    pair_summary["run"] = {
        "index": run_index,
        "warmup": warmup,
        "order": args.order,
        "sync_dir": str(sync_dir),
        "async_dir": str(async_dir),
    }
    compare_dir = run_dir / "compare"
    _write_json(compare_dir / "summary.json", pair_summary)
    write_pair_table(compare_dir / "result_table.md", pair_summary)
    return pair_summary


def _mean(values: list[float]) -> float:
    return round(statistics.mean(values), 6) if values else 0.0


def _stdev(values: list[float]) -> float:
    return round(statistics.stdev(values), 6) if len(values) > 1 else 0.0


def summarize_suite(pairs: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    measured = [item for item in pairs if not item["run"]["warmup"]]
    speedups = [float(item["gain"]["speedup"]) for item in measured]
    saved = [float(item["gain"]["time_saved_s"]) for item in measured]
    sync_times = [float(item["sync_openpi"]["closed_loop_time_s"]) for item in measured]
    async_times = [float(item["dreamzero_async"]["closed_loop_time_s"]) for item in measured]
    async_underruns = [int(item["dreamzero_async"]["underruns"]) for item in measured]
    server_errors = [int(item["dreamzero_async"]["server_error_count"]) for item in measured]
    return {
        "config": {
            "host": args.host,
            "port": args.port,
            "num_chunks": args.num_chunks,
            "control_hz": args.control_hz,
            "order": args.order,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "video_dir": str(args.video_dir),
        },
        "runs": pairs,
        "summary": {
            "measured_repeats": len(measured),
            "sync_time_mean_s": _mean(sync_times),
            "sync_time_stdev_s": _stdev(sync_times),
            "async_time_mean_s": _mean(async_times),
            "async_time_stdev_s": _stdev(async_times),
            "time_saved_mean_s": _mean(saved),
            "time_saved_stdev_s": _stdev(saved),
            "speedup_mean": _mean(speedups),
            "speedup_stdev": _stdev(speedups),
            "async_underrun_total": sum(async_underruns),
            "async_server_error_total": sum(server_errors),
        },
    }


def write_suite_table(path: Path, suite: dict[str, Any]) -> None:
    summary = suite["summary"]
    config = suite["config"]
    lines = [
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Chunks per run | {config['num_chunks']} |",
        f"| Measured repeats | {summary['measured_repeats']} |",
        f"| Sync mean time (s) | {summary['sync_time_mean_s']:.3f} |",
        f"| Async mean time (s) | {summary['async_time_mean_s']:.3f} |",
        f"| Mean time saved (s) | {summary['time_saved_mean_s']:.3f} |",
        f"| Mean speedup | {summary['speedup_mean']:.3f}x |",
        f"| Async underrun rows | {summary['async_underrun_total']} |",
        f"| Async server errors | {summary['async_server_error_total']} |",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live DreamZero sync-vs-async replay benchmark pairs.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--num-chunks", type=int, default=2)
    parser.add_argument("--control-hz", type=float, default=15.0)
    parser.add_argument("--bootstrap-timeout-s", type=float, default=180.0)
    parser.add_argument("--chunk-timeout-s", type=float, default=90.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--order", choices=("async-first", "sync-first"), default="async-first")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "dreamzero_async" / "live_benchmark")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    total_runs = args.warmups + args.repeats
    for run_index in range(total_runs):
        run_dir = args.output_dir / f"run_{run_index + 1:03d}"
        pairs.append(_run_pair(args, run_dir=run_dir, run_index=run_index + 1, warmup=run_index < args.warmups))

    suite = summarize_suite(pairs, args)
    _write_json(args.output_dir / "summary.json", suite)
    write_suite_table(args.output_dir / "result_table.md", suite)
    print(json.dumps(suite["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
