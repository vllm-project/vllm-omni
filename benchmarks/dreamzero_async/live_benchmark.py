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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
    command = [
        args.sync_python or args.python,
        str(SYNC_CLIENT),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--path",
        args.sync_path,
        "--video-dir",
        str(args.video_dir),
        "--num-chunks",
        str(args.num_chunks),
        "--control-hz",
        str(args.control_hz),
        "--output-dir",
        str(output_dir),
    ]
    if args.repeat_last_observation:
        command.append("--repeat-last-observation")
    return command


def _async_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        args.async_python or args.python,
        str(ASYNC_CLIENT),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--path",
        args.async_path,
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
    if args.repeat_last_observation:
        command.append("--repeat-last-observation")
    if args.realtime:
        command.append("--realtime")
    return command


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
        sync_events=_load_jsonl(sync_dir / "client_events.jsonl"),
        async_events=_load_jsonl(async_dir / "client_events.jsonl"),
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
    wait_speedups = [float(item["gain"]["exposed_wait_speedup"]) for item in measured]
    saved = [float(item["gain"]["time_saved_s"]) for item in measured]
    exposed_wait_saved = [float(item["gain"]["exposed_wait_saved_s"]) for item in measured]
    sync_times = [float(item["sync_openpi"]["closed_loop_time_s"]) for item in measured]
    async_times = [float(item["dreamzero_async"]["closed_loop_time_s"]) for item in measured]
    sync_waits = [float(item["sync_openpi"]["idle_time_s"]) for item in measured]
    async_waits = [float(item["dreamzero_async"]["idle_proxy_s"]) for item in measured]
    sync_avg_forwards = [
        float(item["sync_openpi"]["avg_request_latency_s"])
        for item in measured
        if item["sync_openpi"].get("avg_request_latency_s") is not None
    ]
    async_avg_forward_gaps = [
        float(item["dreamzero_async"]["avg_action_receive_gap_s"])
        for item in measured
        if item["dreamzero_async"].get("avg_action_receive_gap_s") is not None
    ]
    async_underruns = [int(item["dreamzero_async"]["underruns"]) for item in measured]
    async_deadline_misses = [int(item["dreamzero_async"].get("deadline_miss_count", 0)) for item in measured]
    async_non_sim_conditioned = [
        len(item["dreamzero_async"].get("non_sim_conditioned_post_bootstrap_chunks", [])) for item in measured
    ]
    server_errors = [int(item["dreamzero_async"]["server_error_count"]) for item in measured]
    valid_speedup_repeats = [item for item in measured if item["validity"]["speedup_claim_valid"]]
    return {
        "config": {
            "host": args.host,
            "port": args.port,
            "num_chunks": args.num_chunks,
            "sync_path": args.sync_path,
            "async_path": args.async_path,
            "control_hz": args.control_hz,
            "repeat_last_observation": args.repeat_last_observation,
            "realtime": args.realtime,
            "order": args.order,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "video_dir": str(args.video_dir),
            "python": args.python,
            "sync_python": args.sync_python,
            "async_python": args.async_python,
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
            "sync_exposed_wait_mean_s": _mean(sync_waits),
            "async_exposed_wait_mean_s": _mean(async_waits),
            "exposed_wait_saved_mean_s": _mean(exposed_wait_saved),
            "exposed_wait_speedup_mean": _mean(wait_speedups),
            "sync_avg_request_latency_mean_s": _mean(sync_avg_forwards),
            "async_avg_action_receive_gap_mean_s": _mean(async_avg_forward_gaps),
            "speedup_mean": _mean(speedups),
            "closed_loop_speedup_mean": _mean(speedups),
            "speedup_stdev": _stdev(speedups),
            "async_underrun_total": sum(async_underruns),
            "async_deadline_miss_total": sum(async_deadline_misses),
            "async_non_sim_conditioned_post_bootstrap_total": sum(async_non_sim_conditioned),
            "async_server_error_total": sum(server_errors),
            "valid_speedup_repeats": len(valid_speedup_repeats),
            "invalid_speedup_repeats": len(measured) - len(valid_speedup_repeats),
            "speedup_claim_valid": len(measured) > 0 and len(valid_speedup_repeats) == len(measured),
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
        f"| Closed-loop speedup | {summary['closed_loop_speedup_mean']:.3f}x |",
        f"| Sync exposed wait mean (s) | {summary['sync_exposed_wait_mean_s']:.3f} |",
        f"| Async exposed wait mean (s) | {summary['async_exposed_wait_mean_s']:.3f} |",
        f"| Exposed wait saved mean (s) | {summary['exposed_wait_saved_mean_s']:.3f} |",
        f"| Exposed wait speedup | {summary['exposed_wait_speedup_mean']:.3f}x |",
        f"| Sync avg request latency (s) | {summary['sync_avg_request_latency_mean_s']:.3f} |",
        f"| Async avg action receive gap (s) | {summary['async_avg_action_receive_gap_mean_s']:.3f} |",
        f"| Speedup claim valid | {summary['speedup_claim_valid']} |",
        f"| Valid speedup repeats | {summary['valid_speedup_repeats']} |",
        f"| Invalid speedup repeats | {summary['invalid_speedup_repeats']} |",
        f"| Async underrun rows | {summary['async_underrun_total']} |",
        f"| Async deadline misses | {summary['async_deadline_miss_total']} |",
        f"| Async non-sim post-bootstrap chunks | {summary['async_non_sim_conditioned_post_bootstrap_total']} |",
        f"| Async server errors | {summary['async_server_error_total']} |",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _speedup_requirement_exit_code(suite: dict[str, Any], *, require_valid: bool) -> int:
    if require_valid and not suite["summary"]["speedup_claim_valid"]:
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live DreamZero sync-vs-async replay benchmark pairs.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--sync-path", default="/v1/realtime/robot/openpi")
    parser.add_argument("--async-path", default="/v1/realtime/robot/dreamzero-async")
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--num-chunks", type=int, default=2)
    parser.add_argument("--control-hz", type=float, default=15.0)
    parser.add_argument("--repeat-last-observation", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--bootstrap-timeout-s", type=float, default=180.0)
    parser.add_argument("--chunk-timeout-s", type=float, default=10.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--order", choices=("async-first", "sync-first"), default="async-first")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--sync-python", default=None)
    parser.add_argument("--async-python", default=None)
    parser.add_argument(
        "--require-valid-speedup",
        action="store_true",
        help=(
            "Exit nonzero unless every measured repeat has zero async underruns, "
            "zero async server errors, and the same execution coverage as sync."
        ),
    )
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
    exit_code = _speedup_requirement_exit_code(suite, require_valid=args.require_valid_speedup)
    if exit_code:
        print(
            "Invalid speedup claim: async run violated the closed-loop benchmark contract "
            "(for example underruns, server errors, non-sim-conditioned post-bootstrap chunks, "
            "different execution coverage, or async not faster than sync). See summary.json for details.",
            file=sys.stderr,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
