#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _round(value: float) -> float:
    return round(float(value), 6)


def _validity(sync: dict[str, Any], async_summary: dict[str, Any]) -> dict[str, Any]:
    reasons = []
    config = async_summary.get("config", {})
    if config and not bool(config.get("realtime", False)):
        reasons.append("async replay did not use realtime control timing")
    if int(async_summary["server_error_count"]) != 0:
        reasons.append("async server reported errors")
    if int(async_summary["underruns"]) != 0:
        reasons.append("async client reported underruns")
    if int(async_summary["executed_rows"]) != int(sync["executed_rows"]):
        reasons.append("async executed row count differs from sync")
    missing_chunks = async_summary.get("missing_chunk_indices") or []
    if missing_chunks:
        reasons.append(f"async missed required chunks: {missing_chunks}")
    executed_chunks = async_summary.get("executed_chunk_indices") or []
    if executed_chunks and len(executed_chunks) != int(sync["action_chunk_count"]):
        reasons.append("async executed chunk count differs from sync")
    if float(async_summary["total_elapsed_s"]) >= float(sync["total_closed_loop_time_s"]):
        reasons.append("async closed-loop time is not faster than sync")
    non_sim_conditioned = async_summary.get("non_sim_conditioned_post_bootstrap_chunks")
    if non_sim_conditioned:
        reasons.append(f"post-bootstrap chunks were not sim-conditioned: {non_sim_conditioned}")
    return {
        "speedup_claim_valid": not reasons,
        "reason": "valid" if not reasons else "; ".join(reasons),
    }


def _mean(values: list[float]) -> float | None:
    return _round(sum(values) / len(values)) if values else None


def _sync_request_latencies(events: list[dict[str, Any]] | None) -> list[float]:
    return [
        float(event["data"]["latency_s"])
        for event in events or []
        if event.get("event") == "action_chunk_received" and "latency_s" in event.get("data", {})
    ]


def _async_receive_gaps(events: list[dict[str, Any]] | None) -> list[float]:
    receive_times = [
        float(event["t_s"])
        for event in events or []
        if event.get("event") == "action_chunk_received" and "t_s" in event
    ]
    return [receive_times[index] - receive_times[index - 1] for index in range(1, len(receive_times))]


def summarize(
    sync: dict[str, Any],
    async_summary: dict[str, Any],
    *,
    control_hz: float,
    sync_events: list[dict[str, Any]] | None = None,
    async_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    sync_time = float(sync["total_closed_loop_time_s"])
    async_time = float(async_summary["total_elapsed_s"])
    async_execution_s = float(async_summary["executed_rows"]) / control_hz
    sync_idle_s = float(sync["idle_time_s"])
    async_idle_proxy_s = max(0.0, async_time - async_execution_s)
    wait_speedup = sync_idle_s / async_idle_proxy_s if async_idle_proxy_s > 0 else 0.0
    closed_loop_speedup = sync_time / async_time if async_time > 0 else 0.0
    sync_latencies = _sync_request_latencies(sync_events)
    async_gaps = _async_receive_gaps(async_events)
    return {
        "config": {
            "control_hz": control_hz,
            "action_execution_s": _round(async_execution_s),
        },
        "sync_openpi": {
            "action_chunk_count": sync["action_chunk_count"],
            "executed_rows": sync["executed_rows"],
            "closed_loop_time_s": sync_time,
            "idle_time_s": sync_idle_s,
            "idle_ratio": sync["effective_control_idle_ratio"],
            "avg_request_latency_s": _mean(sync_latencies),
        },
        "dreamzero_async": {
            "action_chunk_count": async_summary["action_chunk_count"],
            "executed_rows": async_summary["executed_rows"],
            "closed_loop_time_s": async_time,
            "idle_proxy_s": _round(async_idle_proxy_s),
            "avg_action_receive_gap_s": _mean(async_gaps),
            "bootstrap_latency_s": async_summary["bootstrap_latency_s"],
            "underruns": async_summary["underruns"],
            "deadline_miss_count": async_summary.get("deadline_miss_count", 0),
            "sim_conditioned_post_bootstrap_chunks": async_summary.get(
                "sim_conditioned_post_bootstrap_chunks", []
            ),
            "non_sim_conditioned_post_bootstrap_chunks": async_summary.get(
                "non_sim_conditioned_post_bootstrap_chunks", []
            ),
            "server_error_count": async_summary["server_error_count"],
        },
        "gain": {
            "time_saved_s": _round(sync_time - async_time),
            "time_reduction_ratio": _round((sync_time - async_time) / sync_time) if sync_time > 0 else 0.0,
            "closed_loop_speedup": _round(closed_loop_speedup),
            "exposed_wait_saved_s": _round(sync_idle_s - async_idle_proxy_s),
            "exposed_wait_speedup": _round(wait_speedup),
            # Backward-compatible alias for older result readers.
            "speedup": _round(closed_loop_speedup),
        },
        "validity": _validity(sync, async_summary),
    }


def write_result_table(path: Path, summary: dict[str, Any]) -> None:
    sync = summary["sync_openpi"]
    async_item = summary["dreamzero_async"]
    gain = summary["gain"]
    config = summary["config"]
    sync_avg = sync.get("avg_request_latency_s")
    async_gap = async_item.get("avg_action_receive_gap_s")
    lines = [
        "| Mode | Closed-loop time (s) | Exposed wait (s) | Avg forward/gap (s) | Action execution (s) | Received chunks | Rows | Underruns |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| sync_openpi | {sync['closed_loop_time_s']:.3f} | {sync['idle_time_s']:.3f} | "
            f"{'n/a' if sync_avg is None else f'{sync_avg:.3f}'} | {config['action_execution_s']:.3f} | "
            f"{sync['action_chunk_count']} | {sync['executed_rows']} | 0 |"
        ),
        (
            f"| dreamzero_async | {async_item['closed_loop_time_s']:.3f} | {async_item['idle_proxy_s']:.3f} | "
            f"{'n/a' if async_gap is None else f'{async_gap:.3f}'} | {config['action_execution_s']:.3f} | "
            f"{async_item['action_chunk_count']} | {async_item['executed_rows']} | {async_item['underruns']} |"
        ),
        "",
        (
            "Async post-bootstrap sim-conditioned chunks: "
            f"{async_item['sim_conditioned_post_bootstrap_chunks']}; "
            "non-sim-conditioned: "
            f"{async_item['non_sim_conditioned_post_bootstrap_chunks']}."
        ),
        (
            f"Closed-loop time saved: {gain['time_saved_s']:.3f}s "
            f"({gain['time_reduction_ratio']:.1%}); closed-loop speedup: {gain['closed_loop_speedup']:.3f}x."
        ),
        (
            f"Exposed wait saved: {gain['exposed_wait_saved_s']:.3f}s; "
            f"exposed wait speedup: {gain['exposed_wait_speedup']:.3f}x."
        ),
    ]
    validity = summary["validity"]
    if validity["speedup_claim_valid"]:
        lines.append("Speedup claim status: valid.")
    else:
        lines.append(f"Speedup claim status: invalid as speedup proof ({validity['reason']}).")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare DreamZero sync OpenPI and async replay summaries.")
    parser.add_argument("--sync-summary", type=Path, required=True)
    parser.add_argument("--async-summary", type=Path, required=True)
    parser.add_argument("--control-hz", type=float, default=15.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarize(
        _load_json(args.sync_summary),
        _load_json(args.async_summary),
        control_hz=args.control_hz,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_result_table(args.output_dir / "result_table.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
