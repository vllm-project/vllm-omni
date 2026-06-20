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


def summarize(sync: dict[str, Any], async_summary: dict[str, Any], *, control_hz: float) -> dict[str, Any]:
    sync_time = float(sync["total_closed_loop_time_s"])
    async_time = float(async_summary["total_elapsed_s"])
    async_execution_s = float(async_summary["executed_rows"]) / control_hz
    async_idle_proxy_s = max(0.0, async_time - async_execution_s)
    return {
        "config": {
            "control_hz": control_hz,
        },
        "sync_openpi": {
            "action_chunk_count": sync["action_chunk_count"],
            "executed_rows": sync["executed_rows"],
            "closed_loop_time_s": sync_time,
            "idle_time_s": sync["idle_time_s"],
            "idle_ratio": sync["effective_control_idle_ratio"],
        },
        "dreamzero_async": {
            "action_chunk_count": async_summary["action_chunk_count"],
            "executed_rows": async_summary["executed_rows"],
            "closed_loop_time_s": async_time,
            "idle_proxy_s": _round(async_idle_proxy_s),
            "bootstrap_latency_s": async_summary["bootstrap_latency_s"],
            "underruns": async_summary["underruns"],
            "server_error_count": async_summary["server_error_count"],
        },
        "gain": {
            "time_saved_s": _round(sync_time - async_time),
            "time_reduction_ratio": _round((sync_time - async_time) / sync_time) if sync_time > 0 else 0.0,
            "speedup": _round(sync_time / async_time) if async_time > 0 else 0.0,
        },
    }


def write_result_table(path: Path, summary: dict[str, Any]) -> None:
    sync = summary["sync_openpi"]
    async_item = summary["dreamzero_async"]
    gain = summary["gain"]
    lines = [
        "| Mode | Time (s) | Idle/proxy idle (s) | Action chunks | Executed rows | Underruns |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| sync_openpi | {sync['closed_loop_time_s']:.3f} | {sync['idle_time_s']:.3f} | "
            f"{sync['action_chunk_count']} | {sync['executed_rows']} | 0 |"
        ),
        (
            f"| dreamzero_async | {async_item['closed_loop_time_s']:.3f} | {async_item['idle_proxy_s']:.3f} | "
            f"{async_item['action_chunk_count']} | {async_item['executed_rows']} | {async_item['underruns']} |"
        ),
        "",
        f"Time saved: {gain['time_saved_s']:.3f}s ({gain['time_reduction_ratio']:.1%}); speedup: {gain['speedup']:.3f}x.",
    ]
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
