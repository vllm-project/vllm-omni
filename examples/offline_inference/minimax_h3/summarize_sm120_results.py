#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Print concise metrics from MiniMax-H3 SM120 benchmark summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

SUCCESS_STATUSES = {"completed", "passed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="A summary.json file, case directory, or matrix result root.",
    )
    return parser.parse_args()


def find_summaries(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    direct = path / "summary.json"
    if direct.is_file():
        return [direct]
    return sorted(path.glob("*/summary.json"))


def external_peak_memory_gib(case_dir: Path) -> float | None:
    path = case_dir / "gpu_peak_memory.csv"
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    peaks = [float(row["peak_memory_mib"]) for row in rows]
    return max(peaks) / 1024 if peaks else None


def stage_seconds(stages: dict[str, Any], name: str) -> float:
    value = stages.get(name, 0.0)
    return float(value) if value is not None else 0.0


def print_summary(path: Path) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = data.get("tasks", [])
    expected = set(data.get("expected_tasks", []))
    actual = {task.get("task_id", task.get("task")) for task in tasks}
    missing = sorted(expected - actual)
    status = str(data.get("status", "unknown"))
    complete = status in SUCCESS_STATUSES and not missing

    print(f"\n===== {path.parent.name} =====")
    print(f"status:           {status}")
    print(f"parallelism:      {data.get('parallel_config', 'n/a')}")
    print(f"memory placement: {data.get('memory_placement', 'n/a')}")
    print(f"precision:        {data.get('precision', 'n/a')}")
    width = data.get("width", "?")
    height = data.get("height", "?")
    print(f"resolution:       {width}x{height}")
    print(f"duration:         {data.get('duration_seconds', 'n/a')} s")
    requested_steps = int(data.get("num_inference_steps", 0))
    print(f"requested steps:  {requested_steps or 'n/a'}")
    if missing:
        print(f"missing tasks:    {', '.join(missing)}")

    external_peak = external_peak_memory_gib(path.parent)
    if external_peak is not None:
        print(f"external peak:    {external_peak:.2f} GiB/GPU")

    executed_updates = max(requested_steps - 1, 1)
    for task in tasks:
        stages = task.get("stage_durations", {})
        diffuse = stage_seconds(stages, "MiniMaxH3Pipeline.diffuse")
        task_id = task.get("task_id", task.get("task", "unknown"))
        print(f"\n{task_id}")
        print(f"  E2E:       {float(task.get('wall_time_s', 0.0)):.3f} s")
        print(f"  encode:    {stage_seconds(stages, 'MiniMaxH3Pipeline.encode_prompt'):.3f} s")
        print(f"  condition: {stage_seconds(stages, 'MiniMaxH3Pipeline._encode_video_audio_conditions'):.3f} s")
        print(f"  denoise:   {diffuse:.3f} s")
        print(f"  decode:    {stage_seconds(stages, 'MiniMaxH3Pipeline.decode'):.3f} s")
        print(f"  per-step:  {diffuse * 1000 / executed_updates:.1f} ms")
        peak_memory_mb = float(task.get("worker_peak_memory_mb", 0.0))
        print(f"  memory:    {peak_memory_mb / 1024:.2f} GiB")
        print(f"  output:    {task.get('output', 'n/a')}")

    return complete


def main() -> None:
    args = parse_args()
    summaries: list[Path] = []
    for path in args.paths:
        found = find_summaries(path)
        if not found:
            raise SystemExit(f"No summary.json found below {path}")
        summaries.extend(found)

    all_complete = True
    for path in dict.fromkeys(summaries):
        all_complete = print_summary(path) and all_complete
    if not all_complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
