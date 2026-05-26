# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

PROFILE_RE = re.compile(
    r"\[DiffusionBatchProfiler\]\s+batch_size=(?P<batch_size>\d+)\s+"
    r"request_compute_units=(?P<request_compute_units>\d+)\s+"
    r"total_compute_units=(?P<total_compute_units>\d+)\s+"
    r"denoise_step_time=(?P<denoise_step_time>[\d.]+)"
)


def parse_profile_records(log_file: Path) -> list[dict[str, int | float]]:
    records: list[dict[str, int | float]] = []
    with log_file.open("r", encoding="utf-8") as f:
        for line in f:
            match = PROFILE_RE.search(line)
            if match is None:
                continue
            records.append(
                {
                    "batch_size": int(match.group("batch_size")),
                    "request_compute_units": int(match.group("request_compute_units")),
                    "total_compute_units": int(match.group("total_compute_units")),
                    "denoise_step_time": float(match.group("denoise_step_time")),
                }
            )
    return records


def _iqr_filter(values: list[float]) -> list[float]:
    if len(values) < 4:
        return values

    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    lower_half = sorted_values[:midpoint]
    upper_half = sorted_values[-midpoint:]
    q1 = median(lower_half)
    q3 = median(upper_half)
    iqr = q3 - q1
    if iqr == 0:
        return values

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered = [value for value in values if lower_bound <= value <= upper_bound]
    return filtered or values


def _linear_fit(points: list[tuple[int, float]]) -> tuple[float, float, float]:
    xs = [float(point[0]) for point in points]
    ys = [point[1] for point in points]
    x_mean = mean(xs)
    y_mean = mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        slope = 0.0
    else:
        slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    intercept = y_mean - slope * x_mean
    sse = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    return slope, intercept, sse


def _piecewise_fit(
    points: list[tuple[int, float]],
    min_segment_points: int,
) -> dict[str, float | int]:
    min_segment_points = max(2, min_segment_points)
    if len(points) < min_segment_points * 2:
        raise ValueError(
            "Need at least "
            f"{min_segment_points * 2} profiled batch sizes for piecewise fitting, "
            f"got {len(points)}."
        )

    best_fit: dict[str, float | int] | None = None
    for split_index in range(min_segment_points, len(points) - min_segment_points + 1):
        left_points = points[:split_index]
        right_points = points[split_index:]
        left_slope, left_intercept, left_sse = _linear_fit(left_points)
        right_slope, right_intercept, right_sse = _linear_fit(right_points)
        error = left_sse + right_sse
        if best_fit is None or error < best_fit["sse"]:
            best_fit = {
                "split_index": split_index,
                "split_batch_size": points[split_index][0],
                "left_slope": left_slope,
                "left_intercept": left_intercept,
                "right_slope": right_slope,
                "right_intercept": right_intercept,
                "sse": error,
            }

    assert best_fit is not None
    denominator = float(best_fit["left_slope"]) - float(best_fit["right_slope"])
    if abs(denominator) > 1e-12:
        intersection = (float(best_fit["right_intercept"]) - float(best_fit["left_intercept"])) / denominator
        if math.isfinite(intersection):
            best_fit["intersection_batch_size"] = intersection
    return best_fit


def choose_sweet_spot_batch(
    records: list[dict[str, int | float]],
    reference_compute_units: int | None,
    min_samples_per_batch: int,
    min_segment_points: int,
) -> tuple[int, int, dict[str, object]]:
    if not records:
        raise ValueError("No DiffusionBatchProfiler records found in log file.")

    by_units: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        units = int(record["request_compute_units"])
        batch_size = int(record["batch_size"])
        by_units[units][batch_size].append(float(record["denoise_step_time"]))

    ref_units = reference_compute_units or min(by_units)
    if ref_units not in by_units:
        raise ValueError(
            f"reference_compute_units={ref_units} not found in logs; "
            f"available values: {sorted(by_units)}"
        )

    points: list[tuple[int, float]] = []
    for batch_size, values in by_units[ref_units].items():
        if len(values) < min_samples_per_batch:
            continue
        filtered_values = _iqr_filter(values)
        points.append((batch_size, median(filtered_values)))
    points.sort()
    if not points:
        raise ValueError("Need enough samples for at least one profiled batch size.")

    if len(points) >= max(4, min_segment_points * 2):
        fit = _piecewise_fit(points, min_segment_points)
        min_batch = points[0][0]
        max_batch = points[-1][0]
        intersection = fit.get("intersection_batch_size")
        if isinstance(intersection, float) and min_batch <= intersection <= max_batch:
            sweet_spot_batch = round(intersection)
        else:
            sweet_spot_batch = int(fit["split_batch_size"])
        sweet_spot_batch = max(min_batch, min(max_batch, int(sweet_spot_batch)))
        analysis = {
            "method": "piecewise_linear_fit",
            "points": [
                {
                    "batch_size": batch_size,
                    "median_denoise_step_time": median_time,
                    "median_time_per_request": median_time / batch_size,
                }
                for batch_size, median_time in points
            ],
            "fit": fit,
        }
        return ref_units, sweet_spot_batch, analysis

    best_point = min(points, key=lambda point: point[1] / point[0])
    analysis = {
        "method": "best_per_request_fallback",
        "points": [
            {
                "batch_size": batch_size,
                "median_denoise_step_time": median_time,
                "median_time_per_request": median_time / batch_size,
            }
            for batch_size, median_time in points
        ],
    }
    return ref_units, best_point[0], analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate diffusion compute-budget batching config from profiler logs."
    )
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument(
        "--reference-compute-units",
        type=int,
        default=None,
        help=(
            "Compute units for the profiled 512x512 reference request. "
            "Defaults to the smallest value in logs."
        ),
    )
    parser.add_argument(
        "--min-samples-per-batch",
        type=int,
        default=2,
        help="Minimum profiler records required for one batch size.",
    )
    parser.add_argument(
        "--min-segment-points",
        type=int,
        default=2,
        help="Minimum points per segment for the piecewise linear fit.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    records = parse_profile_records(args.log_file)
    ref_units, sweet_spot_batch, analysis = choose_sweet_spot_batch(
        records,
        args.reference_compute_units,
        args.min_samples_per_batch,
        args.min_segment_points,
    )
    compute_unit_budget = ref_units * sweet_spot_batch
    output = {
        "reference_compute_units": ref_units,
        "sweet_spot_batch_size": sweet_spot_batch,
        "analysis": analysis,
        "diffusion_batching_config": {
            "compute_unit_budget": compute_unit_budget,
        },
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
