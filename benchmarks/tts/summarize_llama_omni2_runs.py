#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

_METRICS = (
    "median_audio_ttfp_ms",
    "median_audio_rtf",
    "audio_throughput",
)


def _positive_finite(payload: dict[str, Any], path: Path, metric: str) -> float:
    value = payload.get(metric)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{path} has no positive {metric}")
    return float(value)


def _validate_run_payload(payload: dict[str, Any], path: Path) -> None:
    num_prompts = payload.get("num_prompts")
    completed = payload.get("completed")
    failed = payload.get("failed")
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in (num_prompts, completed, failed)):
        raise ValueError(f"{path} must report integer num_prompts, completed, and failed")
    if failed != 0:
        raise ValueError(f"{path} reported failed={failed}")
    if num_prompts <= 0 or completed != num_prompts:
        raise ValueError(f"{path} completed={completed}, expected num_prompts={num_prompts}")
    _positive_finite(payload, path, "total_audio_duration_s")
    _positive_finite(payload, path, "median_audio_duration_s")


def _load_metric_values(paths: list[Path], metric: str) -> list[float]:
    if len(paths) < 3:
        raise ValueError(f"LLaMA-Omni2 comparisons require at least three runs, got {len(paths)}")
    values: list[float] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        _validate_run_payload(payload, path)
        values.append(_positive_finite(payload, path, metric))
    return values


def _describe(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "stdev": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def summarize_comparison(
    *,
    before_paths: list[Path],
    after_paths: list[Path],
    label: str,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for metric in _METRICS:
        before = _describe(_load_metric_values(before_paths, metric))
        after = _describe(_load_metric_values(after_paths, metric))
        before_median = before["median"]
        if before_median == 0:
            raise ValueError(f"cannot compute relative change for zero {metric}")
        metrics[metric] = {
            "before": before,
            "after": after,
            "relative_change_percent": ((after["median"] - before_median) / before_median * 100.0),
        }
    return {
        "label": label,
        "before_files": [str(path) for path in before_paths],
        "after_files": [str(path) for path in after_paths],
        "metrics": metrics,
    }


def _relative_change(
    comparisons: dict[str, dict[str, Any]],
    label: str,
    metric: str,
) -> float:
    return float(comparisons[label]["metrics"][metric]["relative_change_percent"])


def _strictly_greater(value: float, threshold: float) -> bool:
    return value > threshold and not math.isclose(
        value,
        threshold,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


def _at_least(value: float, threshold: float) -> bool:
    return value > threshold or math.isclose(
        value,
        threshold,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


def _at_most(value: float, threshold: float) -> bool:
    return value < threshold or math.isclose(
        value,
        threshold,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


def evaluate_gate(
    comparisons: dict[str, dict[str, Any]],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if "c1" in comparisons:
        ttfp_change = _relative_change(
            comparisons,
            "c1",
            "median_audio_ttfp_ms",
        )
        rtf_change = _relative_change(
            comparisons,
            "c1",
            "median_audio_rtf",
        )
        if _strictly_greater(ttfp_change, 5.0):
            reasons.append(f"c1 median TTFP regressed {ttfp_change:.2f}% (> 5%)")
        if _strictly_greater(rtf_change, 5.0):
            reasons.append(f"c1 median RTF regressed {rtf_change:.2f}% (> 5%)")

    high_concurrency_labels = [label for label in ("c4", "c8") if label in comparisons]
    if high_concurrency_labels:
        high_concurrency_passed = any(
            _at_least(
                _relative_change(comparisons, label, "audio_throughput"),
                10.0,
            )
            or _at_most(
                _relative_change(comparisons, label, "median_audio_rtf"),
                -10.0,
            )
            for label in high_concurrency_labels
        )
        if not high_concurrency_passed:
            details = ", ".join(
                (
                    f"{label}: throughput "
                    f"{_relative_change(comparisons, label, 'audio_throughput'):.2f}%, "
                    f"RTF {_relative_change(comparisons, label, 'median_audio_rtf'):.2f}%"
                )
                for label in high_concurrency_labels
            )
            reasons.append(f"neither c4 nor c8 improved audio throughput or median RTF by at least 10% ({details})")

    return not reasons, reasons


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate controlled before/after LLaMA-Omni2 benchmark runs and enforce the c1/c4/c8 performance gate."
        )
    )
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument(
        "--before",
        action="append",
        nargs="+",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--after",
        action="append",
        nargs="+",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not (len(args.label) == len(args.before) == len(args.after)):
        raise SystemExit("each --label must have one matching --before and --after group")
    comparisons = {
        label: summarize_comparison(
            before_paths=before,
            after_paths=after,
            label=label,
        )
        for label, before, after in zip(
            args.label,
            args.before,
            args.after,
        )
    }
    passed, reasons = evaluate_gate(comparisons)
    report = {
        "passed": passed,
        "reasons": reasons,
        "comparisons": comparisons,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
