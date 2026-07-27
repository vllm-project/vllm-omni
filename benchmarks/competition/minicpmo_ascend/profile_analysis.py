#!/usr/bin/env python3
"""Summarize and compare exported torch_npu/CANN profiler artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def number(value: Any) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0


def rows(paths: Iterable[Path]) -> Iterable[dict[str, str]]:
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as stream:
            yield from csv.DictReader(stream)


def _files(root: Path, *patterns: str) -> list[Path]:
    result: set[Path] = set()
    for pattern in patterns:
        result.update(root.rglob(pattern))
    return sorted(result)


def _prefer_integrated(root: Path, exact_name: str, fallback_pattern: str) -> list[Path]:
    integrated = _files(root, exact_name)
    return integrated or _files(root, fallback_pattern)


def _top(mapping: dict[str, dict[str, float]], limit: int = 30) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "count": int(values["count"]),
            "total_us": round(values["total_us"], 3),
            "avg_us": round(values["total_us"] / values["count"], 3) if values["count"] else 0.0,
        }
        for name, values in sorted(mapping.items(), key=lambda item: item[1]["total_us"], reverse=True)[:limit]
    ]


def _aggregate(
    source_rows: Iterable[dict[str, str]],
    *,
    name_keys: tuple[str, ...],
    duration_keys: tuple[str, ...],
    count_key: str | None = None,
) -> tuple[float, int, list[dict[str, Any]]]:
    grouped: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0.0, "total_us": 0.0})
    total_us = 0.0
    total_count = 0
    for row in source_rows:
        name = next((row.get(key) for key in name_keys if row.get(key)), "unknown")
        duration = next((number(row.get(key)) for key in duration_keys if row.get(key) not in (None, "")), 0.0)
        count = int(number(row.get(count_key))) if count_key else 1
        grouped[name]["count"] += count
        grouped[name]["total_us"] += duration
        total_us += duration
        total_count += count
    return round(total_us, 3), total_count, _top(grouped)


def _capture_signature(capture: dict[str, Any] | None) -> dict[str, Any] | None:
    if capture is None:
        return None
    return {
        "profile_stages": capture.get("profile_stages"),
        "input_modality": capture.get("input_modality"),
        "output_mode": capture.get("output_mode"),
        "workload": capture.get("workload"),
    }


def analyze_trace_root(trace_root: Path, capture: dict[str, Any] | None = None) -> dict[str, Any]:
    trace_root = trace_root.resolve()
    op_stat_files = _prefer_integrated(trace_root, "op_statistic.csv", "op_statistic*.csv")
    api_stat_files = _prefer_integrated(trace_root, "api_statistic.csv", "api_statistic*.csv")
    kernel_files = _files(trace_root, "kernel_details.csv") or _files(trace_root, "op_summary*.csv")
    operator_files = _files(trace_root, "operator_details.csv")

    op_time, op_calls, top_op_types = _aggregate(
        rows(op_stat_files),
        name_keys=("OP Type", "Type", "Name"),
        duration_keys=("Total Time(us)", "Duration(us)"),
        count_key="Count",
    )
    api_time, api_calls, top_apis = _aggregate(
        rows(api_stat_files),
        name_keys=("API Name", "Name"),
        duration_keys=("Time(us)", "Total Time(us)", "Duration(us)"),
        count_key="Count",
    )
    api_levels: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0.0, "total_us": 0.0})
    for row in rows(api_stat_files):
        level = row.get("Level") or "unknown"
        api_levels[level]["count"] += int(number(row.get("Count")))
        api_levels[level]["total_us"] += number(row.get("Time(us)"))
    kernel_time, kernel_calls, top_kernels = _aggregate(
        rows(kernel_files),
        name_keys=("Name", "Op Name", "OP Type"),
        duration_keys=("Duration(us)", "Task Duration(us)"),
    )
    host_time, operator_calls, top_host_ops = _aggregate(
        rows(operator_files),
        name_keys=("Name",),
        duration_keys=("Host Self Duration(us)",),
    )
    device_time, _, top_device_ops = _aggregate(
        rows(operator_files),
        name_keys=("Name",),
        duration_keys=("Device Self Duration(us)",),
    )

    small_kernel_count = 0
    parsed_kernel_count = 0
    for row in rows(kernel_files):
        duration = number(row.get("Duration(us)") or row.get("Task Duration(us)"))
        parsed_kernel_count += 1
        small_kernel_count += duration <= 50.0

    exported_dirs = sorted({str(path.parent) for path in (*op_stat_files, *kernel_files, *operator_files)})
    traces = [str(path) for path in _files(trace_root, "trace_view.json", "msprof*.json")]
    result = {
        "schema_version": 1,
        "metric_scope": "profiler_diagnostic_not_score",
        "trace_root": str(trace_root),
        "capture": _capture_signature(capture),
        "exported_dirs": exported_dirs,
        "source_files": {
            "op_statistic": [str(path) for path in op_stat_files],
            "api_statistic": [str(path) for path in api_stat_files],
            "kernel_details": [str(path) for path in kernel_files],
            "operator_details": [str(path) for path in operator_files],
            "timelines": traces,
        },
        "operators": {"calls": op_calls, "total_us": op_time, "top": top_op_types},
        "apis": {
            "calls": api_calls,
            "total_us": api_time,
            "levels": {
                level: {"calls": int(values["count"]), "total_us": round(values["total_us"], 3)}
                for level, values in sorted(api_levels.items())
            },
            "top": top_apis,
        },
        "kernels": {
            "calls": kernel_calls,
            "total_us": kernel_time,
            "small_le_50us_calls": small_kernel_count,
            "small_le_50us_ratio": small_kernel_count / parsed_kernel_count if parsed_kernel_count else None,
            "top": top_kernels,
        },
        "torch_operators": {
            "calls": operator_calls,
            "host_self_total_us": host_time,
            "device_self_total_us": device_time,
            "top_host_self": top_host_ops,
            "top_device_self": top_device_ops,
        },
    }
    if not any((op_stat_files, api_stat_files, kernel_files, operator_files)):
        raise FileNotFoundError(f"no exported profiler CSV files under {trace_root}")
    return result


def write_analysis(result: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# MiniCPM-o Ascend Profile Analysis",
        "",
        "- Scope: profiler diagnostic; never use profiler timing as a score result.",
        f"- Trace root: `{result['trace_root']}`",
        f"- Export directories: {len(result['exported_dirs'])}",
        f"- Kernel calls: {result['kernels']['calls']}",
        f"- Aggregated kernel time: {result['kernels']['total_us'] / 1000:.3f} ms",
        f"- Aggregated API time across levels: {result['apis']['total_us'] / 1000:.3f} ms",
        "",
        "Aggregated device time may exceed wall time when streams overlap. "
        "API levels may be nested and are not additive.",
        "",
        "## Top Kernels",
        "",
        "| Kernel | Calls | Total ms | Avg us |",
        "|---|---:|---:|---:|",
    ]
    for item in result["kernels"]["top"][:20]:
        lines.append(f"| {item['name']} | {item['count']} | {item['total_us'] / 1000:.3f} | {item['avg_us']:.3f} |")
    lines += ["", "## Top Runtime APIs", "", "| API | Calls | Total ms | Avg us |", "|---|---:|---:|---:|"]
    for item in result["apis"]["top"][:20]:
        lines.append(f"| {item['name']} | {item['count']} | {item['total_us'] / 1000:.3f} | {item['avg_us']:.3f} |")
    lines += [
        "",
        "## Top Torch Operators by Device Self Time",
        "",
        "| Operator | Calls | Total ms | Avg us |",
        "|---|---:|---:|---:|",
    ]
    for item in result["torch_operators"]["top_device_self"][:20]:
        lines.append(f"| {item['name']} | {item['count']} | {item['total_us'] / 1000:.3f} | {item['avg_us']:.3f} |")
    lines += [
        "",
        "## Fragmentation Signal",
        "",
        f"- Kernels <= 50 us: {result['kernels']['small_le_50us_calls']} "
        f"({(result['kernels']['small_le_50us_ratio'] or 0) * 100:.2f}%).",
        "",
    ]
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def _metric_values(report: dict[str, Any]) -> dict[str, float]:
    return {
        "operator_time_us": report["operators"]["total_us"],
        "operator_calls": report["operators"]["calls"],
        "api_time_us": report["apis"]["total_us"],
        "api_calls": report["apis"]["calls"],
        "kernel_time_us": report["kernels"]["total_us"],
        "kernel_calls": report["kernels"]["calls"],
        "small_kernel_ratio": report["kernels"]["small_le_50us_ratio"] or 0.0,
    }


def compare_reports(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_values = _metric_values(before)
    after_values = _metric_values(after)
    metrics = {}
    for key, left in before_values.items():
        right = after_values[key]
        metrics[key] = {
            "before": left,
            "after": right,
            "delta": right - left,
            "percent": ((right - left) * 100 / left) if left else None,
        }
    left_capture = before.get("capture") or {}
    right_capture = after.get("capture") or {}
    signature_keys = ("profile_stages", "input_modality", "output_mode", "workload")
    mismatches = [
        {"field": key, "before": left_capture.get(key), "after": right_capture.get(key)}
        for key in signature_keys
        if left_capture.get(key) != right_capture.get(key)
    ]
    return {
        "schema_version": 1,
        "metric_scope": "same-preset profiler diagnostic comparison",
        "compatible": not mismatches,
        "mismatches": mismatches,
        "before": before.get("trace_root"),
        "after": after.get("trace_root"),
        "metrics": metrics,
    }


def write_comparison(result: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# MiniCPM-o Ascend Profile Comparison",
        "",
        "Use this only when workload, stages, profiler configuration, and environment match.",
        "",
    ]
    if not result["compatible"]:
        lines += ["Warning: capture signatures differ; timing deltas are invalid.", ""]
        for mismatch in result["mismatches"]:
            lines.append(f"- `{mismatch['field']}`: `{mismatch['before']}` -> `{mismatch['after']}`")
        lines.append("")
    lines += ["| Metric | Before | After | Delta | Change |", "|---|---:|---:|---:|---:|"]
    for name, item in result["metrics"].items():
        change = "N/A" if item["percent"] is None else f"{item['percent']:+.2f}%"
        lines.append(f"| {name} | {item['before']:.3f} | {item['after']:.3f} | {item['delta']:+.3f} | {change} |")
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(root: Path, output: Path) -> None:
    root = root.resolve()
    output = output.resolve()
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.resolve() != output):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        entries.append(f"{digest.hexdigest()}  {path.relative_to(root)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(entries) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("trace_root", type=Path)
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.add_argument("--capture", type=Path)
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("before", type=Path)
    compare_parser.add_argument("after", type=Path)
    compare_parser.add_argument("--output", type=Path, required=True)
    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("root", type=Path)
    manifest_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "analyze":
        capture = json.loads(args.capture.read_text(encoding="utf-8")) if args.capture else None
        result = analyze_trace_root(args.trace_root, capture=capture)
        write_analysis(result, args.output)
        print(args.output)
        return
    if args.command == "compare":
        before = json.loads(args.before.read_text(encoding="utf-8"))
        after = json.loads(args.after.read_text(encoding="utf-8"))
        result = compare_reports(before, after)
        write_comparison(result, args.output)
        print(args.output)
        if not result["compatible"]:
            raise SystemExit("profile captures are not comparable; see mismatches in output")
        return
    write_manifest(args.root, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
