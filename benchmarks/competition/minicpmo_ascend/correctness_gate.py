#!/usr/bin/env python3
"""Produce a machine-readable pass/fail gate from smoke and benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path, label: str, failures: list[str]) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"cannot load {label} results from {path}: {exc}")
        return None


def _check_benchmark(path: Path, label: str, failures: list[str]) -> None:
    benchmark = _load(path, label, failures) or {}
    configurations = benchmark.get("configurations", [])
    if not configurations:
        failures.append(f"{label}: contains no configurations")
    for config in configurations:
        config_label = f"{label} {config.get('mode')} c={config.get('concurrency')}"
        if config.get("aborted"):
            failures.append(f"{config_label}: aborted during warmup")
            continue
        summary = config.get("summary", {})
        if summary.get("failed_requests", 1) != 0:
            failures.append(f"{config_label}: contains failed requests")
        if summary.get("successful_requests", 0) <= 0:
            failures.append(f"{config_label}: contains no successful samples")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-results", type=Path, required=True)
    parser.add_argument("--benchmark-results", type=Path)
    parser.add_argument("--stability-results", type=Path)
    parser.add_argument("--require-input-modalities", nargs="+", default=["text", "image", "audio", "video"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    failures = []
    smoke = _load(args.smoke_results, "smoke", failures) or {}
    records = smoke.get("records", [])
    seen_modalities = {record.get("input_modality") for record in records}
    missing = set(args.require_input_modalities) - seen_modalities
    if missing:
        failures.append(f"missing input modality smoke records: {sorted(missing)}")
    for record in records:
        name = record.get("request_name", "unknown")
        if not record.get("success") or not record.get("complete"):
            failures.append(f"{name}: incomplete or failed: {record.get('errors', [])}")
        audio = record.get("audio", {})
        if record.get("output_mode") == "text":
            if audio.get("chunk_count", 0) != 0:
                failures.append(f"{name}: text-only request unexpectedly produced audio")
        else:
            if audio.get("sample_rate_hz") != 24000 or audio.get("pcm_bytes", 0) <= 0:
                failures.append(f"{name}: invalid 24 kHz audio output")
            if audio.get("adjacent_duplicate_chunks", 0):
                failures.append(f"{name}: adjacent duplicate audio chunks")

    if args.benchmark_results:
        _check_benchmark(args.benchmark_results, "benchmark", failures)
    if args.stability_results:
        _check_benchmark(args.stability_results, "stability", failures)

    result = {
        "schema_version": 1,
        "passed": not failures,
        "failures": failures,
        "official_effect_gate": "UNRESOLVED",
        "scope": "local functional, streaming, and stability proxy",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
