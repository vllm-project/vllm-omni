#!/usr/bin/env python3
"""Generate a compact baseline report and checksum index from raw artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
from pathlib import Path
from typing import Any


def _load(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _metric(summary: dict[str, Any], name: str, statistic: str) -> Any:
    return summary.get(name, {}).get(statistic)


def _percentile_pair(summary: dict[str, Any], name: str) -> str:
    return f"{_format(_metric(summary, name, 'p50'))}/{_format(_metric(summary, name, 'p95'))}"


def _resource_peaks(path: Path) -> dict[str, int | None]:
    resources = _load(path)
    peak_aicore = None
    peak_hbm = None
    peak_host = None
    first_hbm = None
    last_hbm = None
    first_host = None
    last_host = None
    for sample in resources.get("samples", []):
        memory = sample.get("host_memory_bytes", {})
        total = memory.get("MemTotal")
        available = memory.get("MemAvailable")
        if total is not None and available is not None:
            used = total - available
            peak_host = used if peak_host is None else max(peak_host, used)
            first_host = used if first_host is None else first_host
            last_host = used

        output = sample.get("npu_smi", {}).get("stdout", "")
        sample_hbm = 0
        saw_chip = False
        for line in output.splitlines():
            fields = line.split("|")
            if len(fields) < 4 or not re.search(r"[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9]", fields[2]):
                continue
            values = [int(value) for value in re.findall(r"\d+", fields[3])]
            if len(values) < 5:
                continue
            saw_chip = True
            peak_aicore = values[0] if peak_aicore is None else max(peak_aicore, values[0])
            sample_hbm += values[-2]
        if saw_chip:
            peak_hbm = sample_hbm if peak_hbm is None else max(peak_hbm, sample_hbm)
            first_hbm = sample_hbm if first_hbm is None else first_hbm
            last_hbm = sample_hbm
    return {
        "aicore_percent": peak_aicore,
        "aggregate_hbm_mib": peak_hbm,
        "host_memory_bytes": peak_host,
        "aggregate_hbm_delta_mib": last_hbm - first_hbm if first_hbm is not None else None,
        "host_memory_delta_bytes": last_host - first_host if first_host is not None else None,
    }


def _benchmark_table(result: dict[str, Any], root: Path) -> list[str]:
    rows = [
        "| Mode | C | OK/Fail | First text p50/p95 (s) | First audio p50/p95 (s) | "
        "E2E p50/p95 (s) | Req/s | Audio s/s | Wall (s) | Peak AI Core | Peak HBM MiB | "
        "HBM delta MiB | Peak/delta host GiB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for config in result.get("configurations", []):
        summary = config.get("summary", {})
        mode = config.get("mode")
        concurrency = config.get("concurrency")
        resource_path = root / f"{mode}_c{concurrency}" / "resources.json"
        peaks = _resource_peaks(resource_path)
        host = peaks["host_memory_bytes"]
        host_gib = host / (1024**3) if host is not None else None
        host_delta = peaks["host_memory_delta_bytes"]
        host_delta_gib = host_delta / (1024**3) if host_delta is not None else None
        rows.append(
            "| "
            + " | ".join(
                [
                    str(mode),
                    str(concurrency),
                    f"{summary.get('successful_requests', 0)}/{summary.get('failed_requests', 0)}",
                    _percentile_pair(summary, "first_text_s"),
                    _percentile_pair(summary, "first_audio_s"),
                    _percentile_pair(summary, "e2e_s"),
                    _format(summary.get("request_throughput_per_s")),
                    _format(summary.get("audio_seconds_throughput")),
                    _format(summary.get("wall_time_s")),
                    _format(peaks["aicore_percent"], 0),
                    _format(peaks["aggregate_hbm_mib"], 0),
                    _format(peaks["aggregate_hbm_delta_mib"], 0),
                    f"{_format(host_gib)}/{_format(host_delta_gib)}",
                ]
            )
            + " |"
        )
    return rows


def _command(result: dict[str, Any]) -> str:
    return " ".join(shlex.quote(str(value)) for value in result.get("command", []))


def _write_manifest(root: Path, output: Path) -> None:
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == output:
            continue
        entries.append(f"{_sha256(path)}  {path.relative_to(root)}")
    output.write_text("\n".join(entries) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--smoke-results", type=Path, required=True)
    parser.add_argument("--benchmark-results", type=Path, required=True)
    parser.add_argument("--stability-results", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()

    environment = _load(args.environment)
    smoke = _load(args.smoke_results)
    benchmark = _load(args.benchmark_results)
    stability = _load(args.stability_results)
    gate = _load(args.gate)
    git = environment.get("git", {})
    head = git.get("head", {}).get("stdout", "UNKNOWN")
    status = git.get("status", {}).get("stdout", "")
    npu = environment.get("npu", {})
    cards = npu.get("cards", [])
    logical_chips = npu.get("logical_chips", [])

    lines = [
        "# MiniCPM-o 4.5 Ascend 910C Single-Card Baseline",
        "",
        f"- Captured at: {environment.get('captured_at', 'UNKNOWN')}",
        f"- Git SHA: `{head}`",
        f"- Git worktree: {'clean' if not status else 'dirty'}",
        "- Metric scope: local proxy; official dataset and scoring remain `UNRESOLVED`.",
        f"- Gate: {'PASS' if gate.get('passed') else 'FAIL'}",
        "",
        "## Hardware",
        "",
        "- Target: one physical Ascend 910C card with two logical chips.",
        f"- Detected physical cards: {npu.get('physical_card_count')}",
        f"- Detected card inventory: `{json.dumps(cards, sort_keys=True)}`",
        f"- Logical chips: `{json.dumps(logical_chips, sort_keys=True)}`",
        "- Deployment uses logical devices `0` and `1` on the same card.",
        "",
        "## Smoke Gate",
        "",
        "| Request | Input | Output | Result | Text chars | Audio chunks | PCM bytes |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for record in smoke.get("records", []):
        audio = record.get("audio", {})
        lines.append(
            f"| {record.get('request_name')} | {record.get('input_modality')} | {record.get('output_mode')} | "
            f"{'PASS' if record.get('success') else 'FAIL'} | {len(record.get('text', ''))} | "
            f"{audio.get('chunk_count', 0)} | {audio.get('pcm_bytes', 0)} |"
        )

    lines.extend(["", "## Performance", "", *_benchmark_table(benchmark, args.benchmark_results.parent)])
    lines.extend(["", "## Stability", "", *_benchmark_table(stability, args.stability_results.parent)])
    lines.extend(
        [
            "",
            "## Commands",
            "",
            f"- Benchmark: `{_command(benchmark)}`",
            f"- Stability: `{_command(stability)}`",
            "",
            "## Gate Result",
            "",
            f"- Passed: `{gate.get('passed')}`",
            f"- Failures: `{json.dumps(gate.get('failures', []))}`",
            "- Formal score: not reported.",
            "",
            "Raw artifacts are indexed by `artifact_manifest.sha256`.",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_manifest(args.artifact_root, args.manifest_output)
    print(args.output)
    print(args.manifest_output)


if __name__ == "__main__":
    main()
