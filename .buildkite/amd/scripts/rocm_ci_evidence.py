#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Capture deterministic evidence for ROCm GPU CI jobs."""

from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def capture_environment(output: Path) -> None:
    import torch

    expected = int(os.environ.get("VLLM_CI_EXPECTED_GPU_COUNT", "1"))
    actual = torch.accelerator.device_count()
    if not torch.version.hip:
        raise RuntimeError("PyTorch is not a ROCm build")
    if actual != expected:
        raise RuntimeError(f"expected {expected} ROCm GPU(s), found {actual}")

    created = _parse_timestamp(os.environ.get("BUILDKITE_BUILD_CREATED_AT"))
    started = _parse_timestamp(os.environ.get("BUILDKITE_JOB_STARTED_AT"))
    queue_delay = "unknown"
    if created is not None and started is not None:
        queue_delay = f"{max(0.0, (started - created).total_seconds()):.3f}"

    hf_home = os.environ.get("HF_HOME", "")
    try:
        mount = subprocess.run(
            ["findmnt", "-n", "-T", hf_home, "-o", "TARGET,SOURCE,FSTYPE"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except FileNotFoundError:
        mount = ""
    mount_target = mount.split(maxsplit=1)[0] if mount else ""
    cache_mode = "mounted" if mount_target and mount_target != "/" else "root-filesystem"

    lines = [
        f"recorded_at_epoch={time.time():.6f}",
        f"platform={platform.platform()}",
        f"python={platform.python_version()}",
        f"pytorch={torch.__version__}",
        f"rocm={torch.version.hip}",
        f"visible_gpus={actual}",
        f"expected_gpus={expected}",
        f"hip_visible_devices={os.environ.get('HIP_VISIBLE_DEVICES', 'unset')}",
        f"rocr_visible_devices={os.environ.get('ROCR_VISIBLE_DEVICES', 'unset')}",
        f"image=rocm/vllm-omni:{os.environ.get('BUILDKITE_COMMIT', 'unknown')}",
        f"build_created_at={os.environ.get('BUILDKITE_BUILD_CREATED_AT', 'unknown')}",
        f"job_started_at={os.environ.get('BUILDKITE_JOB_STARTED_AT', 'unknown')}",
        f"queue_delay_seconds={queue_delay}",
        f"hf_home={hf_home}",
        f"hf_mount={mount or 'unknown'}",
        f"hf_cache_mode={cache_mode}",
    ]
    for index in range(actual):
        properties = torch.cuda.get_device_properties(index)
        lines.extend(
            [
                f"gpu_{index}_name={torch.cuda.get_device_name(index)}",
                f"gpu_{index}_total_memory_bytes={properties.total_memory}",
            ]
        )
    _write(output, "\n".join(lines) + "\n")


def capture_processes(output: Path) -> None:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,lstart=,stat=,comm="],
        check=True,
        capture_output=True,
        text=True,
    )
    _write(output, result.stdout)


def _process_identities(text: str, ignored_pids: set[int] | None = None) -> dict[str, str]:
    ignored_pids = ignored_pids or set()
    identities: dict[str, str] = {}
    for line in text.splitlines():
        fields = line.split(None, 8)
        if len(fields) != 9 or fields[8] == "ps" or int(fields[0]) in ignored_pids:
            continue
        identity = "|".join((fields[0], *fields[2:7]))
        identities[identity] = line
    return identities


def check_cleanup(before: Path, after: Path, output: Path, settle_seconds: float) -> None:
    time.sleep(settle_seconds)
    capture_processes(after)
    baseline = _process_identities(before.read_text(encoding="utf-8"))
    current = _process_identities(after.read_text(encoding="utf-8"), ignored_pids={os.getpid()})
    leaked = [line for identity, line in current.items() if identity not in baseline]
    status = "status=PASS leaked_processes=0" if not leaked else "status=FAIL leaked_processes_detected=1"
    _write(output, "\n".join([*leaked, status]) + "\n")
    if leaked:
        raise RuntimeError("processes created by the test remain after teardown")


def _junit_counts(paths: list[Path]) -> dict[str, int]:
    totals = {"selected": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    for path in paths:
        report = ElementTree.parse(path).getroot()
        suites = [report] if report.tag == "testsuite" else report.findall("./testsuite")
        tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
        failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
        errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
        skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
        totals["selected"] += tests
        totals["passed"] += tests - failures - errors - skipped
        totals["failed"] += failures + errors
        totals["skipped"] += skipped
        totals["errors"] += errors
    return totals


def summarize_pytest(xml_paths: list[Path], log: Path, output: Path) -> None:
    totals = _junit_counts(xml_paths)
    log_text = log.read_text(encoding="utf-8", errors="replace")
    deselected_matches = [int(value) for value in re.findall(r"(\d+) deselected", log_text)]
    deselected = max(deselected_matches, default=0)
    collected = totals["selected"] + deselected
    executed = totals["passed"] + totals["failed"]
    result = (
        f"collected={collected} selected={totals['selected']} passed={totals['passed']} "
        f"failed={totals['failed']} skipped={totals['skipped']} deselected={deselected} "
        f"errors={totals['errors']} executed={executed}"
    )
    _write(output, result + "\n")
    print(result)
    if collected == 0 or totals["selected"] == 0 or executed == 0:
        raise RuntimeError("pytest did not produce a non-empty executed test set")
    if totals["failed"] or totals["errors"]:
        raise RuntimeError("pytest JUnit report contains failures or errors")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)

    environment = subparsers.add_parser("environment")
    environment.add_argument("--output", type=Path, required=True)

    processes = subparsers.add_parser("processes")
    processes.add_argument("--output", type=Path, required=True)

    pytest_result = subparsers.add_parser("pytest-result")
    pytest_result.add_argument("--xml", type=Path, action="append", required=True)
    pytest_result.add_argument("--log", type=Path, required=True)
    pytest_result.add_argument("--output", type=Path, required=True)

    cleanup = subparsers.add_parser("cleanup")
    cleanup.add_argument("--before", type=Path, required=True)
    cleanup.add_argument("--after", type=Path, required=True)
    cleanup.add_argument("--output", type=Path, required=True)
    cleanup.add_argument("--settle-seconds", type=float, default=5.0)

    args = parser.parse_args()
    if args.action == "environment":
        capture_environment(args.output)
    elif args.action == "processes":
        capture_processes(args.output)
    elif args.action == "pytest-result":
        summarize_pytest(args.xml, args.log, args.output)
    else:
        check_cleanup(args.before, args.after, args.output, args.settle_seconds)


if __name__ == "__main__":
    main()
