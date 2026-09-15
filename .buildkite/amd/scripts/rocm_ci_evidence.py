#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Capture small, deterministic evidence files for ROCm GPU CI jobs."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path


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
    mount = subprocess.run(
        ["findmnt", "-n", "-T", hf_home, "-o", "TARGET,SOURCE,FSTYPE"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lines = [
        f"recorded_at_epoch={time.time():.6f}",
        f"platform={platform.platform()}",
        f"python={platform.python_version()}",
        f"pytorch={torch.__version__}",
        f"rocm={torch.version.hip}",
        f"visible_gpus={actual}",
        f"expected_gpus={expected}",
        f"image=rocm/vllm-omni:{os.environ.get('BUILDKITE_COMMIT', 'unknown')}",
        f"build_created_at={os.environ.get('BUILDKITE_BUILD_CREATED_AT', 'unknown')}",
        f"job_started_at={os.environ.get('BUILDKITE_JOB_STARTED_AT', 'unknown')}",
        f"queue_delay_seconds={queue_delay}",
        f"hf_home={hf_home}",
        f"hf_mount={mount or 'unknown'}",
    ]
    _write(output, "\n".join(lines) + "\n")


def capture_processes(output: Path) -> None:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,lstart=,stat=,comm="],
        check=True,
        capture_output=True,
        text=True,
    )
    _write(output, result.stdout)


def _process_identities(text: str) -> dict[str, str]:
    identities: dict[str, str] = {}
    for line in text.splitlines():
        fields = line.split(None, 8)
        if len(fields) != 9 or fields[8] == "ps":
            continue
        identity = "|".join((fields[0], *fields[2:7]))
        identities[identity] = line
    return identities


def check_cleanup(before: Path, after: Path, output: Path, settle_seconds: float) -> None:
    time.sleep(settle_seconds)
    capture_processes(after)
    baseline = _process_identities(before.read_text(encoding="utf-8"))
    current = _process_identities(after.read_text(encoding="utf-8"))
    leaked = [line for identity, line in current.items() if identity not in baseline]
    status = "status=PASS leaked_processes=0" if not leaked else "status=FAIL leaked_processes_detected=1"
    _write(output, "\n".join([*leaked, status]) + "\n")
    if leaked:
        raise RuntimeError("processes created by the test remain after teardown")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)

    environment = subparsers.add_parser("environment")
    environment.add_argument("--output", type=Path, required=True)

    processes = subparsers.add_parser("processes")
    processes.add_argument("--output", type=Path, required=True)

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
    else:
        check_cleanup(args.before, args.after, args.output, args.settle_seconds)


if __name__ == "__main__":
    main()
