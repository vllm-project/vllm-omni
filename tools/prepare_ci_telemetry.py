# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Validate the native target's CUDA dependency and telemetry exceptions."""

import importlib.metadata as metadata
import platform
import subprocess
import sys
from pathlib import Path

import regex as re
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def _has_known_cusparselt_wheel_tag() -> bool:
    if sys.platform != "linux" or platform.machine() != "aarch64":
        return False
    distribution = metadata.distribution("nvidia-cusparselt-cu13")
    if distribution.version != "0.8.1":
        return False
    wheel = distribution.read_text("WHEEL") or ""
    tags = [line for line in wheel.splitlines() if line.startswith("Tag:")]
    return tags == ["Tag: py3-none-manylinux2014_sbsa"]


def check_runtime() -> None:
    """Allow only the pinned LMCache bounds and the known ARM64 wheel tag."""
    pins = Path(__file__).resolve().parents[1] / "requirements/cuda-runtime.txt"
    expected = dict(line.split("==") for line in pins.read_text().splitlines() if line and not line.startswith("#"))
    if {name: metadata.version(name) for name in expected} != expected:
        raise RuntimeError("Telemetry runtime versions differ from the native target lock")
    result = subprocess.run(["uv", "pip", "check", "--python", sys.executable], capture_output=True, text=True)
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    if result.returncode == 0:
        return
    limits = {
        "opentelemetry-api": Requirement("opentelemetry-api>=1.20.0,<=1.40.0"),
        "opentelemetry-exporter-prometheus": Requirement("opentelemetry-exporter-prometheus>=0.50b0,<=0.61b0"),
    }
    actual_requires = [Requirement(raw) for raw in metadata.requires("lmcache") or ()]
    if result.returncode != 1 or metadata.version("lmcache") != "0.5.6":
        raise RuntimeError("pip check failed outside the pinned LMCache exceptions")
    if any(sum(str(req) == str(limit) for req in actual_requires) != 1 for limit in limits.values()):
        raise RuntimeError("LMCache telemetry metadata differs from the native target exceptions")
    waived = set()
    cusparselt_waived = False
    incompatibilities = None
    for line in result.stdout.splitlines() + result.stderr.splitlines():
        match = re.fullmatch(r"The package `lmcache` requires `([^`]+)`, but `([^`]+)` is installed", line)
        if match:
            req = Requirement(match[1])
            name = canonicalize_name(req.name)
            if name in limits and str(req) == str(limits[name]) and match[2] == expected[name] and name not in waived:
                waived.add(name)
                continue
        if (
            line == "The package `nvidia-cusparselt-cu13` was built for a different platform"
            and not cusparselt_waived
            and _has_known_cusparselt_wheel_tag()
        ):
            cusparselt_waived = True
            continue
        summary = re.fullmatch(r"Found (\d+) incompatibilities", line)
        if summary and incompatibilities is None:
            incompatibilities = int(summary[1])
            continue
        if re.fullmatch(r"Using Python \S+ environment at: .+|Checked \d+ packages? in \S+", line):
            continue
        raise RuntimeError(f"Unexpected pip check output: {line}")
    if waived != set(limits):
        raise RuntimeError("pip check did not report exactly the two native LMCache exceptions")
    if incompatibilities != len(waived) + int(cusparselt_waived):
        raise RuntimeError("pip check summary did not match the recognized exceptions")
    print("WAIVED: LMCache 0.5.6 telemetry upper bounds from native v0.34.0")
    if cusparselt_waived:
        print("WAIVED: nvidia-cusparselt-cu13 0.8.1 ARM64 wheel tag metadata")


if __name__ == "__main__":
    if sys.argv[1:] != ["--check-runtime"]:
        raise RuntimeError("Expected --check-runtime")
    check_runtime()
