# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Validate the native target's CUDA dependency and telemetry exceptions."""

import importlib.metadata as metadata
import subprocess
import sys
from pathlib import Path

import regex as re
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def check_runtime() -> None:
    """Keep pip check strict except for native v0.34.0's two LMCache bounds."""
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
    waived = []
    for line in (result.stdout + result.stderr).splitlines():
        match = re.fullmatch(r"The package `lmcache` requires `([^`]+)`, but `([^`]+)` is installed", line)
        if match:
            req = Requirement(match[1])
            name = canonicalize_name(req.name)
            if name in limits and str(req) == str(limits[name]) and match[2] == expected[name]:
                waived.append(name)
                continue
        if re.fullmatch(
            r"Using Python \S+ environment at: .+|Checked \d+ packages? in \S+|Found 2 incompatibilities", line
        ):
            continue
        raise RuntimeError(f"Unexpected pip check output: {line}")
    if len(waived) != 2 or set(waived) != set(limits):
        raise RuntimeError("pip check did not report exactly the two native LMCache exceptions")
    print("WAIVED: LMCache 0.5.6 telemetry upper bounds from native v0.34.0; all other pip checks remain strict")


if __name__ == "__main__":
    if sys.argv[1:] != ["--check-runtime"]:
        raise RuntimeError("Expected --check-runtime")
    check_runtime()
