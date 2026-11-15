# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Validate the pinned CUDA bootstrap and native target's telemetry exceptions."""

import importlib.metadata as metadata
import subprocess
import sys
from pathlib import Path

import regex as re
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

BASE_IMAGE = "vllm/vllm-openai:v0.33.0@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e"
COMMON = "opentelemetry-exporter-otlp-common"


def check_runtime() -> None:
    """Keep pip check strict except for native 9950ce36's two LMCache bounds."""
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
    print("WAIVED: LMCache 0.5.6 telemetry upper bounds from native 9950ce36; all other pip checks remain strict")


def main() -> None:
    # This immutable image's orphan has no shared or OS-owned payload files.
    # Refuse other bootstraps rather than inferring that the same repair is safe.
    if sys.argv[1:] != [BASE_IMAGE]:
        raise RuntimeError("Telemetry repair requires the pinned vLLM CUDA base image")
    pins = Path(__file__).resolve().parents[1] / "requirements/cuda-bootstrap.txt"
    expected = dict(line.split("==") for line in pins.read_text().splitlines())
    actual = {name: metadata.version(name) for name in expected}
    if actual != expected or metadata.version(COMMON) != "0.66b2":
        raise RuntimeError("Telemetry bootstrap package versions differ from the pinned base")
    for distribution in metadata.distributions():
        if not distribution.metadata.get("Name"):
            raise RuntimeError("Unnamed installed distribution; cannot establish orphan status")
        if any(canonicalize_name(Requirement(r).name) == COMMON for r in distribution.requires or ()):
            raise RuntimeError(f"The common exporter is required by {distribution.metadata['Name']}")
    print("Verified the pinned CUDA bootstrap and its unused common exporter")


if __name__ == "__main__":
    if sys.argv[1:] == ["--check-runtime"]:
        check_runtime()
    else:
        main()
