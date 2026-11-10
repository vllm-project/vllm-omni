# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Guard the orphan-exporter repair for the pinned final vLLM CUDA image."""

import importlib.metadata as metadata
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

BASE_IMAGE = "vllm/vllm-openai:v0.33.0@sha256:d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e"
COMMON = "opentelemetry-exporter-otlp-common"


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
    main()
