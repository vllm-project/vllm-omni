# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Verify the pinned vLLM and compiled CUDA imports on a GPU CI worker."""

from __future__ import annotations

import importlib
import re
from pathlib import Path


def main() -> None:
    dockerfile = Path(__file__).resolve().parents[3] / "docker" / "Dockerfile.ci"
    match = re.search(
        r"^ARG VLLM_VERSION=([0-9]+\.[0-9]+\.[0-9]+)$",
        dockerfile.read_text(),
        re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"Missing vLLM release version in {dockerfile}")
    expected_version = match.group(1)

    import torch
    import vllm

    assert vllm.__version__ == expected_version, vllm.__version__
    assert torch.cuda.is_available(), "vLLM runtime check requires a GPU worker"
    for module in ("flashinfer", "vllm._C_stable_libtorch"):
        importlib.import_module(module)
    print(
        f"Verified vLLM CUDA runtime: vllm={vllm.__version__} "
        f"expected={expected_version} torch={torch.__version__} "
        f"cuda={torch.version.cuda} device={torch.cuda.get_device_name()}"
    )


if __name__ == "__main__":
    main()
