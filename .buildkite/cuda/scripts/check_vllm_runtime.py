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
        r"^ARG VLLM_PRECOMPILED_WHEEL_COMMIT=([0-9a-f]{40})$",
        dockerfile.read_text(),
        re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"Missing vLLM commit pin in {dockerfile}")
    expected_commit = match.group(1)

    import torch
    import vllm
    from vllm._version import __commit_id__

    # Release wheels omit the SHA from __version__, but retain the build
    # commit in _version. Check it so a different wheel of the same release
    # cannot silently pass the pin check.
    installed_commit = (__commit_id__ or "").removeprefix("g")
    assert len(installed_commit) >= 8 and expected_commit.startswith(installed_commit), (
        f"Expected vLLM {expected_commit}, got {__commit_id__} ({vllm.__version__})"
    )
    assert torch.cuda.is_available(), "vLLM runtime check requires a GPU worker"
    for module in ("flashinfer", "vllm._C_stable_libtorch"):
        importlib.import_module(module)
    print(
        f"Verified vLLM CUDA runtime: vllm={vllm.__version__} "
        f"commit={expected_commit} torch={torch.__version__} "
        f"cuda={torch.version.cuda} device={torch.cuda.get_device_name()}"
    )


if __name__ == "__main__":
    main()
