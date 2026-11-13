# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Verify the pinned vLLM and compiled CUDA imports on a GPU CI worker."""

from __future__ import annotations

import importlib
import re
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    dockerfile = root / "docker" / "Dockerfile.ci"
    text = dockerfile.read_text()
    match = re.search(
        r"^ARG VLLM_PRECOMPILED_WHEEL_COMMIT=([0-9a-f]{40})$",
        text,
        re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"Missing vLLM commit pin in {dockerfile}")
    expected_commit = match.group(1)
    version_match = re.search(r"^ARG VLLM_VERSION=(\S+)$", text, re.MULTILINE)
    if version_match is None:
        raise RuntimeError(f"Missing vLLM version pin in {dockerfile}")
    expected_version = version_match.group(1)
    cudnn_requirements = root / "requirements/cudnn-runtime.txt"
    cudnn_match = re.search(
        r"^nvidia-cudnn-cu13==(\d+)\.(\d+)\.(\d+)\.\d+(?:\s|$)",
        cudnn_requirements.read_text(),
        re.MULTILINE,
    )
    if cudnn_match is None:
        raise RuntimeError(f"Missing cuDNN runtime pin in {cudnn_requirements}")
    major, minor, patch = map(int, cudnn_match.groups())
    expected_cudnn = major * 10000 + minor * 100 + patch

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
    # A final wheel can share its source SHA with an RC while containing
    # different native binaries. Both identities must match this CI cycle.
    assert vllm.__version__ == expected_version, f"Expected vLLM version {expected_version}, got {vllm.__version__}"
    assert torch.cuda.is_available(), "vLLM runtime check requires a GPU worker"
    actual_cudnn = torch.backends.cudnn.version()
    assert actual_cudnn == expected_cudnn, f"Expected cuDNN {expected_cudnn}, got {actual_cudnn}"
    # Python dependency metadata cannot detect a retained connector wheel's
    # Torch ABI mismatch. Import it on the GPU worker before running models.
    for module in ("flashinfer", "vllm._C_stable_libtorch", "lmcache.cuda_ops"):
        importlib.import_module(module)
    print(
        f"Verified vLLM CUDA runtime: vllm={vllm.__version__} "
        f"commit={expected_commit} torch={torch.__version__} "
        f"cuda={torch.version.cuda} cudnn={actual_cudnn} device={torch.cuda.get_device_name()}"
    )


if __name__ == "__main__":
    main()
