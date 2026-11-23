# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm_omni.platforms.npu.platform import NPUOmniPlatform

__all__ = ["NPUOmniPlatform", "log_ascend_runtime_build"]


def log_ascend_runtime_build() -> None:
    """Log the torch_npu / CANN build identity (#9286, request 2).

    torch_npu post-release builds of the same ``2.10.0`` line carry
    materially different op-plugin behavior (e.g. the ``with_stack``
    SIGSEGV fixed upstream by ``dc5c4b4`` in post6 while the nightly
    images still shipped post4). Writing the running build into each
    stage's startup log makes that class of image/stack mismatch
    diagnosable from the log alone.
    """
    try:
        import torch_npu
    except ImportError:
        return
    from vllm.logger import init_logger

    init_logger(__name__).info(
        "Ascend runtime build: torch_npu %s (git %s), CANN %s, torch %s",
        torch_npu.__version__,
        getattr(getattr(torch_npu, "version", None), "git_version", "unknown"),
        getattr(torch.version, "cann", "unknown"),
        torch.__version__,
    )


def is_a5(device: torch.device | None = None) -> bool:
    """Return True on Ascend 950 (A5) devices, False otherwise."""
    if device is not None and device.type != "npu":
        return False
    try:
        from vllm_ascend.device.device_config import is_950

        return bool(is_950())
    except (ImportError, AttributeError):
        return False


def __getattr__(name: str):
    if name != "NPUOmniPlatform":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from vllm_omni.platforms.npu.platform import NPUOmniPlatform

    globals()[name] = NPUOmniPlatform
    return NPUOmniPlatform
