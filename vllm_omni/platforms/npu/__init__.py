# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm_omni.platforms.npu.platform import NPUOmniPlatform

__all__ = ["NPUOmniPlatform"]


def is_a5(device: torch.device | None = None) -> bool:
    """Return True on Ascend 950 (A5) devices, False otherwise.
    """
    if device is not None and device.type != "npu":
        return False
    try:
        from vllm_ascend.utils import is_950

        return bool(is_950())
    except (ImportError, AttributeError):
        return False


def __getattr__(name: str):
    if name != "NPUOmniPlatform":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from vllm_omni.platforms.npu.platform import NPUOmniPlatform

    globals()[name] = NPUOmniPlatform
    return NPUOmniPlatform
