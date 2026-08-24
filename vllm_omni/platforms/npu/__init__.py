# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_omni.platforms.npu.platform import NPUOmniPlatform

__all__ = ["NPUOmniPlatform"]


def __getattr__(name: str):
    if name != "NPUOmniPlatform":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from vllm_omni.platforms.npu.platform import NPUOmniPlatform

    globals()[name] = NPUOmniPlatform
    return NPUOmniPlatform
