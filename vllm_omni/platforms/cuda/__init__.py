# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_omni.platforms.cuda.platform import CudaOmniPlatform

__all__ = ["CudaOmniPlatform"]


def __getattr__(name: str):
    """Resolve ``CudaOmniPlatform`` lazily.

    Importing the platform class pulls in ``vllm.platforms.cuda`` and the
    compiled vLLM extension, which needs a CUDA driver. Submodules such as
    ``vllm_omni.platforms.cuda.host_registration`` are usable without it, so a
    CPU test or a non-CUDA inspection should not fail just because it imports
    this package.
    """
    if name == "CudaOmniPlatform":
        from vllm_omni.platforms.cuda.platform import CudaOmniPlatform

        return CudaOmniPlatform
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
