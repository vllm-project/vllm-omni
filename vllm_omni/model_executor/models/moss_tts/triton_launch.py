# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA-only Triton launch options for MOSS kernels."""

from vllm.platforms import current_platform


def cuda_pdl_launch_kwargs(use_pdl: bool) -> dict[str, bool]:
    """Return Triton's ``launch_pdl`` option only where that runtime accepts it.

    Non-CUDA Triton runtimes can reject the keyword even when the value is
    false. Callers still pass the ``USE_PDL`` constexpr, which stays false
    off Hopper, so kernel results do not change.
    """
    if not current_platform.is_cuda():
        return {}
    return {"launch_pdl": use_pdl}
