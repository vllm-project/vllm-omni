# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""ROCm CI setup for deterministic tiny diffusion kernels."""

from __future__ import annotations

import os


def _configure_rocm_ci() -> None:
    force_math_sdpa = os.environ.get("VLLM_OMNI_ROCM_CI_FORCE_MATH_SDPA") == "1"
    disable_miopen = os.environ.get("VLLM_OMNI_ROCM_CI_DISABLE_MIOPEN") == "1"
    if not (force_math_sdpa or disable_miopen):
        return

    try:
        import torch
    except ModuleNotFoundError:
        return

    if not torch.version.hip:
        return

    if disable_miopen:
        torch.backends.cudnn.enabled = False

    if force_math_sdpa:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


_configure_rocm_ci()
