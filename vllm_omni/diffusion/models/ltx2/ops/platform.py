# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Shared platform eligibility for LTX-2 eager kernels."""

from __future__ import annotations

from functools import cache

import torch
from vllm.triton_utils import HAS_TRITON

from vllm_omni.platforms import current_omni_platform

# Extend this exact whitelist only after correctness and E2E validation on the
# corresponding platform. SM100 and SM103 are the next intended targets.
_VERIFIED_CUDA_COMPUTE_CAPABILITIES = frozenset({90})


@cache
def _is_verified_cuda_device(device_index: int) -> bool:
    if not HAS_TRITON or not current_omni_platform.is_cuda() or not current_omni_platform.is_available():
        return False
    capability = current_omni_platform.get_device_capability(device_id=device_index)
    return capability is not None and capability.to_int() in _VERIFIED_CUDA_COMPUTE_CAPABILITIES


def is_ltx2_ops_eligible(tensor: torch.Tensor) -> bool:
    """Return whether ``tensor`` satisfies the shared LTX-2 eager contract."""

    if not tensor.is_cuda or torch.compiler.is_compiling() or torch.is_grad_enabled():
        return False
    device_index = tensor.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    return _is_verified_cuda_device(int(device_index))


__all__ = ["is_ltx2_ops_eligible"]
