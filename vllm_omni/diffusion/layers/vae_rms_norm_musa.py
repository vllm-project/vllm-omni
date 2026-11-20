# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MUSA counterpart of ``vae_rms_norm.rms_norm_fastpath`` for dense channel-first inputs.

MUSA's ``linalg.vector_norm`` reduces in a different order than the ``x.float().norm()``
that ``F.normalize`` runs, so this path keeps the latter and then runs the shared
epilogue, which leaves the output bitwise equal to the reference forward.
"""

import torch
from torch import nn
from vllm.triton_utils import HAS_TRITON

from vllm_omni.diffusion.layers import vae_norm_epilogue as rn
from vllm_omni.platforms import current_omni_platform

# The epilogue runs on MUSA only; the helper declines wherever it does not.
_HAS_MUSA_KERNEL = HAS_TRITON and current_omni_platform.is_musa()


def rms_norm_fastpath_musa(norm: nn.Module, x: torch.Tensor) -> torch.Tensor | None:
    """``WanRMS_norm(x)`` on MUSA for dense fp16/bf16 inputs, or ``None``.

    ``None`` means the caller must run the reference forward.
    """
    gamma = norm.gamma
    if (
        not _HAS_MUSA_KERNEL
        or x.device.type != "musa"
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
        or not getattr(norm, "channel_first", False)
        or not (isinstance(norm.bias, float) and norm.bias == 0.0)
        or x.dtype not in (torch.float16, torch.bfloat16)
        or gamma.dtype is not x.dtype
        or x.dim() not in (4, 5)
        or gamma.dim() != x.dim() - 1
        or not x.is_contiguous()
        or x.numel() == 0
    ):
        return None
    # F.normalize's own reduction: MUSA's vector_norm(dtype=float32) reduces in another order.
    denom = x.float().norm(2, 1, keepdim=True).clamp_min_(1e-12)
    n, c = x.shape[:2]
    output = rn.rms_norm_scale(x.view(n, c, -1), denom.view(n, 1, -1), gamma.reshape(-1), norm.scale)
    return None if output is None else output.view(x.shape)
