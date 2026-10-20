# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact channel-first VAE normalization shared by supported diffusion models.

The reduction stays in ATen. The common epilogue preserves activation-dtype
roundings after divide, scale, gamma and scalar zero bias. Model-specific
channels-last/approximation policy belongs to the caller, not this module.
"""

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from vllm_omni.diffusion.layers import vae_norm_epilogue as rn

_NORM_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def kernels_allowed(x: torch.Tensor) -> bool:
    """Allow eager CUDA kernels only when autograd and compilation are inactive."""
    return x.is_cuda and not torch.is_grad_enabled() and not torch.compiler.is_compiling()


def _add_channel_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return x + bias.view(1, -1, *([1] * (x.dim() - 2)))


def _as_rows(
    x: torch.Tensor, denom: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, Callable[[torch.Tensor], torch.Tensor]] | None:
    """View ``x``/``denom`` as contiguous ``(N, C, S)`` / ``(N, 1, S)`` without copying.

    Handles the dense layouts the decoder produces: plain contiguous 4D/5D
    tensors and the frame-major 5D view ``WanResample`` emits
    (``permute(0, 2, 1, 3, 4)`` of a contiguous ``(b, t, c, h, w)`` buffer).
    Returns the two views and a function mapping a kernel output back to
    ``x``'s shape and strides, or ``None`` for any other layout.
    """
    if x.dim() == 5:
        b, c, t, h, w = x.shape
        if x.is_contiguous():
            spatial = t * h * w
            return x.view(b, c, spatial), denom.view(b, 1, spatial), lambda out: out.view(b, c, t, h, w)
        frame_major = x.permute(0, 2, 1, 3, 4)
        denom_frame_major = denom.permute(0, 2, 1, 3, 4)
        if frame_major.is_contiguous() and denom_frame_major.is_contiguous():
            spatial = h * w
            return (
                frame_major.view(b * t, c, spatial),
                denom_frame_major.view(b * t, 1, spatial),
                lambda out: out.view(b, t, c, h, w).permute(0, 2, 1, 3, 4),
            )
        return None
    if x.dim() == 4 and x.is_contiguous():
        n, c, h, w = x.shape
        spatial = h * w
        return x.view(n, c, spatial), denom.view(n, 1, spatial), lambda out: out.view(n, c, h, w)
    return None


def rms_norm_fastpath(
    norm: nn.Module, x: torch.Tensor, *, silu: bool = False, bias: torch.Tensor | None = None
) -> torch.Tensor | None:
    """``WanRMS_norm(x + bias)`` (optionally followed by SiLU) with fewer passes, or ``None``.

    ``bias`` is the un-added per-channel bias of the convolution that produced
    ``x``; only the channels-last kernel absorbs it, every other path adds it
    first with ATen's rounding. ``None`` means the caller must run the reference
    forward (autograd, tensor bias, channel-last norm, unsupported dtype,
    gamma/activation dtype mismatch) and is responsible for adding ``bias`` itself.
    """
    if torch.is_grad_enabled():
        return None
    if not getattr(norm, "channel_first", False):
        return None
    norm_bias = norm.bias
    if not (isinstance(norm_bias, float) and norm_bias == 0.0):
        return None
    gamma = norm.gamma
    if x.dtype not in _NORM_DTYPES or gamma.dtype is not x.dtype:
        return None
    if x.dim() not in (4, 5) or gamma.dim() != x.dim() - 1 or x.numel() == 0:
        return None

    if bias is not None:
        x = _add_channel_bias(x, bias)

    # Same reduction kernel as ``F.normalize(x.float(), dim=1)`` runs, reading
    # the low-precision tensor once instead of materializing an fp32 copy.
    denom = torch.linalg.vector_norm(x, dim=1, keepdim=True, dtype=torch.float32).clamp_min_(1e-12)

    if kernels_allowed(x):
        rows = _as_rows(x, denom)
        if rows is not None:
            x_rows, denom_rows, restore = rows
            out = rn.rms_norm_scale(x_rows, denom_rows, gamma.reshape(-1), norm.scale, silu=silu)
            if out is not None:
                return restore(out)

    # Exact PyTorch restructuring: the divide computes in fp32 opmath and rounds
    # once on store, exactly like ``(x.float() / denom).to(x.dtype)``; the
    # epilogue is upstream's own expression.
    normalized = torch.empty_like(x)
    torch.div(x, denom, out=normalized)
    out = normalized * norm.scale * gamma + norm_bias
    return F.silu(out) if silu else out
