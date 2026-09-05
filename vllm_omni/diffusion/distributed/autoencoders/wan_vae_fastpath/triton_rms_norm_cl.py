# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: N803
"""Single-pass channels-last RMSNorm(+SiLU) for the Wan VAE decoder (Tier 2, not bit-exact).

With channels-last activations every spatial position owns a contiguous vector
of ``C`` channels, so one program can load a ``(BLOCK_P, C)`` tile, reduce over
the channels in registers and write the normalized (and optionally SiLU'd)
result: the activation is read once and written once instead of the two-pass
scheme :mod:`.triton_rms_norm` needs for channels-first layouts.

This kernel is only dispatched at the tolerance-based ``channels_last`` level,
so it uses fast math: one reciprocal per position, ``scale`` folded into
``gamma``, the approximate ``exp`` and divide for SiLU, and a single rounding
to the activation dtype at the end. The bit-exact epilogue sequence (IEEE
``div_rn`` twice, libdevice ``exp``, six intermediate roundings) measured
ALU-bound at 1.6 TB/s on the 512-channel stage.

The kernel can also absorb the bias of the convolution that produced ``x``:
``WanResidualBlock.conv1`` feeds only ``norm2``, so at this level the
convolution runs without its bias and the norm adds it per channel in fp32
before reducing, which removes ATen's separate strided ``add_`` pass (about
0.6 ms per 720p chunk per block). The sum is rounded to the activation dtype
exactly as ATen's ``add_`` does, so folding the bias changes no numerics.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MAX_CHANNELS = 1024
# Positions x channels per program. Autotuning on the first call picked one
# position per program (512 bytes in flight per CTA, 0.8 TB/s on the 256-channel
# stage); a fixed ~4096-element tile keeps enough bytes in flight to approach
# HBM bandwidth, as measured for the data-movement kernels.
_TILE_ELEMENTS = 4096
_NUM_WARPS = 4

if HAS_TRITON:

    @triton.jit
    def _rms_norm_cl_kernel(
        X,
        GAMMA,
        BIAS,
        OUT,
        rows,
        scale,
        eps,
        C: tl.constexpr,
        BLOCK_C: tl.constexpr,
        SILU: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64) * BLOCK_P + tl.arange(0, BLOCK_P)
        col = tl.arange(0, BLOCK_C)
        row_mask = row < rows
        col_mask = col < C
        mask = row_mask[:, None] & col_mask[None, :]
        offsets = row[:, None] * C + col[None, :]

        dtype = X.dtype.element_ty
        x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
        if HAS_BIAS:
            # ATen's in-place conv bias ``add_``: fp32 opmath, rounded once.
            bias = tl.load(BIAS + col, mask=col_mask, other=0.0).to(tl.float32)
            x = (x + bias[None, :]).to(dtype).to(tl.float32)
        inv_norm = 1.0 / tl.maximum(tl.sqrt(tl.sum(x * x, axis=1)), eps)
        gamma = tl.load(GAMMA + col, mask=col_mask, other=0.0).to(tl.float32) * scale
        v = x * inv_norm[:, None] * gamma[None, :]
        if SILU:
            v = v / (1.0 + tl.exp(-v))
        tl.store(OUT + offsets, v.to(dtype), mask=mask)


def _rows_view(x: torch.Tensor) -> torch.Tensor | None:
    """``(positions, C)`` view of a channels-last tensor, or ``None`` if it is not one."""
    channels = x.shape[1]
    if x.dim() == 5:
        if not x.is_contiguous(memory_format=torch.channels_last_3d):
            return None
        rows = x.permute(0, 2, 3, 4, 1).reshape(-1, channels)
    elif x.dim() == 4:
        if not x.is_contiguous(memory_format=torch.channels_last):
            return None
        rows = x.permute(0, 2, 3, 1).reshape(-1, channels)
    else:
        return None
    if rows.data_ptr() != x.data_ptr() or not rows.is_contiguous():
        return None
    return rows


def rms_norm_channels_last(
    x: torch.Tensor,
    gamma: torch.Tensor,
    scale: float,
    *,
    silu: bool = False,
    bias: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor | None:
    """``F.normalize(x + bias, dim=1) * scale * gamma`` [then SiLU] for channels-last ``x`` (fast math), or ``None``.

    ``bias`` is an optional per-channel vector (the un-added bias of the
    convolution that produced ``x``); ``None`` skips the add.
    """
    if not HAS_TRITON or not x.is_cuda or x.dtype not in _SUPPORTED_DTYPES:
        return None
    channels = x.shape[1]
    if channels < 2 or channels > _MAX_CHANNELS or x.numel() == 0:
        return None
    if gamma.numel() != channels or gamma.dtype is not x.dtype or gamma.device != x.device:
        return None
    if bias is not None and (
        bias.dim() != 1 or bias.numel() != channels or bias.device != x.device or bias.dtype not in _SUPPORTED_DTYPES
    ):
        return None
    x_rows = _rows_view(x)
    if x_rows is None:
        return None
    out = torch.empty_like(x)
    out_rows = _rows_view(out)
    if out_rows is None:
        return None
    rows = x_rows.shape[0]
    block_c = triton.next_power_of_2(channels)
    block_p = max(1, _TILE_ELEMENTS // block_c)
    grid = (triton.cdiv(rows, block_p),)

    with torch.get_device_module().device(x.device):
        _rms_norm_cl_kernel[grid](
            x_rows,
            gamma.reshape(-1).contiguous(),
            x_rows if bias is None else bias.contiguous(),
            out_rows,
            rows,
            float(scale),
            float(eps),
            C=channels,
            BLOCK_C=block_c,
            SILU=silu,
            HAS_BIAS=bias is not None,
            BLOCK_P=block_p,
            num_warps=_NUM_WARPS,
            enable_fp_fusion=False,
        )
    return out


__all__ = ["rms_norm_channels_last"]
