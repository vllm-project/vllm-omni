# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused kernels for the Wan VAE decoder on channels_last_3d activations.

Adapted from SGLang's ``sglang.kernels.ops.diffusion`` (``norm/wan_rmsnorm_silu_triton.py`` and
``layout/nearest_upsample_nhwc_triton.py``, Apache-2.0), reduced to what the LingBot-World streaming decode
uses. Each kernel comes with a ``can_use_*`` predicate and a torch reference of the eager op chain it
replaces; the predicate decides, the kernel raises on an input it does not support.

- :func:`wan_rmsnorm_silu`: ``SiLU(F.normalize(x, dim=1) * scale * gamma + bias)`` on a dense channels_last_3d
  ``[B, C, T, H, W]`` tensor, one program per pixel. fp32 channel statistics; the intermediate dtype
  boundaries of the eager chain are kept (normalise and ``* scale`` in ``x.dtype``, ``* gamma`` in the
  promoted dtype, SiLU in fp32) and the result is stored in ``x.dtype``, which is the dtype the following
  convolution reads under autocast anyway. Not bit-identical to aten (the channel reduction order differs),
  so it is gated with the bf16 decode option rather than mounted unconditionally.
- :func:`nearest_upsample_nhwc`: integer-factor nearest upsample of a dense channels_last ``[N, C, H, W]``
  tensor as a gather, bit-exact vs ``nn.Upsample`` in ``nearest`` and ``nearest-exact`` modes; aten's own NHWC
  nearest kernel is several times slower than its NCHW one, which is what a channels_last decoder would hit.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

try:  # Triton is optional: CPU tests and non-CUDA platforms use the torch references.
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

_MAX_CHANNELS = 1024
_MAX_INT32 = 2**31 - 1
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


# --------------------------------------------------------------------------- RMSNorm + SiLU
def wan_rmsnorm_silu_reference(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | float,
    scale: float,
    eps: float = 1e-12,
    upcast: bool = True,
) -> torch.Tensor:
    """The eager norm chain followed by SiLU, cast back to ``x.dtype``.

    ``upcast=True`` is diffusers' ``WanRMS_norm`` (normalise in fp32, round to ``x.dtype``); ``upcast=False``
    is vLLM-Omni's ``RMSNormVAE`` (``F.normalize`` on ``x`` itself, so every step rounds to ``x.dtype``).
    """
    if upcast:
        normalized = F.normalize(x.float(), dim=1, eps=eps).to(x.dtype)
    else:
        normalized = F.normalize(x, dim=1, eps=eps)
    y = normalized * scale * gamma + bias
    return F.silu(y).to(x.dtype)


def _dense_channels_last_3d(x: torch.Tensor) -> bool:
    if x.dim() != 5 or x.shape[1] <= 1:
        return False
    b, c, t, h, w = x.shape
    return x.stride() == (t * h * w * c, 1, h * w * c, w * c, c)


def can_use_wan_rmsnorm_silu(x: torch.Tensor, gamma: torch.Tensor, bias: torch.Tensor | float | None) -> bool:
    return (
        _HAS_TRITON
        and x.is_cuda
        and not (torch.is_grad_enabled() and x.requires_grad)
        and x.dtype in _SUPPORTED_DTYPES
        and x.numel() > 0
        and 0 < x.shape[1] <= _MAX_CHANNELS
        and _dense_channels_last_3d(x)
        and gamma.is_cuda
        and gamma.device == x.device
        and gamma.numel() == x.shape[1]
        and (gamma.dtype == x.dtype or gamma.dtype == torch.float32)
        and (
            not isinstance(bias, torch.Tensor)
            or (bias.device == x.device and bias.numel() == x.shape[1] and bias.dtype in (x.dtype, torch.float32))
        )
    )


if _HAS_TRITON:

    @triton.jit
    def _wan_rmsnorm_silu_kernel(
        x_ptr,
        gamma_ptr,
        bias_ptr,
        out_ptr,
        channels: tl.constexpr,
        rms_scale,
        eps,
        has_bias: tl.constexpr,
        round_affine: tl.constexpr,
        upcast: tl.constexpr,
        block_c: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        offsets = tl.arange(0, block_c)
        mask = offsets < channels
        # Dense channels_last_3d stores each pixel as one contiguous channel row.
        row_offsets = row * channels + offsets
        x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
        norm = tl.sqrt(tl.sum(x * x, axis=0))
        if upcast:
            # WanRMS_norm: fp32 normalise, one rounding to x.dtype.
            y = (x / tl.maximum(norm, eps)).to(x_ptr.dtype.element_ty)
        else:
            # RMSNormVAE: F.normalize on x itself, so the norm, the clamp and the quotient each round to x.dtype.
            denom = tl.maximum(norm.to(x_ptr.dtype.element_ty).to(tl.float32), eps)
            denom = denom.to(x_ptr.dtype.element_ty).to(tl.float32)
            y = (x / denom).to(x_ptr.dtype.element_ty)
        # Then the eager boundaries: * scale in x.dtype, * gamma / + bias in the promoted dtype, SiLU in fp32,
        # stored in x.dtype.
        y = (y * rms_scale).to(x_ptr.dtype.element_ty)
        gamma = tl.load(gamma_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
        y = y.to(tl.float32) * gamma
        if round_affine:
            # gamma (and bias) share x's dtype: the eager product and sum are rounded to it.
            y = y.to(x_ptr.dtype.element_ty).to(tl.float32)
        if has_bias:
            bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            y = y + bias
            if round_affine:
                y = y.to(x_ptr.dtype.element_ty).to(tl.float32)
        y = y * tl.sigmoid(y)
        tl.store(out_ptr + row_offsets, y, mask=mask)


def wan_rmsnorm_silu(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | float | None,
    scale: float,
    eps: float = 1e-12,
    upcast: bool = True,
) -> torch.Tensor:
    """Fused ``SiLU(normalize(x) * scale * gamma + bias)`` on a dense channels_last_3d tensor; raises otherwise."""
    if not can_use_wan_rmsnorm_silu(x, gamma, bias):
        raise ValueError(
            "wan_rmsnorm_silu needs a CUDA, dense channels_last_3d, bf16/fp16/fp32 [B, C, T, H, W] tensor with "
            f"C <= {_MAX_CHANNELS}; got {x.device.type} {tuple(x.shape)} {x.dtype} strides {tuple(x.stride())}"
        )
    b, c, t, h, w = x.shape
    out = torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=x.dtype)
    has_bias = isinstance(bias, torch.Tensor)
    bias_arg = bias.reshape(-1) if isinstance(bias, torch.Tensor) else gamma
    # With affine parameters in x's dtype the eager chain rounds after ``* gamma`` and ``+ bias``; fp32
    # parameters promote the chain to fp32 and the only rounding left is the store.
    round_affine = gamma.dtype == x.dtype and x.dtype != torch.float32
    block_c = triton.next_power_of_2(c)
    num_warps = 1 if block_c <= 64 else 4 if block_c <= 512 else 8
    with torch.get_device_module(x.device).device(x.device):
        _wan_rmsnorm_silu_kernel[(b * t * h * w,)](
            x,
            gamma.reshape(-1),
            bias_arg,
            out,
            c,
            float(scale),
            float(eps),
            has_bias,
            round_affine,
            upcast,
            block_c,
            num_warps=num_warps,
        )
    return out


# --------------------------------------------------------------------------- nearest upsample, NHWC
def _integer_scale(scale) -> tuple[int, int] | None:
    if isinstance(scale, bool):
        return None
    if isinstance(scale, (int, float)):
        scale = (scale, scale)
    if not isinstance(scale, (tuple, list)) or len(scale) != 2:
        return None
    out = []
    for s in scale:
        if isinstance(s, bool) or not isinstance(s, (int, float)):
            return None
        f = float(s)
        if not math.isfinite(f) or f < 1.0 or f != int(f):
            return None
        out.append(int(f))
    return out[0], out[1]


def canonical_nhwc(x: torch.Tensor) -> bool:
    """Dense channels_last with ``C > 1`` and canonical strides on every dim, size-1 dims included."""
    if x.dim() != 4:
        return False
    _, c, h, w = x.shape
    return c > 1 and x.stride() == (h * w * c, 1, w * c, c)


def can_use_nearest_upsample_nhwc(x: torch.Tensor, scale_factor, mode: str) -> bool:
    return (
        _HAS_TRITON
        and isinstance(x, torch.Tensor)
        and x.is_cuda
        and not (torch.is_grad_enabled() and x.requires_grad)
        and mode in ("nearest", "nearest-exact")
        and x.dim() == 4
        and x.numel() > 0
        and x.dtype in _SUPPORTED_DTYPES
        and canonical_nhwc(x)
        and _integer_scale(scale_factor) is not None
    )


if _HAS_TRITON:

    @triton.jit
    def _nearest_upsample_nhwc_kernel(
        x_ptr, out_ptr, total, channels, out_h, out_w, fh, fw, sxn, sxh, sxw, idx64: tl.constexpr, block: tl.constexpr
    ):
        pid = tl.program_id(0)
        if idx64:
            offs = pid.to(tl.int64) * block + tl.arange(0, block).to(tl.int64)
        else:
            offs = pid * block + tl.arange(0, block)
        mask = offs < total
        # Output is dense NHWC: offs = ((n * out_h + h) * out_w + w) * channels + c.
        c = offs % channels
        t = offs // channels
        w = t % out_w
        t = t // out_w
        h = t % out_h
        n = t // out_h
        src = n * sxn + (h // fh) * sxh + (w // fw) * sxw + c
        vals = tl.load(x_ptr + src, mask=mask)
        tl.store(out_ptr + offs, vals, mask=mask)


def nearest_upsample_nhwc(x: torch.Tensor, scale_factor) -> torch.Tensor:
    """Integer-factor nearest upsample of a canonical channels_last ``[N, C, H, W]`` tensor, dense channels_last out."""
    factors = _integer_scale(scale_factor)
    if factors is None or not can_use_nearest_upsample_nhwc(x, scale_factor, "nearest"):
        raise ValueError(
            "nearest_upsample_nhwc needs a CUDA, canonical channels_last 4D bf16/fp16/fp32 tensor and an integer "
            f"scale; got {x.device.type} {tuple(x.shape)} strides {tuple(x.stride())} scale {scale_factor!r}"
        )
    fh, fw = factors
    n, c, h, w = x.shape
    out_h, out_w = h * fh, w * fw
    out = torch.empty((n, c, out_h, out_w), device=x.device, dtype=x.dtype, memory_format=torch.channels_last)
    total = out.numel()
    if total == 0:
        return out
    sxn, _, sxh, sxw = x.stride()
    block = 1024
    with torch.get_device_module(x.device).device(x.device):
        _nearest_upsample_nhwc_kernel[(triton.cdiv(total, block),)](
            x,
            out,
            total,
            c,
            out_h,
            out_w,
            fh,
            fw,
            sxn,
            sxh,
            sxw,
            idx64=total >= _MAX_INT32 or x.numel() >= _MAX_INT32,
            block=block,
        )
    return out
