# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pillow-identical LANCZOS resize of 8-bit images on the device.

Pillow resamples 8-bit images with fixed-point integer arithmetic: double
precision LANCZOS weights (``precompute_coeffs``) are normalized and rounded
to ``PRECISION_BITS = 22`` bit integers (``normalize_coeffs_8bpc``), and each
separable pass (horizontal, then vertical) accumulates ``pixel * weight`` in
int32 from a half-unit bias, shifts, and clamps to [0, 255]. The weights are
computed here exactly as Pillow does (the same double math through ``math``),
and the integer passes run on the device, so the bytes match
``Image.resize(..., LANCZOS)``.
"""

from __future__ import annotations

import functools
import math

import torch

_PRECISION_BITS = 32 - 8 - 2
_LANCZOS_SUPPORT = 3.0


def _sinc(x: float) -> float:
    if x == 0.0:
        return 1.0
    x = x * math.pi
    return math.sin(x) / x


def _lanczos(x: float) -> float:
    if -_LANCZOS_SUPPORT <= x < _LANCZOS_SUPPORT:
        return _sinc(x) * _sinc(x / _LANCZOS_SUPPORT)
    return 0.0


@functools.lru_cache(maxsize=32)
def _lanczos_coefficients(in_size: int, out_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(first_index [out], weights [out, ksize])`` of one axis, as Pillow."""
    scale = filterscale = in_size / out_size
    if filterscale < 1.0:
        filterscale = 1.0
    support = _LANCZOS_SUPPORT * filterscale
    ksize = int(math.ceil(support)) * 2 + 1
    first, weights = [], []
    for out_index in range(out_size):
        center = (out_index + 0.5) * scale
        inverse = 1.0 / filterscale
        start = max(0, int(center - support + 0.5))
        count = min(in_size, int(center + support + 0.5)) - start
        taps = [_lanczos((tap + start - center + 0.5) * inverse) for tap in range(count)]
        total = 0.0
        for tap in taps:  # left to right, as the C loop
            total += tap
        if total != 0.0:
            taps = [tap / total for tap in taps]
        fixed = [int((-0.5 if tap < 0 else 0.5) + tap * (1 << _PRECISION_BITS)) for tap in taps]
        first.append(start)
        weights.append(fixed + [0] * (ksize - count))
    return torch.tensor(first, dtype=torch.long), torch.tensor(weights, dtype=torch.int32)


def _resample_axis(image: torch.Tensor, in_size: int, out_size: int) -> torch.Tensor:
    """One pass over the last axis of an int32 tensor."""
    first, weights = _lanczos_coefficients(in_size, out_size)
    first, weights = first.to(image.device), weights.to(image.device)
    acc = torch.full((*image.shape[:-1], out_size), 1 << (_PRECISION_BITS - 1), dtype=torch.int32, device=image.device)
    for tap in range(weights.shape[1]):
        index = (first + tap).clamp_(max=in_size - 1)  # clamped taps carry weight 0
        acc.add_(image.index_select(-1, index) * weights[:, tap])
    return torch.clamp(acc >> _PRECISION_BITS, 0, 255)


def resize_lanczos_uint8(images: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Resize uint8 ``[..., H, W]`` images exactly as Pillow's LANCZOS."""
    if images.dtype != torch.uint8:
        raise ValueError(f"expected uint8 images, got {images.dtype}")
    in_height, in_width = images.shape[-2:]
    data = images.to(torch.int32)
    if width != in_width:
        data = _resample_axis(data, in_width, width)
    if height != in_height:
        data = _resample_axis(data.transpose(-1, -2), in_height, height).transpose(-1, -2)
    return data.to(torch.uint8).contiguous()


__all__ = ["resize_lanczos_uint8"]
