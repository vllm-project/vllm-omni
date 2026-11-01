# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared helpers for configuring Wan VAE fast path kernels."""

from __future__ import annotations

import torch

_ROW_BLOCK_WIDTHS = (64, 128, 256)


def encoder_nrmse_limit(dtype: torch.dtype) -> float:
    """Channels-last posterior tolerance; BF16 permits the observed GB200 drift."""
    return 0.02 if dtype == torch.bfloat16 else 0.01


def suggests_channels_last(x: torch.Tensor) -> bool:
    """Whether ATen's ``Tensor.suggest_memory_format()`` is channels_last(_3d) for a 4D/5D ``x``.

    Port of ``c10::is_channels_last_strides_2d_s4``/``_3d_s5``, the heuristic that
    ``F.pad``, ``torch.cat`` and ``F.interpolate`` use to lay out their outputs.
    ``is_contiguous(memory_format=...)`` disagrees on ambiguous strides, e.g. a
    contiguous tensor whose only non-unit dimension is C, or a size-1 batch with a
    non-canonical stride, so bit-exact kernels must decide with this instead.
    """
    if x.layout != torch.strided or x.dim() not in (4, 5):
        return False
    sizes, strides = x.shape, x.stride()
    if strides[1] == 0:
        return False
    order = (1, 3, 2, 0) if x.dim() == 4 else (1, 4, 3, 2, 0)
    min_stride = 0
    for d in order:
        if sizes[d] == 0 or strides[d] < min_stride:
            return False
        # ATen resolves ambiguous N111-like strides to channels-first.
        if d == 0 and min_stride == strides[1]:
            return False
        min_stride = strides[d] * sizes[d] if sizes[d] > 1 else strides[d]
    return True


def _pick_block_width(width: int) -> int:
    """The column block (64/128/256) that pads ``width`` the least; ties go to the wider block."""
    best, best_padded = _ROW_BLOCK_WIDTHS[0], None
    for block in _ROW_BLOCK_WIDTHS:
        padded = -(-width // block) * block
        if best_padded is None or padded <= best_padded:
            best, best_padded = block, padded
    return best
