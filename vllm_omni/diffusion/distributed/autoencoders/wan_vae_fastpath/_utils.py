# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared helpers for configuring Wan VAE fast path kernels."""

from __future__ import annotations

import torch

_ROW_BLOCK_WIDTHS = (64, 128, 256)


def encoder_nrmse_limit(dtype: torch.dtype) -> float:
    """Channels-last posterior tolerance; BF16 permits the observed GB200 drift."""
    return 0.02 if dtype == torch.bfloat16 else 0.01


def _pick_block_width(width: int) -> int:
    """The column block (64/128/256) that pads ``width`` the least; ties go to the wider block."""
    best, best_padded = _ROW_BLOCK_WIDTHS[0], None
    for block in _ROW_BLOCK_WIDTHS:
        padded = -(-width // block) * block
        if best_padded is None or padded <= best_padded:
            best, best_padded = block, padded
    return best
