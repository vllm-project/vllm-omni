# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared helpers for configuring Wan VAE fast path kernels."""

from __future__ import annotations

_ROW_BLOCK_WIDTHS = (64, 128, 256)


def _pick_block_width(width: int) -> int:
    """The column block (64/128/256) that pads ``width`` the least; ties go to the wider block."""
    best, best_padded = _ROW_BLOCK_WIDTHS[0], None
    for block in _ROW_BLOCK_WIDTHS:
        padded = -(-width // block) * block
        if best_padded is None or padded <= best_padded:
            best, best_padded = block, padded
    return best
