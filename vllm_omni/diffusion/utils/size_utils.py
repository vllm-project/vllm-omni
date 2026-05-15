# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared size normalization helpers for diffusion pipelines."""


def normalize_min_aligned_size(height: int, width: int, alignment: int) -> tuple[int, int]:
    """Clamp dimensions to the minimum valid aligned size.

    This preserves floor-to-alignment behavior for normal requests while
    preventing very small dimensions from collapsing to zero after alignment.
    """

    alignment = int(alignment)
    if alignment <= 0:
        raise ValueError(f"Expected positive alignment, got {alignment}")

    normalized_height = max(alignment, (int(height) // alignment) * alignment)
    normalized_width = max(alignment, (int(width) // alignment) * alignment)
    return normalized_height, normalized_width
