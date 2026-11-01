# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Cached 3D RoPE tables for LTX DiffVAE kernels."""

from __future__ import annotations

import torch

_TABLE_CACHE_MAX = 16
_TABLE_CACHE: dict[
    tuple[torch.device, int, int, int, tuple[int, int, int], float],
    tuple[tuple[torch.Tensor, torch.Tensor], ...],
] = {}


def _axis_tables(
    length: int,
    dim: int,
    device: torch.device,
    base: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    exponents = torch.arange(0, dim, 2, dtype=torch.float64, device=device) / dim
    inv_freqs = (1.0 / base**exponents).to(torch.float32)
    positions = torch.arange(length, dtype=torch.float32, device=device)
    angles = positions[:, None] * inv_freqs[None, :]
    return angles.cos().contiguous(), angles.sin().contiguous()


def get_rope_tables(
    query: torch.Tensor,
    dim_split: tuple[int, int, int],
    base: float,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    num_frames, height, width = query.shape[1:4]
    cache_key = (query.device, num_frames, height, width, dim_split, base)
    cached = _TABLE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    tables = tuple(
        _axis_tables(length, dim, query.device, base)
        for length, dim in zip((num_frames, height, width), dim_split, strict=True)
    )
    if len(_TABLE_CACHE) >= _TABLE_CACHE_MAX:
        _TABLE_CACHE.pop(next(iter(_TABLE_CACHE)))
    _TABLE_CACHE[cache_key] = tables
    return tables


__all__ = ["get_rope_tables"]
