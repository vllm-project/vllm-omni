# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact, finite-schedule AdaLN tables; no interpolation or timestep rounding."""

from __future__ import annotations

import torch


def canonical_lookup_timesteps(values: list[float]) -> list[float]:
    times = torch.tensor(values, dtype=torch.float32, device="cpu")
    if times.ndim != 1 or times.numel() == 0:
        raise ValueError("adaln_lookup_timesteps must be a nonempty list")
    if not bool(torch.isfinite(times).all()) or bool(((times < 0) | (times > 1)).any()):
        raise ValueError("adaln_lookup_timesteps must be finite and in [0, 1]")
    if bool((times[1:] <= times[:-1]).any()):
        raise ValueError("adaln_lookup_timesteps must be strictly increasing after FP32 conversion")
    return times.tolist()


def lookup_timestep_indices(timesteps: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    if timesteps.ndim != 1 or timesteps.dtype != torch.float32:
        raise ValueError("AdaLN lookup requires one-dimensional FP32 timesteps")
    indices = torch.searchsorted(table, timesteps)
    safe_indices = indices.clamp(max=table.numel() - 1)
    if not bool((table[safe_indices] == timesteps).all().item()):
        raise ValueError(
            "Request timestep is absent from the AdaLN lookup checkpoint. "
            "Use its exported step count, flow shifts and condition timesteps, or a BF16 AdaLN checkpoint."
        )
    return safe_indices
