# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Realization noise=0/data=1 clocks, including the original DMD re-noise protocol."""

import torch


def sampling_grid(steps, stage="causal", shift=1.0):
    if stage == "causal" and shift == 1.0:
        return tuple(i / steps for i in range(steps)), (1 / steps,) * steps, (1.0,) * steps
    if stage == "dmd":
        u = torch.linspace(1.0, 0.0, 1001)[:-1]
        indices = torch.arange(steps) * 1000 // steps
        times = (1 - shift * u / (1 + (shift - 1) * u))[indices].tolist()
        dt = tuple(1 - t for t in times)
        next_times = tuple(times[1:]) + (1.0,)
    elif stage == "causal":
        u = 1 - torch.linspace(0.0, 1.0, steps + 1)
        grid = (1 - shift * u / (1 + (shift - 1) * u)).tolist()
        times = grid[:-1]
        dt = tuple(b - a for a, b in zip(grid, grid[1:]))
        next_times = (1.0,) * steps
    else:
        raise ValueError("Realization sampling stage must be causal or dmd")
    return tuple(times), dt, next_times


def renoise(media, noise, next_time):
    for value, eps in zip(media, noise, strict=True):
        value.mul_(next_time).add_((1 - next_time) * eps)
