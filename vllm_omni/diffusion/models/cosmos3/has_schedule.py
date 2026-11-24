# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Pure PyTorch Horizon-Aware Schedule helpers for FASTER.

The functions operate in normalized rectified-flow time where 1 is noise and
0 is clean. Converting normalized times to model timesteps (for example,
``tau * 1000``) remains the caller's responsibility.
"""

from __future__ import annotations

from typing import Literal

import torch

HASFormula = Literal["paper_v1", "paper_v3", "paper", "code"]


def _validate_inputs(
    prefix_lengths: torch.Tensor,
    horizon: int,
    alpha: float,
    u0: float,
    formula: str,
) -> None:
    if prefix_lengths.ndim != 1:
        raise ValueError(f"prefix_lengths must have shape [B], got {tuple(prefix_lengths.shape)}")
    if prefix_lengths.dtype == torch.bool or torch.is_floating_point(prefix_lengths):
        raise TypeError("prefix_lengths must use an integer dtype")
    if horizon < 1:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if torch.any(prefix_lengths < 0) or torch.any(prefix_lengths >= horizon):
        raise ValueError(f"prefix lengths must satisfy 0 <= d < H={horizon}")
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must satisfy 0 < alpha <= 1, got {alpha}")
    if not 0.0 <= u0 < 1.0:
        raise ValueError(f"u0 must satisfy 0 <= u0 < 1, got {u0}")
    if formula not in ("paper_v1", "paper_v3", "paper", "code"):
        raise ValueError(f"formula must be 'paper_v1', 'paper_v3', 'paper', or 'code', got {formula!r}")


def compute_has_hit_times(
    prefix_lengths: torch.Tensor,
    horizon: int,
    *,
    alpha: float,
    u0: float,
    formula: HASFormula = "paper_v1",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return per-position hit times with shape ``[B, H]``.

    ``formula='paper_v1'`` implements ``u0 * (1-q)**alpha`` from the v1
    manuscript. ``formula='paper_v3'`` implements ``u0 * (1-q**alpha)`` from
    the current v3 manuscript and official implementation. The legacy aliases
    ``paper`` and ``code`` map to ``paper_v1`` and ``paper_v3`` respectively.
    The two formula families are identical at alpha=1.
    Prefix entries are assigned the first valid suffix hit time; callers must
    still force their local times to zero with the prefix mask.
    """
    _validate_inputs(prefix_lengths, horizon, alpha, u0, formula)
    device = prefix_lengths.device
    positions = torch.arange(horizon, device=device, dtype=dtype)[None, :]  # [1,H]
    delay = prefix_lengths.to(dtype=dtype)[:, None]  # [B,1]
    denominator = (horizon - 1 - delay).clamp_min(1.0)
    q = (positions - delay).clamp_min(0.0) / denominator

    if formula in ("paper_v3", "code"):
        shape = 1.0 - q.pow(alpha)
    else:
        shape = (1.0 - q).clamp_min(0.0).pow(alpha)
    return shape * u0


def compute_has_local_times(
    global_times: torch.Tensor,
    prefix_lengths: torch.Tensor,
    horizon: int,
    *,
    alpha: float,
    u0: float,
    formula: HASFormula = "paper_v1",
) -> torch.Tensor:
    """Map global times to per-action local times.

    Supported input shapes are ``[B,1]`` for training and ``[S,B,1]`` for an
    inference schedule. Outputs have shape ``[B,H]`` and ``[S,B,H]``.
    """
    hit_times = compute_has_hit_times(
        prefix_lengths,
        horizon,
        alpha=alpha,
        u0=u0,
        formula=formula,
        dtype=global_times.dtype,
    )
    if global_times.ndim == 2:
        if global_times.shape != (prefix_lengths.numel(), 1):
            raise ValueError(
                f"2D global_times must have shape [B,1], got {tuple(global_times.shape)} for B={prefix_lengths.numel()}"
            )
        prefix_mask = torch.arange(horizon, device=prefix_lengths.device)[None, :] < prefix_lengths[:, None]
    elif global_times.ndim == 3:
        if global_times.shape[1:] != (prefix_lengths.numel(), 1):
            raise ValueError(
                "3D global_times must have shape [S,B,1], got "
                f"{tuple(global_times.shape)} for B={prefix_lengths.numel()}"
            )
        hit_times = hit_times[None, :, :]
        prefix_mask = torch.arange(horizon, device=prefix_lengths.device)[None, None, :] < prefix_lengths[None, :, None]
    else:
        raise ValueError(f"global_times must be 2D or 3D, got {global_times.ndim}D")

    hit_times = hit_times.to(device=global_times.device)
    prefix_mask = prefix_mask.to(device=global_times.device)
    local_times = ((global_times - hit_times) / (1.0 - hit_times)).clamp(0.0, 1.0)
    return torch.where(prefix_mask, torch.zeros_like(local_times), local_times)
