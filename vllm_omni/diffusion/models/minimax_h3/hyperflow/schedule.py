# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Schedule grids adapted from Video-Rebirth/hyperflow-h3-flash (Apache-2.0).
# `shift_sigmas` / `build_row_time_pairs` follow MiniMax-H3 + HyperFlow two-time
# conditioning; see upstream THIRD_PARTY_NOTICES.md.
"""HyperFlow H3 Flash 4 / 6 / 8-step sigma grids and ``(t, r)`` row plans.

MiniMax-H3 / HyperFlow use the ``t = 1 - sigma`` axis (``t = 1`` is clean).
Each step also supplies endpoint ``r_i = 1 - sigma_{i+1}``. Pairs are uniqued as
pairs (not by ``t`` alone): at step 0 video and audio can share ``t`` but differ in ``r``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from vllm_omni.diffusion.sched.sigma_schedule import DMD2SigmaSchedule

#: Raw (unshifted) sigma grids by step count.
SIGMA_GRIDS: dict[int, tuple[float, ...]] = {
    4: (1.0, 0.681872, 0.205019, 0.036496, 0.0),
    6: (1.0, 0.825217, 0.471725, 0.147401, 0.049626, 0.015517, 0.0),
    8: (1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0),
}
DEFAULT_STEPS = 6
SUPPORTED_STEPS = frozenset(SIGMA_GRIDS)
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0


def sigma_grid(steps: int | None = None) -> tuple[float, ...]:
    steps = DEFAULT_STEPS if steps is None else int(steps)
    if steps not in SIGMA_GRIDS:
        raise ValueError(f"HyperFlow H3 Flash samples {sorted(SIGMA_GRIDS)} steps, got {steps}")
    return SIGMA_GRIDS[steps]


def hyperflow_schedule(steps: int | None = None) -> DMD2SigmaSchedule:
    return DMD2SigmaSchedule.from_positions(sigma_grid(steps))


def shift_sigmas(sigmas: Sequence[float] | torch.Tensor, shift: float) -> torch.Tensor:
    grid = torch.as_tensor(sigmas, dtype=torch.float32).flatten().cpu()
    return shift * grid / (1.0 + (shift - 1.0) * grid)


def build_row_time_pairs(
    *,
    video_indices: torch.Tensor,
    audio_indices: torch.Tensor,
    num_condition_video_rows: int,
    num_condition_audio_rows: int,
    num_text_tokens: int,
    video_time: tuple[float, float],
    audio_time: tuple[float, float],
    condition_video_timestep: float,
    condition_audio_timestep: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign every packed row a ``(t, r)`` pair and collapse to unique pairs."""
    video_indices = video_indices.cpu()
    audio_indices = audio_indices.cpu()
    sequence_length = int(video_indices.numel() + audio_indices.numel() + num_text_tokens)

    pairs = torch.tensor(video_time, dtype=torch.float32).repeat(sequence_length, 1)
    pairs[video_indices[:num_condition_video_rows]] = float(condition_video_timestep)
    pairs[audio_indices[num_condition_audio_rows:]] = torch.tensor(audio_time, dtype=torch.float32)
    pairs[audio_indices[:num_condition_audio_rows]] = float(condition_audio_timestep)

    unique, indices = torch.unique(pairs, dim=0, sorted=True, return_inverse=True)
    return unique[:, 0].contiguous(), unique[:, 1].contiguous(), indices


__all__ = [
    "AUDIO_SHIFT",
    "DEFAULT_STEPS",
    "SIGMA_GRIDS",
    "SUPPORTED_STEPS",
    "VIDEO_SHIFT",
    "build_row_time_pairs",
    "hyperflow_schedule",
    "shift_sigmas",
    "sigma_grid",
]
