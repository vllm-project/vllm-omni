# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""TaoMate-H3 sigma schedules and the positive-only Euler (eta = 0) update.

The student runs three denoise steps per chunk at the sigma positions
``(0, 16, 33, 49)`` of MiniMax-H3's 50-point time-shifted schedule (shift 12
for video, 3 for audio). The audio teacher runs the base model's 10-point
schedule (nine forwards). Both are the release's ``time_shift_sigmas`` with
``torch.linspace(1, 0, n)``; they are kept here verbatim so the port cannot
drift from the distilled contract through a shared helper's defaults.
"""

from __future__ import annotations

import math

import torch

DISTILLED_STATE_INDICES = (0, 16, 33, 49)
STUDENT_SCHEDULE_STEPS = 50
TEACHER_SCHEDULE_STEPS = 10
TEACHER_STATE_NUMBERS = (3, 6, 9)
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0


def time_shift_sigmas(*, num_steps: int, shift_scale: float) -> list[float]:
    if num_steps <= 1 or not math.isfinite(shift_scale) or shift_scale <= 0:
        raise ValueError("sigma schedule requires num_steps > 1 and shift_scale > 0")
    base = torch.linspace(1.0, 0.0, num_steps, dtype=torch.float32, device="cpu")
    shifted = shift_scale * base / (1.0 + (shift_scale - 1.0) * base)
    shifted = torch.unique_consecutive(shifted)
    if int(shifted.numel()) != num_steps:
        raise ValueError("shifted sigma schedule changed cardinality")
    return [float(item) for item in shifted.tolist()]


def select_time_shift_sigmas(
    *,
    num_steps: int,
    shift_scale: float,
    state_indices: tuple[int, ...] | None = None,
) -> list[float]:
    schedule = time_shift_sigmas(num_steps=num_steps, shift_scale=shift_scale)
    if state_indices is None:
        return schedule
    indices = tuple(state_indices)
    if len(indices) < 2 or indices[0] != 0 or indices[-1] != num_steps - 1 or tuple(sorted(set(indices))) != indices:
        raise ValueError("retained sigma indices must be sorted, unique, and span the schedule")
    return [schedule[index] for index in indices]


def student_sigmas() -> tuple[list[float], list[float]]:
    """(video, audio) sigma ladders of the three-step distilled student."""
    return (
        select_time_shift_sigmas(
            num_steps=STUDENT_SCHEDULE_STEPS, shift_scale=VIDEO_SHIFT, state_indices=DISTILLED_STATE_INDICES
        ),
        select_time_shift_sigmas(
            num_steps=STUDENT_SCHEDULE_STEPS, shift_scale=AUDIO_SHIFT, state_indices=DISTILLED_STATE_INDICES
        ),
    )


def teacher_sigmas() -> tuple[list[float], list[float]]:
    """(video, audio) sigma ladders of the ten-state base audio teacher."""
    return (
        select_time_shift_sigmas(num_steps=TEACHER_SCHEDULE_STEPS, shift_scale=VIDEO_SHIFT),
        select_time_shift_sigmas(num_steps=TEACHER_SCHEDULE_STEPS, shift_scale=AUDIO_SHIFT),
    )


@torch.no_grad()
def euler_eta0_update_(
    state: torch.Tensor,
    velocity: torch.Tensor,
    *,
    sigma_curr: float,
    sigma_next: float,
) -> torch.Tensor:
    """In-place rectified-flow Euler step with eta = 0, in the release's op order.

    ``denoised = state + sigma * v``; ``state <- ratio * state + (1 - ratio) * denoised``
    with ``ratio = sigma_next / sigma_curr``.
    """
    denoised = torch.add(state, velocity, alpha=float(sigma_curr))
    ratio = float(sigma_next) / float(sigma_curr)
    state.mul_(ratio).add_(denoised, alpha=1.0 - ratio)
    return state


__all__ = [
    "AUDIO_SHIFT",
    "DISTILLED_STATE_INDICES",
    "TEACHER_STATE_NUMBERS",
    "VIDEO_SHIFT",
    "euler_eta0_update_",
    "select_time_shift_sigmas",
    "student_sigmas",
    "teacher_sigmas",
    "time_shift_sigmas",
]
