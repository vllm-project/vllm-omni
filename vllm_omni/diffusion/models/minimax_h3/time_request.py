# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from collections.abc import Sequence


def _align_frame_count(frame_count: int) -> int:
    """Snap ``frame_count`` up to the MiniMax H3 17n+5 frame boundary."""
    if frame_count <= 0:
        return 1
    current = int(frame_count)
    while current % 17 != 5:
        current += 1
    return current


def _video_latent_t(frame_count: int) -> int:
    if frame_count <= 5:
        return 2
    return ((int(frame_count) - 5) // 17) * 5 + 2


def _frame_count_from_video_latent_t(out_t: int) -> int:
    if out_t == 1:
        return 1
    if out_t < 2 or (out_t - 2) % 5 != 0:
        raise ValueError("MiniMax H3 video latent T must be 1 or match 5n+2")
    return 17 * ((int(out_t) - 2) // 5) + 5


def _audio_latent_t(duration_seconds: float) -> int:
    # Rounding happens at the 40 Hz audio latent boundary.
    return int(round(float(duration_seconds) * 40.0))


def _validate_base_schedule(base_schedule: Sequence[float]) -> list[float]:
    values = [float(value) for value in base_schedule]
    if len(values) < 2:
        raise ValueError("MiniMax H3 base_schedule needs at least 2 entries")
    # Ordering comparisons are all false against NaN, so non-finite entries have
    # to be rejected before the monotonicity check rather than after it.
    if any(not math.isfinite(value) for value in values):
        raise ValueError("MiniMax H3 base_schedule entries must be finite")
    if values[0] != 1.0 or values[-1] != 0.0:
        raise ValueError("MiniMax H3 base_schedule must start at 1.0 and end at 0.0")
    if any(curr <= nxt for curr, nxt in zip(values, values[1:], strict=False)):
        raise ValueError("MiniMax H3 base_schedule must be strictly decreasing")
    return values


def _time_shift_sigmas(
    *,
    num_steps: int = 50,
    shift_scale: float = 6.0,
    base_schedule: Sequence[float] | None = None,
) -> list[float]:
    """Build a shifted sigma schedule.

    ``base_schedule`` supplies the rectified-flow positions explicitly and takes
    precedence over ``num_steps``. Distilled checkpoints need it because their
    few-step schedule is not the uniform one ``num_steps`` produces.
    """
    if shift_scale <= 0:
        raise ValueError("MiniMax H3 shift_scale must be > 0")

    import torch

    if base_schedule is not None:
        base = torch.tensor(
            _validate_base_schedule(base_schedule),
            device="cpu",
            dtype=torch.float32,
        )
        shifted = float(shift_scale) * base / (1 + (float(shift_scale) - 1) * base)
        return [float(value) for value in shifted.tolist()]

    if num_steps <= 0:
        raise ValueError("MiniMax H3 num_steps must be > 0")

    # The rectified-flow sigma range is fixed at [1.0, 0.0].
    base = torch.linspace(
        1.0,
        0.0,
        int(num_steps),
        device="cpu",
        dtype=torch.float32,
    )
    shifted = float(shift_scale) * base / (1 + (float(shift_scale) - 1) * base)
    shifted, _ = torch.unique_consecutive(shifted, return_counts=True)
    # A one-point request is still exactly one point.  Normal serving uses
    # multiple points, but preserving the requested cardinality keeps
    # ``num_inference_steps`` the sole schedule-size control.
    if num_steps > 1 and shifted[-1].item() > 0.0:
        shifted = torch.cat([shifted, torch.tensor([0.0], dtype=shifted.dtype)])
    return [float(value) for value in shifted.tolist()]


class MiniMaxH3ShapePlanner:
    """Compute MiniMax H3 frame, latent-shape, and time-shift values."""

    def align_frame_count(self, frame_count: int) -> int:
        return _align_frame_count(frame_count)

    def video_latent_t(self, frame_count: int) -> int:
        return _video_latent_t(frame_count)

    def frame_count_from_video_latent_t(self, out_t: int) -> int:
        return _frame_count_from_video_latent_t(out_t)

    def audio_latent_t(self, duration_seconds: float) -> int:
        return _audio_latent_t(duration_seconds)

    def time_shift_sigmas(
        self,
        *,
        num_steps: int = 50,
        shift_scale: float = 6.0,
        base_schedule: Sequence[float] | None = None,
    ) -> list[float]:
        return _time_shift_sigmas(
            num_steps=num_steps,
            shift_scale=shift_scale,
            base_schedule=base_schedule,
        )


MINIMAX_H3_SHAPE_PLANNER = MiniMaxH3ShapePlanner()


def minimax_h3_align_frame_count(frame_count: int) -> int:
    return MINIMAX_H3_SHAPE_PLANNER.align_frame_count(frame_count)


def minimax_h3_frame_count_from_video_latent_t(out_t: int) -> int:
    return MINIMAX_H3_SHAPE_PLANNER.frame_count_from_video_latent_t(out_t)


def minimax_h3_time_shift_sigmas(
    *,
    num_steps: int = 50,
    shift_scale: float = 6.0,
    base_schedule: Sequence[float] | None = None,
) -> list[float]:
    return MINIMAX_H3_SHAPE_PLANNER.time_shift_sigmas(
        num_steps=num_steps,
        shift_scale=shift_scale,
        base_schedule=base_schedule,
    )


def minimax_h3_validate_base_schedule(base_schedule: Sequence[float]) -> list[float]:
    return _validate_base_schedule(base_schedule)
