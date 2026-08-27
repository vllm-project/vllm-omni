# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Sequence

from vllm_omni.diffusion.sched.sigma_schedule import DMD2SigmaSchedule


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


def _time_shift_sigmas(
    *,
    num_steps: int = 50,
    shift_scale: float = 6.0,
    base_schedule: Sequence[float] | None = None,
) -> list[float]:
    """Build a shifted sigma schedule.

    ``base_schedule`` supplies the rectified-flow positions explicitly and takes
    precedence over ``num_steps``. Distilled checkpoints need it because their
    few-step schedule is not the uniform one ``num_steps`` produces; validation
    and the shift itself live in the shared :class:`DMD2SigmaSchedule`.
    """
    if shift_scale <= 0:
        raise ValueError("MiniMax H3 shift_scale must be > 0")

    if base_schedule is not None:
        return DMD2SigmaSchedule.from_positions(base_schedule).shifted_sigmas(shift_scale)

    import torch

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
