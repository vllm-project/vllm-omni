# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

BASE_SCHEDULE_KEY = "base_schedule"


@dataclass(frozen=True)
class DMD2SigmaSchedule:
    """Continuous rectified-flow positions pinned by a distilled checkpoint.

    A DMD2 student only ever sees the few noise levels it was trained on, so a
    distilled release ships the exact positions instead of letting the server
    derive a uniform schedule from ``num_inference_steps``.

    This is deliberately distinct from
    ``vllm_omni.diffusion.models.dmd2.DMD2Config.denoising_timesteps``, which
    carries integer scheduler timesteps for scheduler-backed pipelines. Here the
    entries are continuous positions in ``[0, 1]`` that still need a per-modality
    time shift applied, which is what lets one schedule drive several coupled
    modalities at different shift scales.
    """

    base_schedule: tuple[float, ...]

    def __post_init__(self) -> None:
        values = self.base_schedule
        if len(values) < 2:
            raise ValueError("DMD2 base_schedule needs at least 2 entries")
        # Ordering comparisons are all false against NaN, so non-finite entries
        # have to be rejected before the monotonicity check rather than after it.
        if any(not math.isfinite(value) for value in values):
            raise ValueError("DMD2 base_schedule entries must be finite")
        if values[0] != 1.0 or values[-1] != 0.0:
            raise ValueError("DMD2 base_schedule must start at 1.0 and end at 0.0")
        if any(curr <= nxt for curr, nxt in zip(values, values[1:], strict=False)):
            raise ValueError("DMD2 base_schedule must be strictly decreasing")

    @classmethod
    def from_positions(cls, base_schedule: Sequence[float]) -> DMD2SigmaSchedule:
        return cls(base_schedule=tuple(float(value) for value in base_schedule))

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, Any],
        *,
        key: str = BASE_SCHEDULE_KEY,
    ) -> DMD2SigmaSchedule | None:
        """Read a schedule from checkpoint metadata.

        An absent key means the release is not distilled and keeps the legacy
        uniform schedule. An explicitly empty value is a malformed contract and
        is rejected rather than silently falling back.
        """
        raw = metadata.get(key)
        if raw is None:
            return None
        return cls.from_positions(raw)

    @property
    def num_inference_steps(self) -> int:
        """Denoising steps, i.e. one per interval between sigma boundaries."""
        return len(self.base_schedule) - 1

    def shifted_sigmas(self, shift_scale: float) -> list[float]:
        """Apply the rectified-flow time shift for one modality."""
        if shift_scale <= 0:
            raise ValueError("DMD2 shift_scale must be > 0")
        shift = float(shift_scale)
        return [shift * u / (1 + (shift - 1) * u) for u in self.base_schedule]


__all__ = [
    "BASE_SCHEDULE_KEY",
    "DMD2SigmaSchedule",
]
