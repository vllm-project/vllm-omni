from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class StepLatentRecord:
    step_index: int
    timestep: float
    latent: torch.Tensor


class StepLatentsRecorder:
    """
    Records latent tensors at each denoising step during a request.

    Implemented as a denoising-step hook: pipelines call
    ``register_diffuse_step_hook`` (see ``ProgressBarMixin``) with an instance of
    this class, and the diffuse loop will invoke ``on_diffuse_step_end`` after
    every scheduler step. After the forward pass completes, the runner reads
    ``records`` and passes them to DiTCacheStore for inter-request caching.

    When ``resume_from_step > 0``, ``on_diffuse_step_begin`` returns False for
    the first ``resume_from_step`` iterations so the pipeline skips them (the
    latents are seeded from ``resume_latents`` by the caller).
    """

    def __init__(self) -> None:
        self._records: list[StepLatentRecord] = []
        self._enabled: bool = True
        # Resume support: skip the first N steps (latents seeded from cache).
        self.resume_from_step: int = 0

    # -- hook interface (called by the pipeline via ProgressBarMixin) ---------
    def on_diffuse_step_begin(self, step_index: int, timestep) -> bool:
        if not self._enabled:
            return True
        # Return False to skip steps below the resume point.
        if step_index < self.resume_from_step:
            return False
        return True

    def on_diffuse_step_end(self, step_index: int, timestep, latent: torch.Tensor) -> None:
        if not self._enabled:
            return
        if step_index < self.resume_from_step:
            return
        self.record(step_index, float(timestep), latent)

    # -- recording -----------------------------------------------------------
    def record(self, step_index: int, timestep: float, latent: torch.Tensor) -> None:
        if not self._enabled:
            return
        self._records.append(
            StepLatentRecord(
                step_index=step_index,
                timestep=timestep,
                latent=latent.detach().clone().cpu(),
            )
        )

    @property
    def records(self) -> list[StepLatentRecord]:
        return self._records

    @property
    def num_steps(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        self._records.clear()

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True
