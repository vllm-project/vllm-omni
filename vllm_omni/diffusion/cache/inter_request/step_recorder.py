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

    When attached to a pipeline as ``pipeline._step_latents_recorder``, the
    diffuse loop will call ``record()`` after every scheduler step.  After the
    forward pass completes, the runner reads ``records`` and passes them to
    DiTCacheStore for inter-request caching.
    """

    def __init__(self) -> None:
        self._records: list[StepLatentRecord] = []
        self._enabled: bool = True

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
