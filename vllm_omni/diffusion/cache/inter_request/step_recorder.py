from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
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
    Records latent tensors at each denoising step during the first request.

    When attached to a pipeline as ``pipeline._step_latents_recorder``, the
    diffuse loop will call ``record()`` after every scheduler step.  After the
    forward pass completes, call ``save()`` to persist all step latents to disk
    or ``clear()`` to discard them.
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

    def save(self, save_dir: str | Path, prefix: str = "step") -> list[str]:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[str] = []
        for rec in self._records:
            filename = f"{prefix}_step{rec.step_index:04d}_t{rec.timestep:.1f}.pt"
            filepath = save_dir / filename
            torch.save(
                {"step_index": rec.step_index, "timestep": rec.timestep, "latent": rec.latent},
                filepath,
            )
            saved_paths.append(str(filepath))
        logger.info(
            "Saved %d step latents to %s (%.2f MB total)",
            len(saved_paths),
            save_dir,
            sum(os.path.getsize(p) for p in saved_paths) / 1024**2,
        )
        return saved_paths

    def clear(self) -> None:
        self._records.clear()

    def disable(self) -> None:
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True
