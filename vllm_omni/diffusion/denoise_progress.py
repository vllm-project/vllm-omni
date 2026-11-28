# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Count the denoising steps a pipeline forward completes through ``ProgressBarMixin.progress_bar``."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class DenoiseProgress:
    num_completed_steps: int = 0
    num_loops: int = 0

    def result(self) -> int | None:
        """Completed steps, or None when no progress bar reported (the pipeline does not use one)."""
        return self.num_completed_steps if self.num_loops else None


_active: ContextVar[DenoiseProgress | None] = ContextVar("_active_denoise_progress", default=None)


def active_denoise_progress() -> DenoiseProgress | None:
    return _active.get()


@contextmanager
def denoise_progress_scope(enabled: bool = True) -> Iterator[DenoiseProgress | None]:
    if not enabled:
        yield None
        return
    progress = DenoiseProgress()
    token = _active.set(progress)
    try:
        yield progress
    finally:
        _active.reset(token)
