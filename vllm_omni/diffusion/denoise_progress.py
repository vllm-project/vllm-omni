# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Count the denoising steps a pipeline forward completes.

While a ``denoise_progress_scope`` is open, every denoising-loop progress bar
opened through ``ProgressBarMixin.progress_bar`` reports its iterations here.
Pipelines without that progress bar report nothing (``None``).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class DenoiseProgress:
    """Steps counted during one pipeline forward."""

    num_completed_steps: int = 0
    num_loops: int = 0

    def result(self) -> int | None:
        """Completed denoising steps, or None when no loop reported."""
        if not self.num_loops:
            return None
        return self.num_completed_steps


_active_denoise_progress: ContextVar[DenoiseProgress | None] = ContextVar("_active_denoise_progress", default=None)


def active_denoise_progress() -> DenoiseProgress | None:
    return _active_denoise_progress.get()


@contextmanager
def denoise_progress_scope(enabled: bool = True) -> Iterator[DenoiseProgress | None]:
    """Count the denoising steps of the pipeline forward run inside this scope."""
    if not enabled:
        yield None
        return
    progress = DenoiseProgress()
    token = _active_denoise_progress.set(progress)
    try:
        yield progress
    finally:
        _active_denoise_progress.reset(token)
