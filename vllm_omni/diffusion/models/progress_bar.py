# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
Progress bar mixin for diffusion pipelines.

Provides a diffusers-compatible progress_bar() method that wraps tqdm,
automatically disabling output on non-zero ranks in distributed settings.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from tqdm.auto import tqdm

_step_counter: ContextVar[list | None] = ContextVar("denoise_step_counter", default=None)


@contextmanager
def count_denoise_steps(enabled: bool = True) -> Iterator[list | None]:
    """Count the steps denoising loops complete through ``progress_bar``; ``[None]`` if no loop uses one."""
    if not enabled:
        yield None
        return
    counter = [None]
    token = _step_counter.set(counter)
    try:
        yield counter
    finally:
        _step_counter.reset(token)


class _CountingTqdm(tqdm):
    """Counts completed steps, also when disabled. Iteration counts an item once its body finishes;
    ``update(n)`` counts only outside iteration, since an enabled tqdm calls it while iterating."""

    def __init__(self, *args, counter: list, **kwargs):
        self._iterating = False
        super().__init__(*args, **kwargs)
        self._counter = counter
        counter[0] = counter[0] or 0

    def update(self, n=1):
        if not self._iterating:
            self._counter[0] += n
        return super().update(n)

    def __iter__(self):
        self._iterating = True
        try:
            for obj in super().__iter__():
                yield obj
                self._counter[0] += 1
        finally:
            self._iterating = False


class ProgressBarMixin:
    """Mixin that provides a progress bar for denoising loops.

    Usage in pipeline:
        class MyPipeline(nn.Module, CFGParallelMixin, ProgressBarMixin):
            def diffuse(self, ...):
                with self.progress_bar(total=num_steps) as pbar:
                    for i, t in enumerate(timesteps):
                        ...
                        pbar.update()
    """

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        config = dict(self._progress_bar_config)
        # Only show progress bar on rank 0 in distributed settings
        if "disable" not in config:
            config["disable"] = not _is_rank_zero()

        counter = _step_counter.get()
        if counter is not None and (iterable is not None or total is not None):
            return _CountingTqdm(iterable, total=total, counter=counter, **config)
        if iterable is not None:
            return tqdm(iterable, **config)
        elif total is not None:
            return tqdm(total=total, **config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs


def _is_rank_zero() -> bool:
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0
