# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
Progress bar mixin for diffusion pipelines.

Provides a diffusers-compatible progress_bar() method that wraps tqdm,
automatically disabling output on non-zero ranks in distributed settings.
"""

import torch
from tqdm.auto import tqdm

from vllm_omni.diffusion.denoise_progress import DenoiseProgress, active_denoise_progress


class _CountingTqdm(tqdm):
    """tqdm that reports its steps to the enclosing ``denoise_progress_scope``.

    Counts also when the bar is disabled on non-zero ranks: each ``update(n)``
    adds ``n``, and an iterated item counts once the loop asks for the next one.
    """

    def __init__(self, *args, progress: DenoiseProgress, **kwargs):
        super().__init__(*args, **kwargs)
        self._denoise_progress = progress
        progress.num_loops += 1
        if progress.num_scheduled_steps is not None:
            progress.num_scheduled_steps = (
                None if self.total is None else progress.num_scheduled_steps + int(self.total)
            )

    def update(self, n=1):
        self._denoise_progress.num_completed_steps += n
        return super().update(n)

    def __iter__(self):
        for obj in super().__iter__():
            yield obj
            # Resumed only after the loop body for ``obj`` finished.
            self._denoise_progress.num_completed_steps += 1


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

        progress = active_denoise_progress()
        if progress is not None and (iterable is not None or total is not None):
            return _CountingTqdm(iterable, total=total, progress=progress, **config)
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
