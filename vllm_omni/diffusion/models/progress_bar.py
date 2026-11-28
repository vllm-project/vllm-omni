# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Progress bar mixin for diffusion pipelines.

Provides a diffusers-compatible progress_bar() method that wraps tqdm,
automatically disabling output on non-zero ranks in distributed settings.
"""

import torch
from tqdm.auto import tqdm

from vllm_omni.diffusion.denoise_progress import DenoiseProgress, active_denoise_progress


class _CountingTqdm(tqdm):
    """tqdm that counts completed steps into the enclosing ``denoise_progress_scope``, also when disabled.

    Iteration counts an item once its loop body finishes; ``update(n)`` counts only outside iteration,
    since an enabled tqdm calls it while iterating to refresh the display.
    """

    def __init__(self, *args, progress: DenoiseProgress, **kwargs):
        self._denoise_iterating = False
        super().__init__(*args, **kwargs)
        self._denoise_progress = progress
        progress.num_loops += 1

    def update(self, n=1):
        if not self._denoise_iterating:
            self._denoise_progress.num_completed_steps += n
        return super().update(n)

    def __iter__(self):
        self._denoise_iterating = True
        try:
            for obj in super().__iter__():
                yield obj
                self._denoise_progress.num_completed_steps += 1
        finally:
            self._denoise_iterating = False


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
