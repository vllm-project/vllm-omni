# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Progress bar mixin for diffusion pipelines.

Provides a diffusers-compatible progress_bar() method that wraps tqdm,
automatically disabling output on non-zero ranks in distributed settings.
"""

import torch
from tqdm.auto import tqdm


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

    # ------------------------------------------------------------------
    # Denoising step callbacks (general-purpose, not inter-request specific)
    # ------------------------------------------------------------------
    # Pipelines call on_diffuse_step_begin / on_diffuse_step_end around each
    # denoising iteration. Registered callbacks can observe steps (e.g. to
    # record intermediate latents) or veto a step (return False from
    # on_diffuse_step_begin to skip it, e.g. to resume from a cached step).
    # When no callbacks are registered these are near-free no-ops.

    def _ensure_step_hooks(self):
        if not hasattr(self, "_diffuse_step_hooks"):
            self._diffuse_step_hooks = []

    def register_diffuse_step_hook(self, hook):
        """Register a denoising-step hook.

        A hook is any object with optional methods:
            on_diffuse_step_begin(self, step_idx, timestep) -> bool
                Return False to skip this step (advance latents from cache).
            on_diffuse_step_end(self, step_idx, timestep, latents) -> None
                Called after the step's latents are updated.
        """
        self._ensure_step_hooks()
        self._diffuse_step_hooks.append(hook)

    def on_diffuse_step_begin(self, step_idx: int, timestep) -> bool:
        """Return True to run the step, False to skip it."""
        hooks = getattr(self, "_diffuse_step_hooks", None)
        if not hooks:
            return True
        run = True
        for h in hooks:
            if hasattr(h, "on_diffuse_step_begin"):
                if h.on_diffuse_step_begin(step_idx, timestep) is False:
                    run = False
        return run

    def on_diffuse_step_end(self, step_idx: int, timestep, latents):
        hooks = getattr(self, "_diffuse_step_hooks", None)
        if not hooks:
            return
        for h in hooks:
            if hasattr(h, "on_diffuse_step_end"):
                h.on_diffuse_step_end(step_idx, timestep, latents)

    # ------------------------------------------------------------------

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
