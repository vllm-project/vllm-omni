# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Progress bar mixin for diffusion pipelines.

Provides a diffusers-compatible progress_bar() method that wraps tqdm,
automatically disabling output on non-zero ranks in distributed settings.
"""

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch
from tqdm.auto import tqdm


@dataclass(frozen=True)
class DiffusionProgress:
    request_id: str
    completed: int
    total: int


_progress_sink: ContextVar[Callable[[DiffusionProgress], None] | None] = ContextVar(
    "diffusion_progress_sink", default=None
)
_progress_requests: ContextVar[tuple[str, ...]] = ContextVar("diffusion_progress_requests", default=())


@contextmanager
def progress_sink(sink):
    """Bind the replying worker's transport for the duration of one RPC."""
    token = _progress_sink.set(sink)
    try:
        yield
    finally:
        _progress_sink.reset(token)


@contextmanager
def progress_requests(requests):
    """Bind identities after DP selection, and only for opted-in requests."""
    token = _progress_requests.set(
        tuple(req.request_id for req in requests if req.sampling_params.emit_request_lifecycle)
    )
    try:
        yield
    finally:
        _progress_requests.reset(token)


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

        if iterable is None and total is None:
            raise ValueError("Either `total` or `iterable` has to be defined.")
        bar = tqdm(iterable, total=total, **config)
        sink = _progress_sink.get()
        request_ids = _progress_requests.get()
        # Report explicit step updates through the active request transport.
        if iterable is None and total and sink is not None and request_ids:
            update = bar.update
            completed = 0

            def report_update(n=1):
                nonlocal completed
                result = update(n)
                completed += n
                for request_id in request_ids:
                    sink(DiffusionProgress(request_id, completed, total))
                return result

            bar.update = report_update
        return bar

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs


def _is_rank_zero() -> bool:
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0
