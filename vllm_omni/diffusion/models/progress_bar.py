# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Progress bar mixin for diffusion pipelines.

Provides a diffusers-compatible progress_bar() method that wraps tqdm,
automatically disabling output on non-zero ranks in distributed settings.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch
from tqdm.auto import tqdm

from vllm_omni.diffusion.request import OmniDiffusionRequest


@dataclass(frozen=True)
class DiffusionProgress:
    request_id: str
    completed: int
    total: int


_progress_callback: ContextVar[Callable[[DiffusionProgress], None] | None] = ContextVar(
    "diffusion_progress_callback", default=None
)
_progress_requests: ContextVar[tuple[str, ...]] = ContextVar("diffusion_progress_requests", default=())


@contextmanager
def progress_callback(callback: Callable[[DiffusionProgress], None] | None) -> Generator[None, None, None]:
    """Use this callback to report progress within the block; restore it on exit."""
    token = _progress_callback.set(callback)
    try:
        yield
    finally:
        _progress_callback.reset(token)


@contextmanager
def progress_requests(requests: Iterable[OmniDiffusionRequest]) -> Generator[None, None, None]:
    """Associate progress updates within the block with these request IDs."""
    token = _progress_requests.set(tuple(req.request_id for req in requests))
    try:
        yield
    finally:
        _progress_requests.reset(token)


class RequestProgressBar(tqdm):
    def __init__(
        self,
        iterable: Iterable[object] | None = None,
        *,
        total: int | None = None,
        callback: Callable[[DiffusionProgress], None],
        request_ids: tuple[str, ...],
        **kwargs: object,
    ) -> None:
        self.callback = callback
        self.request_ids = request_ids
        self.completed = 0
        super().__init__(iterable, total=total, **kwargs)

    def update(self, n: int = 1) -> bool | None:
        refreshed = super().update(n)
        self.completed += n
        if self.total:
            for request_id in self.request_ids:
                self.callback(DiffusionProgress(request_id, self.completed, self.total))
        return refreshed

    def __iter__(self) -> Iterator[object]:
        # tqdm's disabled iterator bypasses update; reporting must still run.
        try:
            for item in self.iterable:
                yield item
                self.update()
        finally:
            self.close()


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
        callback = _progress_callback.get()
        request_ids = _progress_requests.get()
        if callback is None or not request_ids:
            return tqdm(iterable, total=total, **config)
        return RequestProgressBar(iterable, total=total, callback=callback, request_ids=request_ids, **config)

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs


def _is_rank_zero() -> bool:
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0
