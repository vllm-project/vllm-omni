# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared denoise-stage execution primitives for LTX pipelines."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any, Protocol

import torch


class LTXStagePipeline(Protocol):
    """Pipeline surface required by the model-agnostic stage loop."""

    _current_timestep: torch.Tensor | None

    @property
    def interrupt(self) -> bool: ...

    def progress_bar(self, iterable=None, total=None): ...


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    intercept = base_shift - slope * base_seq_len
    return image_seq_len * slope + intercept


@contextmanager
def denoise_stage(
    pipeline: LTXStagePipeline,
    timesteps: Iterable[torch.Tensor],
) -> Iterator[tuple[Iterator[tuple[int, torch.Tensor]], Any]]:
    """Own common denoise-stage lifecycle while leaving step math local."""
    timesteps = tuple(timesteps)

    def active_steps() -> Iterator[tuple[int, torch.Tensor]]:
        for index, timestep in enumerate(timesteps):
            if pipeline.interrupt:
                continue
            pipeline._current_timestep = timestep
            yield index, timestep

    with pipeline.progress_bar(total=len(timesteps)) as progress_bar:
        yield active_steps(), progress_bar


class VideoAudioScheduler:
    """Composite scheduler dispatching video and audio updates."""

    def __init__(self, video_scheduler, audio_scheduler):
        self.video_scheduler = video_scheduler
        self.audio_scheduler = audio_scheduler

    def step(self, noise_pred, t, latents, return_dict=False, generator=None):
        video_out = self.video_scheduler.step(
            noise_pred[0],
            t[0],
            latents[0],
            return_dict=False,
            generator=generator,
        )[0]
        audio_out = self.audio_scheduler.step(
            noise_pred[1],
            t[1],
            latents[1],
            return_dict=False,
            generator=generator,
        )[0]
        return ((video_out, audio_out),)
