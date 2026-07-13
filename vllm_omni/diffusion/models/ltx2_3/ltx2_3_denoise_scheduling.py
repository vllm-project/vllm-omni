# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LTX-2.3 denoise scheduling helpers.

This module owns the denoise-time scheduling used by LTX-2.3 pipelines. The
current implementation includes the official one-stage sigma schedule and the
composite video/audio scheduler used by the multimodal denoise loop.

Future two-stage, HQ, and I2V-specialized scheduling variants should live here
when they are ready to be shared across public pipeline wrappers.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

from .ltx2_3_misc import _LTX23RequestInputs


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


class _VideoAudioScheduler:
    """Composite scheduler dispatching to video and audio schedulers."""

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


class _I2VVideoAudioScheduler:
    """Keep the image-conditioned first video latent frame fixed while stepping audio normally."""

    def __init__(self, pipeline, audio_scheduler, latent_num_frames, latent_height, latent_width):
        self.video_scheduler = pipeline.scheduler
        self.audio_scheduler = audio_scheduler
        self._pipeline = pipeline
        self._latent_num_frames = latent_num_frames
        self._latent_height = latent_height
        self._latent_width = latent_width

    def step(self, noise_pred, t, latents, return_dict=False, generator=None):
        video_out = self._pipeline._step_video_latents_i2v(
            noise_pred[0],
            latents[0],
            t[0],
            self._latent_num_frames,
            self._latent_height,
            self._latent_width,
        )
        audio_out = self.audio_scheduler.step(
            noise_pred[1],
            t[1],
            latents[1],
            return_dict=False,
            generator=generator,
        )[0]
        return ((video_out, audio_out),)


class LTX23SchedulerMixin:
    def _make_video_audio_scheduler(
        self,
        audio_scheduler: Any,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> Any:
        return _VideoAudioScheduler(self.scheduler, audio_scheduler)

    def _official_ltx23_sigmas(self, num_inference_steps: int, device: torch.device) -> torch.Tensor:
        config = self.scheduler.config
        base_shift = config.get("base_shift", 0.95)
        max_shift = config.get("max_shift", 2.05)
        base_anchor = config.get("base_image_seq_len", 1024)
        max_anchor = config.get("max_image_seq_len", 4096)
        tokens = max_anchor

        sigmas = torch.linspace(1.0, 0.0, num_inference_steps + 1)
        slope = (max_shift - base_shift) / (max_anchor - base_anchor)
        intercept = base_shift - slope * base_anchor
        sigma_shift = tokens * slope + intercept
        exp_shift = math.exp(sigma_shift)
        sigmas = torch.where(sigmas != 0, exp_shift / (exp_shift + (1 / sigmas - 1)), 0)

        terminal = config.get("shift_terminal", 0.1)
        if terminal:
            non_zero_mask = sigmas != 0
            non_zero_sigmas = sigmas[non_zero_mask]
            one_minus_z = 1.0 - non_zero_sigmas
            scale_factor = one_minus_z[-1] / (1.0 - terminal)
            sigmas[non_zero_mask] = 1.0 - (one_minus_z / scale_factor)

        return sigmas.to(dtype=torch.float32, device=device)

    @staticmethod
    def _set_scheduler_sigmas(scheduler: Any, sigmas: torch.Tensor) -> torch.Tensor:
        sigmas = sigmas.to(dtype=torch.float32)
        timesteps = sigmas[:-1] * scheduler.config.num_train_timesteps
        scheduler.sigmas = sigmas
        scheduler.timesteps = timesteps
        scheduler.num_inference_steps = len(timesteps)
        scheduler._step_index = None
        scheduler._begin_index = None
        return timesteps

    def _prepare_scheduler_stage(
        self,
        request_inputs: _LTX23RequestInputs,
        *,
        device: torch.device,
        sigmas: list[float] | None,
        timesteps: list[int] | None,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> tuple[Any, Any, torch.Tensor]:
        audio_scheduler = copy.deepcopy(self.scheduler)
        video_audio_scheduler = self._make_video_audio_scheduler(
            audio_scheduler,
            latent_num_frames,
            latent_height,
            latent_width,
        )
        if sigmas is None and timesteps is None:
            scheduler_sigmas = self._official_ltx23_sigmas(request_inputs.num_inference_steps, device)
            timesteps_tensor = self._set_scheduler_sigmas(self.scheduler, scheduler_sigmas)
            self._set_scheduler_sigmas(audio_scheduler, scheduler_sigmas.clone())
            self._num_timesteps = len(timesteps_tensor)
            return audio_scheduler, video_audio_scheduler, timesteps_tensor

        sigmas = (
            np.linspace(1.0, 1 / request_inputs.num_inference_steps, request_inputs.num_inference_steps)
            if sigmas is None
            else sigmas
        )
        # Use max_image_seq_len (not actual video_sequence_length) for mu calculation,
        # matching diffusers' LTX2Pipeline which hardcodes this value.
        mu = calculate_shift(
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_image_seq_len", 1024),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.95),
            self.scheduler.config.get("max_shift", 2.05),
        )
        _ = retrieve_timesteps(
            audio_scheduler,
            request_inputs.num_inference_steps,
            device,
            timesteps,
            sigmas=sigmas,
            mu=mu,
        )
        timesteps, _ = retrieve_timesteps(
            self.scheduler,
            request_inputs.num_inference_steps,
            device,
            timesteps,
            sigmas=sigmas,
            mu=mu,
        )
        self._num_timesteps = len(timesteps)
        return audio_scheduler, video_audio_scheduler, timesteps
