# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared denoise execution primitives for LTX pipelines."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps

from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from .ltx2_latents import LTXAVState

if TYPE_CHECKING:
    from .ltx2_conditioning import LTXPromptContext
    from .ltx2_request import LTXRequestInputs


@dataclass
class LTXForwardContext:
    """Immutable metadata and schedulers for one LTX denoise phase."""

    req: DiffusionRequestBatch
    request_inputs: LTXRequestInputs
    prompt_context: LTXPromptContext
    device: torch.device
    cfg_parallel_ready: bool
    attention_kwargs: dict[str, Any] | None
    latent_num_frames: int
    latent_height: int
    latent_width: int
    latent_mel_bins: int
    original_audio_num_frames: int
    padded_audio_num_frames: int
    timesteps: torch.Tensor
    audio_scheduler: Any
    video_audio_scheduler: Any

    @property
    def batch_size(self) -> int:
        return self.prompt_context.batch_size

    @property
    def num_videos_per_prompt(self) -> int:
        return self.request_inputs.num_videos_per_prompt


@dataclass
class LTXDenoiseContext:
    """Mutable AV state and positional metadata for a denoise phase."""

    latents: torch.Tensor
    audio_latents: torch.Tensor
    video_coords: torch.Tensor
    audio_coords: torch.Tensor
    conditioning_mask: torch.Tensor | None = None
    conditioning_mask_for_model: torch.Tensor | None = None


@dataclass
class LTXPhaseResult:
    """Denoised, unpacked AV latents and the context used to produce them."""

    forward_context: LTXForwardContext
    video: torch.Tensor
    audio: torch.Tensor


class LTXDenoisePipeline(Protocol):
    """Pipeline state required by :class:`LTXDenoiseExecutor`."""

    @property
    def interrupt(self) -> bool: ...

    def progress_bar(self, iterable=None, total=None): ...


LTXDenoiseStep = Callable[[int, torch.Tensor, LTXAVState], LTXAVState]


class LTXDenoiseExecutor:
    """Run the one shared LTX denoise loop.

    Prediction and scheduler math remain injectable so structural refactors do
    not change the existing LTX2/LTX2.3 numerical paths. Guidance will replace
    that step policy independently.
    """

    @staticmethod
    def run(
        pipeline: LTXDenoisePipeline,
        state: LTXAVState,
        timesteps: Iterable[torch.Tensor],
        step: LTXDenoiseStep,
    ) -> LTXAVState:
        timesteps = tuple(timesteps)
        with pipeline.progress_bar(total=len(timesteps)) as progress_bar:
            for index, timestep in enumerate(timesteps):
                if pipeline.interrupt:
                    continue
                state = step(index, timestep, state)
                progress_bar.update()
        return state


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


class I2VVideoAudioScheduler:
    """Update the unconditioned video frames and the full audio state."""

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


def prepare_scheduler_stage(
    pipeline: Any,
    request_inputs: LTXRequestInputs,
    *,
    device: torch.device,
    sigmas: list[float] | None,
    timesteps: list[int] | None,
    latent_num_frames: int,
    latent_height: int,
    latent_width: int,
) -> tuple[Any, Any, torch.Tensor]:
    sigmas = (
        np.linspace(1.0, 1 / request_inputs.num_inference_steps, request_inputs.num_inference_steps)
        if sigmas is None
        else sigmas
    )
    mu = calculate_shift(
        pipeline.scheduler.config.get("max_image_seq_len", 4096),
        pipeline.scheduler.config.get("base_image_seq_len", 1024),
        pipeline.scheduler.config.get("max_image_seq_len", 4096),
        pipeline.scheduler.config.get("base_shift", 0.95),
        pipeline.scheduler.config.get("max_shift", 2.05),
    )
    audio_scheduler = copy.deepcopy(pipeline.scheduler)
    video_audio_scheduler = pipeline._make_video_audio_scheduler(
        audio_scheduler,
        latent_num_frames,
        latent_height,
        latent_width,
    )
    retrieve_timesteps(
        audio_scheduler,
        request_inputs.num_inference_steps,
        device,
        timesteps,
        sigmas=sigmas,
        mu=mu,
    )
    timesteps_tensor, _ = retrieve_timesteps(
        pipeline.scheduler,
        request_inputs.num_inference_steps,
        device,
        timesteps,
        sigmas=sigmas,
        mu=mu,
    )
    return audio_scheduler, video_audio_scheduler, timesteps_tensor


def prepare_rope_coords_stage(
    pipeline: Any,
    forward_ctx: LTXForwardContext,
    latents: torch.Tensor,
    audio_latents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    video_coords = pipeline.transformer.rope.prepare_video_coords(
        latents.shape[0],
        forward_ctx.latent_num_frames,
        forward_ctx.latent_height,
        forward_ctx.latent_width,
        latents.device,
        fps=forward_ctx.request_inputs.frame_rate,
    )
    audio_coords = pipeline.transformer.audio_rope.prepare_audio_coords(
        audio_latents.shape[0],
        forward_ctx.padded_audio_num_frames,
        audio_latents.device,
    )
    return video_coords, audio_coords


def build_transformer_kwargs(
    pipeline: Any,
    forward_ctx: LTXForwardContext,
    denoise_ctx: LTXDenoiseContext,
    *,
    hidden_states: torch.Tensor,
    audio_hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    audio_encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor | None,
    audio_encoder_attention_mask: torch.Tensor | None,
    ts: torch.Tensor,
) -> dict[str, Any]:
    return {
        "hidden_states": hidden_states,
        "audio_hidden_states": audio_hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "audio_encoder_hidden_states": audio_encoder_hidden_states,
        **pipeline._denoise_timestep_kwargs(ts, forward_ctx, denoise_ctx),
        "encoder_attention_mask": encoder_attention_mask,
        "audio_encoder_attention_mask": audio_encoder_attention_mask,
        "num_frames": forward_ctx.latent_num_frames,
        "height": forward_ctx.latent_height,
        "width": forward_ctx.latent_width,
        "fps": forward_ctx.request_inputs.frame_rate,
        "audio_num_frames": forward_ctx.padded_audio_num_frames,
        "video_coords": denoise_ctx.video_coords,
        "audio_coords": denoise_ctx.audio_coords,
        "attention_kwargs": forward_ctx.attention_kwargs,
        "return_dict": False,
    }


def step_denoised_latents(
    pipeline: Any,
    forward_ctx: LTXForwardContext,
    denoise_ctx: LTXDenoiseContext,
    noise_pred_video: torch.Tensor,
    noise_pred_audio: torch.Tensor,
    timestep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    latents = pipeline.scheduler_step_maybe_with_cfg(
        (noise_pred_video, noise_pred_audio),
        (timestep, timestep),
        (denoise_ctx.latents, denoise_ctx.audio_latents),
        do_true_cfg=pipeline.do_classifier_free_guidance,
        per_request_scheduler=forward_ctx.video_audio_scheduler,
    )
    return pipeline._synchronize_cfg_parallel_step_output(
        latents,
        do_true_cfg=pipeline.do_classifier_free_guidance,
    )


class LTXPhaseExecutor:
    """Prepare and execute one LTX phase without owning model modules."""

    @staticmethod
    def run(
        pipeline: Any,
        req: DiffusionRequestBatch,
        request_inputs: LTXRequestInputs,
        *,
        noise_scale: float,
        sigmas: list[float] | None,
        timesteps: list[int] | None,
        attention_kwargs: dict[str, Any] | None,
        image: Any | None = None,
        prompt_context: LTXPromptContext | None = None,
    ) -> LTXPhaseResult:
        pipeline._check_forward_inputs(request_inputs, image=image)
        cfg_parallel_ready = pipeline._setup_forward_runtime(request_inputs, attention_kwargs)
        device = pipeline.device
        if prompt_context is None:
            prompt_context = pipeline._prepare_prompt_context(
                prompt=request_inputs.prompt,
                negative_prompt=request_inputs.negative_prompt,
                prompt_embeds=request_inputs.prompt_embeds,
                negative_prompt_embeds=request_inputs.negative_prompt_embeds,
                prompt_attention_mask=request_inputs.prompt_attention_mask,
                negative_prompt_attention_mask=request_inputs.negative_prompt_attention_mask,
                num_videos_per_prompt=request_inputs.num_videos_per_prompt,
                max_sequence_length=request_inputs.max_sequence_length,
            )

        latent_num_frames, latent_height, latent_width = pipeline._resolve_video_latent_dimensions(request_inputs)
        latents, conditioning_mask = pipeline._prepare_video_latents_stage(
            request_inputs,
            prompt_context,
            device=device,
            noise_scale=noise_scale,
            image=image,
        )
        audio_latents, original_audio_num_frames, padded_audio_num_frames, latent_mel_bins = (
            pipeline._prepare_audio_latents_stage(
                request_inputs,
                prompt_context,
                device=device,
                noise_scale=noise_scale,
            )
        )
        audio_scheduler, video_audio_scheduler, timesteps_tensor = prepare_scheduler_stage(
            pipeline,
            request_inputs,
            device=device,
            sigmas=sigmas,
            timesteps=timesteps,
            latent_num_frames=latent_num_frames,
            latent_height=latent_height,
            latent_width=latent_width,
        )
        forward_ctx = LTXForwardContext(
            req=req,
            request_inputs=request_inputs,
            prompt_context=prompt_context,
            device=device,
            cfg_parallel_ready=cfg_parallel_ready,
            attention_kwargs=attention_kwargs,
            latent_num_frames=latent_num_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            latent_mel_bins=latent_mel_bins,
            original_audio_num_frames=original_audio_num_frames,
            padded_audio_num_frames=padded_audio_num_frames,
            timesteps=timesteps_tensor,
            audio_scheduler=audio_scheduler,
            video_audio_scheduler=video_audio_scheduler,
        )
        video_coords, audio_coords = prepare_rope_coords_stage(pipeline, forward_ctx, latents, audio_latents)
        denoise_ctx = LTXDenoiseContext(
            latents=latents,
            audio_latents=audio_latents,
            video_coords=video_coords,
            audio_coords=audio_coords,
            conditioning_mask=conditioning_mask,
        )
        denoise_ctx = pipeline._prepare_denoise_context_for_cfg(forward_ctx, denoise_ctx)
        state = LTXDenoiseExecutor.run(
            pipeline,
            LTXAVState(video=denoise_ctx.latents, audio=denoise_ctx.audio_latents),
            forward_ctx.timesteps,
            lambda index, timestep, state: pipeline._denoise_step(
                index,
                timestep,
                state,
                forward_ctx,
                denoise_ctx,
            ),
        )
        denoise_ctx.latents = state.video
        denoise_ctx.audio_latents = state.audio
        latents, audio_latents = pipeline._unpack_and_denormalize_stage(
            forward_ctx,
            state.video,
            state.audio,
        )
        return LTXPhaseResult(forward_context=forward_ctx, video=latents, audio=audio_latents)
