# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared runtime surface for LTX pipeline variants."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.parallel_state import get_classifier_free_guidance_world_size
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch, split_diffusion_output_by_request

from . import ltx2_latents as latent_ops
from .ltx2_denoise import (
    LTXDenoiseContext,
    LTXForwardContext,
    LTXPhaseExecutor,
    LTXPhaseResult,
    VideoAudioScheduler,
    build_transformer_kwargs,
    prepare_rope_coords_stage,
    prepare_scheduler_stage,
    step_denoised_latents,
)


@dataclass
class LTXRequestInputs:
    """Resolved request values shared by all LTX execution variants."""

    prompt: str | list[str] | None
    negative_prompt: str | list[str] | None
    height: int
    width: int
    num_frames: int
    frame_rate: float
    num_inference_steps: int
    guidance_scale: float
    guidance_rescale: float
    num_videos_per_prompt: int
    generator: torch.Generator | list[torch.Generator] | None
    latents: torch.Tensor | None
    audio_latents: torch.Tensor | None
    prompt_embeds: torch.Tensor | None
    negative_prompt_embeds: torch.Tensor | None
    prompt_attention_mask: torch.Tensor | None
    negative_prompt_attention_mask: torch.Tensor | None
    decode_timestep: float | list[float]
    decode_noise_scale: float | list[float] | None
    output_type: str
    max_sequence_length: int


@dataclass
class LTXPromptContext:
    """Connector outputs consumed by an LTX denoise phase."""

    batch_size: int
    connector_prompt_embeds: torch.Tensor
    connector_audio_prompt_embeds: torch.Tensor
    connector_attention_mask: torch.Tensor
    positive_connector_prompt_embeds: torch.Tensor
    positive_connector_audio_prompt_embeds: torch.Tensor
    positive_connector_attention_mask: torch.Tensor
    negative_connector_prompt_embeds: torch.Tensor | None
    negative_connector_audio_prompt_embeds: torch.Tensor | None
    negative_connector_attention_mask: torch.Tensor | None


class LTXPipelineBase(nn.Module, CFGParallelMixin, ProgressBarMixin, SupportsComponentDiscovery):
    """Common state and primitives shared by LTX2 and LTX2.3 pipelines."""

    _pack_latents = staticmethod(latent_ops.pack_latents)
    _unpack_latents = staticmethod(latent_ops.unpack_latents)
    _normalize_latents = staticmethod(latent_ops.normalize_latents)
    _normalize_audio_latents = staticmethod(latent_ops.normalize_audio_latents)
    _denormalize_latents = staticmethod(latent_ops.denormalize_latents)
    _denormalize_audio_latents = staticmethod(latent_ops.denormalize_audio_latents)
    _create_noised_state = staticmethod(latent_ops.create_noised_state)
    _pack_audio_latents = staticmethod(latent_ops.pack_audio_latents)
    _unpack_audio_latents = staticmethod(latent_ops.unpack_audio_latents)
    _unpad_audio_latents = staticmethod(latent_ops.unpad_audio_latents)
    _get_sp_padded_audio_latent_length = staticmethod(latent_ops.get_sp_padded_audio_latent_length)
    _resolve_video_latent_shape = staticmethod(latent_ops.resolve_video_latent_shape)

    def prepare_latents(
        self,
        batch_size: int = 1,
        num_channels_latents: int = 128,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        noise_scale: float = 0.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return latent_ops.prepare_video_latents(
            self,
            batch_size,
            num_channels_latents,
            height,
            width,
            num_frames,
            noise_scale,
            dtype,
            device,
            generator,
            latents,
        )

    def prepare_audio_latents(
        self,
        batch_size: int = 1,
        num_channels_latents: int = 8,
        audio_latent_length: int = 1,
        num_mel_bins: int = 64,
        noise_scale: float = 0.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int, int]:
        return latent_ops.prepare_audio_latents(
            self,
            batch_size,
            num_channels_latents,
            audio_latent_length,
            num_mel_bins,
            noise_scale,
            dtype,
            device,
            generator,
            latents,
        )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    def _transformer_cache_context(self, context_name: str):
        cache_context = getattr(self.transformer, "cache_context", None)
        if callable(cache_context):
            return cache_context(context_name)
        return nullcontext()

    def predict_noise(self, **kwargs):
        with self._transformer_cache_context("cond_uncond"):
            noise_pred_video, noise_pred_audio = self.transformer(**kwargs)
        return noise_pred_video.float(), noise_pred_audio.float()

    def _synchronize_cfg_parallel_step_output(
        self,
        latents: tuple[torch.Tensor, torch.Tensor],
        do_true_cfg: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (do_true_cfg and get_classifier_free_guidance_world_size() > 1):
            return latents

        # CUDA async execution otherwise permits numerical drift to accumulate
        # across CFG-parallel denoise steps.
        latents = tuple(tensor.contiguous() for tensor in latents)
        device = next((tensor.device for tensor in latents if tensor.is_cuda), None)
        if device is not None:
            torch.cuda.current_stream(device).synchronize()
        return latents

    def _setup_forward_runtime(
        self,
        request_inputs: LTXRequestInputs,
        attention_kwargs: dict[str, Any] | None,
    ) -> bool:
        self._guidance_scale = request_inputs.guidance_scale
        self._guidance_rescale = request_inputs.guidance_rescale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        self._current_timestep = None
        return self.do_classifier_free_guidance and get_classifier_free_guidance_world_size() > 1

    def _check_forward_inputs(
        self,
        request_inputs: LTXRequestInputs,
        image: Any | None = None,
    ) -> None:
        self.check_inputs(
            prompt=request_inputs.prompt,
            height=request_inputs.height,
            width=request_inputs.width,
            prompt_embeds=request_inputs.prompt_embeds,
            negative_prompt_embeds=request_inputs.negative_prompt_embeds,
            prompt_attention_mask=request_inputs.prompt_attention_mask,
            negative_prompt_attention_mask=request_inputs.negative_prompt_attention_mask,
        )

    def _prepare_video_latents_stage(
        self,
        request_inputs: LTXRequestInputs,
        prompt_context: LTXPromptContext,
        *,
        device: torch.device,
        noise_scale: float,
        image: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        latents = self.prepare_latents(
            prompt_context.batch_size * request_inputs.num_videos_per_prompt,
            self.transformer.config.in_channels,
            request_inputs.height,
            request_inputs.width,
            request_inputs.num_frames,
            noise_scale,
            torch.float32,
            device,
            request_inputs.generator,
            request_inputs.latents,
        )
        return latents, None

    def _resolve_video_latent_dimensions(self, request_inputs: LTXRequestInputs) -> tuple[int, int, int]:
        latent_num_frames, latent_height, latent_width = self._resolve_video_latent_shape(
            request_inputs.height,
            request_inputs.width,
            request_inputs.num_frames,
            vae_spatial_compression_ratio=self.vae_spatial_compression_ratio,
            vae_temporal_compression_ratio=self.vae_temporal_compression_ratio,
        )
        latents = request_inputs.latents
        if latents is not None:
            if latents.ndim == 5:
                _, _, latent_num_frames, latent_height, latent_width = latents.shape
            elif latents.ndim != 3:
                raise ValueError(
                    f"Provided `latents` tensor has shape {latents.shape}, expected a packed 3D or unpacked 5D tensor."
                )
        return latent_num_frames, latent_height, latent_width

    def _prepare_audio_latents_stage(
        self,
        request_inputs: LTXRequestInputs,
        prompt_context: LTXPromptContext,
        *,
        device: torch.device,
        noise_scale: float,
    ) -> tuple[torch.Tensor, int, int, int]:
        duration_s = request_inputs.num_frames / request_inputs.frame_rate
        audio_latents_per_second = (
            self.audio_sampling_rate / self.audio_hop_length / float(self.audio_vae_temporal_compression_ratio)
        )
        audio_num_frames = round(duration_s * audio_latents_per_second)
        audio_num_frames = self._resolve_audio_latent_length(audio_num_frames, request_inputs.audio_latents)

        num_mel_bins = self.audio_vae.config.mel_bins if self.audio_vae is not None else 64
        latent_mel_bins = num_mel_bins // self.audio_vae_mel_compression_ratio
        num_channels = self.audio_vae.config.latent_channels if self.audio_vae is not None else 8
        audio_latents, original_num_frames, padded_num_frames = self.prepare_audio_latents(
            prompt_context.batch_size * request_inputs.num_videos_per_prompt,
            num_channels_latents=num_channels,
            audio_latent_length=audio_num_frames,
            num_mel_bins=num_mel_bins,
            noise_scale=noise_scale,
            dtype=torch.float32,
            device=device,
            generator=request_inputs.generator,
            latents=request_inputs.audio_latents,
        )
        return audio_latents, original_num_frames, padded_num_frames, latent_mel_bins

    def _resolve_audio_latent_length(
        self,
        requested_length: int,
        audio_latents: torch.Tensor | None,
    ) -> int:
        if audio_latents is not None and audio_latents.ndim == 4:
            return audio_latents.shape[2]
        return requested_length

    def _scheduler_shift_sequence_length(
        self,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> int:
        return latent_num_frames * latent_height * latent_width

    def _make_video_audio_scheduler(
        self,
        audio_scheduler: Any,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> Any:
        return VideoAudioScheduler(self.scheduler, audio_scheduler)

    def _prepare_scheduler_stage(
        self,
        request_inputs: LTXRequestInputs,
        *,
        device: torch.device,
        sigmas: list[float] | None,
        timesteps: list[int] | None,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> tuple[Any, Any, torch.Tensor]:
        return prepare_scheduler_stage(
            self,
            request_inputs,
            device=device,
            sigmas=sigmas,
            timesteps=timesteps,
            latent_num_frames=latent_num_frames,
            latent_height=latent_height,
            latent_width=latent_width,
        )

    def _prepare_rope_coords_stage(
        self,
        forward_ctx: LTXForwardContext,
        latents: torch.Tensor,
        audio_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return prepare_rope_coords_stage(self, forward_ctx, latents, audio_latents)

    def _prepare_denoise_context_for_cfg(
        self,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> LTXDenoiseContext:
        return denoise_ctx

    def _denoise_timestep_kwargs(
        self,
        ts: torch.Tensor,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> dict[str, torch.Tensor]:
        return {"timestep": ts}

    def _build_transformer_kwargs(
        self,
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
        return build_transformer_kwargs(
            self,
            forward_ctx,
            denoise_ctx,
            hidden_states=hidden_states,
            audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            audio_encoder_hidden_states=audio_encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            audio_encoder_attention_mask=audio_encoder_attention_mask,
            ts=ts,
        )

    def _step_denoised_latents(
        self,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
        noise_pred_video: torch.Tensor,
        noise_pred_audio: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return step_denoised_latents(
            self,
            forward_ctx,
            denoise_ctx,
            noise_pred_video,
            noise_pred_audio,
            timestep,
        )

    def _unpack_and_denormalize_stage(
        self,
        forward_ctx: LTXForwardContext,
        latents: torch.Tensor,
        audio_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self._unpack_latents(
            latents,
            forward_ctx.latent_num_frames,
            forward_ctx.latent_height,
            forward_ctx.latent_width,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )
        latents = self._denormalize_latents(
            latents,
            self.vae.latents_mean,
            self.vae.latents_std,
            self.vae.config.scaling_factor,
        )

        audio_latents = self._unpad_audio_latents(audio_latents, forward_ctx.original_audio_num_frames)
        audio_latents = self._denormalize_audio_latents(
            audio_latents,
            self.audio_vae.latents_mean,
            self.audio_vae.latents_std,
        )
        audio_latents = self._unpack_audio_latents(
            audio_latents,
            forward_ctx.original_audio_num_frames,
            num_mel_bins=forward_ctx.latent_mel_bins,
        )
        return latents, audio_latents

    def _run_denoise_phase(
        self,
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
        """Prepare and execute one phase without decoding its output."""
        return LTXPhaseExecutor.run(
            self,
            req,
            request_inputs,
            noise_scale=noise_scale,
            sigmas=sigmas,
            timesteps=timesteps,
            attention_kwargs=attention_kwargs,
            image=image,
            prompt_context=prompt_context,
        )

    def _decode_and_split(
        self,
        forward_ctx: LTXForwardContext,
        latents: torch.Tensor,
        audio_latents: torch.Tensor,
    ):
        request_inputs = forward_ctx.request_inputs
        output = self._decode_output(
            latents=latents,
            audio_latents=audio_latents,
            output_type=request_inputs.output_type,
            connector_prompt_embeds=forward_ctx.prompt_context.connector_prompt_embeds,
            generator=request_inputs.generator,
            device=forward_ctx.device,
            decode_timestep=request_inputs.decode_timestep,
            decode_noise_scale=request_inputs.decode_noise_scale,
            prompt_batch_size=forward_ctx.batch_size,
        )
        if not self.supports_request_batch:
            return output
        return split_diffusion_output_by_request(
            output,
            forward_ctx.req,
            num_outputs_per_prompt=forward_ctx.num_videos_per_prompt,
        )

    def _forward_impl(
        self,
        req: DiffusionRequestBatch,
        request_inputs: LTXRequestInputs,
        *,
        noise_scale: float,
        sigmas: list[float] | None,
        timesteps: list[int] | None,
        attention_kwargs: dict[str, Any] | None,
        image: Any | None = None,
    ):
        """Execute one LTX phase and decode its AV output."""
        phase = self._run_denoise_phase(
            req,
            request_inputs,
            noise_scale=noise_scale,
            sigmas=sigmas,
            timesteps=timesteps,
            attention_kwargs=attention_kwargs,
            image=image,
        )
        return self._decode_and_split(phase.forward_context, phase.video, phase.audio)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)
