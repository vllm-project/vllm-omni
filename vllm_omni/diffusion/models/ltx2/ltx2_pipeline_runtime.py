# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared runtime surface for LTX pipeline variants."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any, ClassVar

import torch
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.parallel_state import get_classifier_free_guidance_world_size
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch, split_diffusion_output_by_request

from . import ltx2_latents as latent_ops
from .ltx2_components import LTXComponentProfile, initialize_pipeline_components
from .ltx2_conditioning import LTXPromptContext, LTXTextConditioningMixin
from .ltx2_denoise import (
    LTXDenoiseContext,
    LTXForwardContext,
    LTXPhaseExecutor,
    LTXPhaseResult,
    VideoAudioScheduler,
    build_transformer_kwargs,
    step_denoised_latents,
)
from .ltx2_guidance import LTXGuidanceStrategy
from .ltx2_recipes import LTXOneStageRecipe
from .ltx2_request import LTXRequestInputs, LTXRequestMixin


def _expand_per_prompt_decode_value(
    value: float | list[float],
    *,
    prompt_batch_size: int,
    effective_batch_size: int,
    field_name: str,
) -> list[float]:
    if not isinstance(value, list):
        return [value] * effective_batch_size
    if len(value) == 1:
        return value * effective_batch_size
    if len(value) == effective_batch_size:
        return value
    if prompt_batch_size > 0 and len(value) == prompt_batch_size and effective_batch_size % prompt_batch_size == 0:
        repeats = effective_batch_size // prompt_batch_size
        return [item for item in value for _ in range(repeats)]
    raise ValueError(
        f"`{field_name}` must have length 1, prompt batch size ({prompt_batch_size}), or effective batch size"
        f" ({effective_batch_size}); got {len(value)}."
    )


def _prepare_decode_timestep_conditioning(
    *,
    decode_timestep: float | list[float],
    decode_noise_scale: float | list[float] | None,
    prompt_batch_size: int,
    effective_batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    decode_timestep_values = _expand_per_prompt_decode_value(
        decode_timestep,
        prompt_batch_size=prompt_batch_size,
        effective_batch_size=effective_batch_size,
        field_name="decode_timestep",
    )
    decode_noise_scale_values = (
        decode_timestep_values
        if decode_noise_scale is None
        else _expand_per_prompt_decode_value(
            decode_noise_scale,
            prompt_batch_size=prompt_batch_size,
            effective_batch_size=effective_batch_size,
            field_name="decode_noise_scale",
        )
    )
    return (
        torch.tensor(decode_timestep_values, device=device, dtype=dtype),
        torch.tensor(decode_noise_scale_values, device=device, dtype=dtype)[:, None, None, None, None],
    )


class LTXPipelineRuntime(
    LTXRequestMixin,
    LTXTextConditioningMixin,
    nn.Module,
    CFGParallelMixin,
    ProgressBarMixin,
    SupportsComponentDiscovery,
    DiffusionPipelineProfilerMixin,
):
    """Shared Omni runtime for explicitly composed LTX denoise phases."""

    component_profile: ClassVar[LTXComponentProfile]
    guidance_strategy: ClassVar[LTXGuidanceStrategy]
    one_stage_recipe: ClassVar[LTXOneStageRecipe]
    supports_request_batch = False
    connector_batches_cfg = False
    distributed_video_decode = True
    support_image_input = False
    dummy_run_num_frames = 2
    preserve_sp_padded_audio_duration = False
    reports_stage_durations = False

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        del prefix
        super().__init__()
        initialize_pipeline_components(self, od_config)
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

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
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1.0

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

    def combine_cfg_noise(
        self,
        positive_noise_pred,
        negative_noise_pred,
        true_cfg_scale,
        cfg_normalize=False,
        kwargs: dict[str, Any] | None = None,
        **context: Any,
    ):
        if kwargs is not None:
            context = {**kwargs, **context}
        return self.guidance_strategy.combine_cfg_noise(
            self,
            positive_noise_pred,
            negative_noise_pred,
            true_cfg_scale,
            cfg_normalize,
            context,
        )

    def predict_noise_with_parallel_cfg(self, *args, **kwargs):
        predict_parallel_cfg = getattr(self.guidance_strategy, "predict_parallel_cfg", None)
        if predict_parallel_cfg is None:
            raise NotImplementedError("The selected LTX guidance strategy does not implement parallel CFG.")
        return predict_parallel_cfg(self, *args, **kwargs)

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
        del attention_kwargs
        self._interrupt = False
        cfg_world_size = get_classifier_free_guidance_world_size()
        if self.do_classifier_free_guidance:
            self.guidance_strategy.validate_cfg_world_size(cfg_world_size)
        return self.do_classifier_free_guidance and cfg_world_size > 1

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

    def _resolve_request_image(
        self,
        req: DiffusionRequestBatch,
        image: Any | None,
        request_inputs: LTXRequestInputs,
    ) -> Any | None:
        del req, request_inputs
        return image

    def _make_output(self, output: tuple[torch.Tensor, torch.Tensor]) -> DiffusionOutput:
        if self.reports_stage_durations:
            return DiffusionOutput(
                output=output,
                stage_durations=getattr(self, "stage_durations", None),
            )
        return DiffusionOutput(output=output)

    def _decode_output(
        self,
        *,
        latents: torch.Tensor,
        audio_latents: torch.Tensor,
        output_type: str,
        connector_prompt_embeds: torch.Tensor,
        generator: torch.Generator | list[torch.Generator] | None,
        device: torch.device,
        decode_timestep: float | list[float],
        decode_noise_scale: float | list[float] | None,
        prompt_batch_size: int,
    ) -> DiffusionOutput:
        if output_type == "latent":
            return self._make_output((latents, audio_latents))

        latents = latents.to(connector_prompt_embeds.dtype)
        if not self.vae.config.timestep_conditioning:
            timestep_decode = None
        else:
            noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
            timestep_decode, decode_noise_scale_t = _prepare_decode_timestep_conditioning(
                decode_timestep=decode_timestep,
                decode_noise_scale=decode_noise_scale,
                prompt_batch_size=prompt_batch_size,
                effective_batch_size=latents.shape[0],
                device=device,
                dtype=latents.dtype,
            )
            latents = (1 - decode_noise_scale_t) * latents + decode_noise_scale_t * noise

        dist_initialized = torch.distributed.is_initialized()
        is_output_rank = not dist_initialized or torch.distributed.get_rank() == 0
        vae_decode_needs_all_ranks = False
        is_distributed_vae_enabled = getattr(self.vae, "is_distributed_enabled", None)
        if self.distributed_video_decode and dist_initialized and callable(is_distributed_vae_enabled):
            try:
                # Distributed tiled decode is collective, so every rank must enter it.
                vae_decode_needs_all_ranks = bool(is_distributed_vae_enabled())
            except Exception:
                pass

        should_decode_video = not self.distributed_video_decode or is_output_rank or vae_decode_needs_all_ranks
        if should_decode_video:
            video = self.vae.decode(latents.to(self.vae.dtype), timestep_decode, return_dict=False)[0]
        else:
            video = torch.empty(0, device=latents.device, dtype=latents.dtype)

        if self.distributed_video_decode and not is_output_rank:
            return self._make_output(
                (
                    torch.empty(0, device=video.device, dtype=video.dtype),
                    torch.empty(0, device=audio_latents.device, dtype=audio_latents.dtype),
                )
            )

        if video.numel() > 0:
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        generated_mel = self.audio_vae.decode(audio_latents.to(self.audio_vae.dtype), return_dict=False)[0]
        audio = self.vocoder(generated_mel)
        return self._make_output((video, audio))

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
        latent_num_frames, latent_height, latent_width = latent_ops.resolve_video_latent_shape(
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
        if audio_latents is None or audio_latents.ndim != 4:
            return requested_length

        provided_length = audio_latents.shape[2]
        if not self.preserve_sp_padded_audio_duration:
            return provided_length

        sp_size = getattr(self.od_config.parallel_config, "sequence_parallel_size", 1) or 1
        padded_length = latent_ops.get_sp_padded_audio_latent_length(requested_length, int(sp_size))
        return requested_length if provided_length in {requested_length, padded_length} else provided_length

    def _make_video_audio_scheduler(
        self,
        audio_scheduler: Any,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> Any:
        return VideoAudioScheduler(self.scheduler, audio_scheduler)

    def _prepare_denoise_context_for_cfg(
        self,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> LTXDenoiseContext:
        return self.guidance_strategy.prepare_denoise_context(self, forward_ctx, denoise_ctx)

    def _denoise_timestep_kwargs(
        self,
        ts: torch.Tensor,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> dict[str, torch.Tensor]:
        return self.guidance_strategy.timestep_kwargs(ts, forward_ctx, denoise_ctx)

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

    def _predict_noise_for_step(
        self,
        index: int,
        timestep: torch.Tensor,
        state: latent_ops.LTXAVState,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.guidance_strategy.predict_noise(
            self,
            index,
            timestep,
            state,
            forward_ctx,
            denoise_ctx,
        )

    def _denoise_step(
        self,
        index: int,
        timestep: torch.Tensor,
        state: latent_ops.LTXAVState,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> latent_ops.LTXAVState:
        denoise_ctx.latents = state.video
        denoise_ctx.audio_latents = state.audio
        noise_pred_video, noise_pred_audio = self._predict_noise_for_step(
            index,
            timestep,
            state,
            forward_ctx,
            denoise_ctx,
        )
        video, audio = step_denoised_latents(
            self,
            forward_ctx,
            denoise_ctx,
            noise_pred_video,
            noise_pred_audio,
            timestep,
        )
        return latent_ops.LTXAVState(video=video, audio=audio)

    def _unpack_and_denormalize_stage(
        self,
        forward_ctx: LTXForwardContext,
        latents: torch.Tensor,
        audio_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latents = latent_ops.unpack_latents(
            latents,
            forward_ctx.latent_num_frames,
            forward_ctx.latent_height,
            forward_ctx.latent_width,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )
        latents = latent_ops.denormalize_latents(
            latents,
            self.vae.latents_mean,
            self.vae.latents_std,
            self.vae.config.scaling_factor,
        )

        audio_latents = latent_ops.unpad_audio_latents(audio_latents, forward_ctx.original_audio_num_frames)
        audio_latents = latent_ops.denormalize_audio_latents(
            audio_latents,
            self.audio_vae.latents_mean,
            self.audio_vae.latents_std,
        )
        audio_latents = latent_ops.unpack_audio_latents(
            audio_latents,
            forward_ctx.original_audio_num_frames,
            num_mel_bins=forward_ctx.latent_mel_bins,
        )
        return latents, audio_latents

    def run_phase(
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

    def decode_phase(self, phase: LTXPhaseResult) -> DiffusionOutput | list[DiffusionOutput]:
        """Decode one completed phase and restore per-request outputs."""
        forward_ctx = phase.forward_context
        request_inputs = forward_ctx.request_inputs
        output = self._decode_output(
            latents=phase.video,
            audio_latents=phase.audio,
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

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)
