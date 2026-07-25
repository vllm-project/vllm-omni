# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Guidance strategies for the LTX model family."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg

from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)

if TYPE_CHECKING:
    from .ltx2_denoise import LTXDenoiseContext, LTXForwardContext
    from .ltx2_latents import LTXAVState


class LTXGuidanceStrategy(Protocol):
    """Model-version guidance behavior consumed by the shared pipeline."""

    def validate_cfg_world_size(self, cfg_world_size: int) -> None: ...

    def prepare_denoise_context(
        self,
        pipeline: Any,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> LTXDenoiseContext: ...

    def timestep_kwargs(
        self,
        ts: torch.Tensor,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> dict[str, torch.Tensor]: ...

    def combine_cfg_noise(
        self,
        pipeline: Any,
        positive_noise_pred: tuple[torch.Tensor, torch.Tensor],
        negative_noise_pred: tuple[torch.Tensor, torch.Tensor],
        true_cfg_scale: float,
        cfg_normalize: bool,
        context: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def predict_noise(
        self,
        pipeline: Any,
        index: int,
        timestep: torch.Tensor,
        state: LTXAVState,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class _LTXGuidanceStrategyBase:
    def validate_cfg_world_size(self, cfg_world_size: int) -> None:
        del cfg_world_size

    def prepare_denoise_context(
        self,
        pipeline: Any,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> LTXDenoiseContext:
        del pipeline, forward_ctx
        return denoise_ctx

    def timestep_kwargs(
        self,
        ts: torch.Tensor,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> dict[str, torch.Tensor]:
        del forward_ctx, denoise_ctx
        return {"timestep": ts}


def combine_velocity_via_x0(
    sample: torch.Tensor,
    positive_velocity: torch.Tensor,
    negative_velocity: torch.Tensor,
    sigma: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Apply CFG in x0 space and convert the result back to velocity."""
    x0_cond = sample - positive_velocity * sigma
    x0_uncond = sample - negative_velocity * sigma
    x0_guided = x0_cond + (guidance_scale - 1) * (x0_cond - x0_uncond)
    return (sample - x0_guided) / sigma


class LTXLegacyVelocityGuidance(_LTXGuidanceStrategyBase):
    """Existing LTX2 Diffusers-style velocity-space CFG behavior."""

    def combine_cfg_noise(
        self,
        pipeline: Any,
        positive_noise_pred: tuple[torch.Tensor, torch.Tensor],
        negative_noise_pred: tuple[torch.Tensor, torch.Tensor],
        true_cfg_scale: float,
        cfg_normalize: bool,
        context: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del context
        video_pos, audio_pos = positive_noise_pred
        video_neg, audio_neg = negative_noise_pred
        video_combined = CFGParallelMixin.combine_cfg_noise(
            pipeline,
            video_pos,
            video_neg,
            true_cfg_scale,
            cfg_normalize,
        )
        audio_combined = CFGParallelMixin.combine_cfg_noise(
            pipeline,
            audio_pos,
            audio_neg,
            true_cfg_scale,
            cfg_normalize,
        )
        if pipeline.guidance_rescale and pipeline.guidance_rescale > 0:
            video_combined = rescale_noise_cfg(
                video_combined,
                video_pos,
                guidance_rescale=pipeline.guidance_rescale,
            )
            audio_combined = rescale_noise_cfg(
                audio_combined,
                audio_pos,
                guidance_rescale=pipeline.guidance_rescale,
            )
        return video_combined, audio_combined

    def predict_noise(
        self,
        pipeline: Any,
        index: int,
        timestep: torch.Tensor,
        state: LTXAVState,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del index
        prompt_context = forward_ctx.prompt_context
        video_input = state.video.to(prompt_context.positive_connector_prompt_embeds.dtype)
        audio_input = state.audio.to(prompt_context.positive_connector_prompt_embeds.dtype)
        expanded_timestep = timestep.expand(video_input.shape[0])
        positive_kwargs = pipeline._build_transformer_kwargs(
            forward_ctx,
            denoise_ctx,
            hidden_states=video_input,
            audio_hidden_states=audio_input,
            encoder_hidden_states=prompt_context.positive_connector_prompt_embeds,
            audio_encoder_hidden_states=prompt_context.positive_connector_audio_prompt_embeds,
            encoder_attention_mask=prompt_context.positive_connector_attention_mask,
            audio_encoder_attention_mask=prompt_context.positive_connector_attention_mask,
            ts=expanded_timestep,
        )
        negative_kwargs = (
            {
                **positive_kwargs,
                "encoder_hidden_states": prompt_context.negative_connector_prompt_embeds,
                "audio_encoder_hidden_states": prompt_context.negative_connector_audio_prompt_embeds,
                "encoder_attention_mask": prompt_context.negative_connector_attention_mask,
                "audio_encoder_attention_mask": prompt_context.negative_connector_attention_mask,
            }
            if pipeline.do_classifier_free_guidance
            else None
        )
        return pipeline.predict_noise_maybe_with_cfg(
            do_true_cfg=pipeline.do_classifier_free_guidance,
            true_cfg_scale=forward_ctx.request_inputs.guidance_scale,
            positive_kwargs=positive_kwargs,
            negative_kwargs=negative_kwargs,
            cfg_normalize=False,
        )


class LTXOfficialX0Guidance(_LTXGuidanceStrategyBase):
    """Official LTX x0-space CFG behavior currently used by LTX2.3."""

    def validate_cfg_world_size(self, cfg_world_size: int) -> None:
        if cfg_world_size not in (1, 2):
            raise ValueError(
                f"LTX x0 guidance supports CFG parallelism with cfg_parallel_size 1 or 2, but got {cfg_world_size}."
            )

    def prepare_denoise_context(
        self,
        pipeline: Any,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> LTXDenoiseContext:
        if pipeline.do_classifier_free_guidance and not forward_ctx.cfg_parallel_ready:
            denoise_ctx.video_coords = denoise_ctx.video_coords.repeat(
                (2,) + (1,) * (denoise_ctx.video_coords.ndim - 1)
            )
            denoise_ctx.audio_coords = denoise_ctx.audio_coords.repeat(
                (2,) + (1,) * (denoise_ctx.audio_coords.ndim - 1)
            )
        return denoise_ctx

    def timestep_kwargs(
        self,
        ts: torch.Tensor,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> dict[str, torch.Tensor]:
        del forward_ctx, denoise_ctx
        return {"timestep": ts, "sigma": ts}

    def combine_cfg_noise(
        self,
        pipeline: Any,
        positive_noise_pred: tuple[torch.Tensor, torch.Tensor],
        negative_noise_pred: tuple[torch.Tensor, torch.Tensor],
        true_cfg_scale: float,
        cfg_normalize: bool,
        context: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        required = ("video_latents", "audio_latents", "video_sigma", "audio_sigma")
        if any(context.get(name) is None for name in required):
            raise ValueError("LTX x0-space CFG requires video/audio latents and sigmas.")

        video_pos, audio_pos = positive_noise_pred
        video_neg, audio_neg = negative_noise_pred
        video_combined = combine_velocity_via_x0(
            context["video_latents"],
            video_pos,
            video_neg,
            context["video_sigma"],
            true_cfg_scale,
        )
        audio_combined = combine_velocity_via_x0(
            context["audio_latents"],
            audio_pos,
            audio_neg,
            context["audio_sigma"],
            true_cfg_scale,
        )
        if cfg_normalize:
            video_combined = pipeline.cfg_normalize_function(video_pos, video_combined)
            audio_combined = pipeline.cfg_normalize_function(audio_pos, audio_combined)
        return video_combined, audio_combined

    def predict_parallel_cfg(
        self,
        pipeline: Any,
        true_cfg_scale: float,
        positive_kwargs: dict[str, Any],
        negative_kwargs: dict[str, Any],
        cfg_normalize: bool = True,
        output_slice: int | None = None,
        **context: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def maybe_slice(pred: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            if output_slice is None:
                return pred
            return pred[0][:, :output_slice], pred[1][:, :output_slice]

        cfg_world_size = get_classifier_free_guidance_world_size()
        self.validate_cfg_world_size(cfg_world_size)
        if cfg_world_size != 2:
            raise ValueError(f"Parallel LTX x0 guidance requires cfg_parallel_size 2, but got {cfg_world_size}.")

        cfg_rank = get_classifier_free_guidance_rank()
        branch_kwargs = positive_kwargs if cfg_rank == 0 else negative_kwargs
        local_video_pred, local_audio_pred = maybe_slice(pipeline.predict_noise(**branch_kwargs))
        cfg_group = get_cfg_group()
        gathered_video = cfg_group.all_gather(local_video_pred, separate_tensors=True)
        gathered_audio = cfg_group.all_gather(local_audio_pred, separate_tensors=True)
        return self.combine_cfg_noise(
            pipeline,
            (gathered_video[0], gathered_audio[0]),
            (gathered_video[1], gathered_audio[1]),
            true_cfg_scale,
            cfg_normalize,
            context,
        )

    def predict_noise(
        self,
        pipeline: Any,
        index: int,
        timestep: torch.Tensor,
        state: LTXAVState,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_context = forward_ctx.prompt_context
        guidance_scale = forward_ctx.request_inputs.guidance_scale
        audio_scheduler = forward_ctx.audio_scheduler
        if forward_ctx.cfg_parallel_ready:
            video_input = state.video.to(prompt_context.positive_connector_prompt_embeds.dtype)
            audio_input = state.audio.to(prompt_context.positive_connector_prompt_embeds.dtype)
            ts = timestep.expand(video_input.shape[0])
            positive_kwargs = pipeline._build_transformer_kwargs(
                forward_ctx,
                denoise_ctx,
                hidden_states=video_input,
                audio_hidden_states=audio_input,
                encoder_hidden_states=prompt_context.positive_connector_prompt_embeds,
                audio_encoder_hidden_states=prompt_context.positive_connector_audio_prompt_embeds,
                encoder_attention_mask=prompt_context.positive_connector_attention_mask,
                audio_encoder_attention_mask=prompt_context.positive_connector_attention_mask,
                ts=ts,
            )
            negative_kwargs = {
                **positive_kwargs,
                "encoder_hidden_states": prompt_context.negative_connector_prompt_embeds,
                "audio_encoder_hidden_states": prompt_context.negative_connector_audio_prompt_embeds,
                "encoder_attention_mask": prompt_context.negative_connector_attention_mask,
                "audio_encoder_attention_mask": prompt_context.negative_connector_attention_mask,
            }
            return self.predict_parallel_cfg(
                pipeline,
                true_cfg_scale=guidance_scale,
                positive_kwargs=positive_kwargs,
                negative_kwargs=negative_kwargs,
                cfg_normalize=False,
                video_latents=state.video,
                audio_latents=state.audio,
                video_sigma=pipeline.scheduler.sigmas[index],
                audio_sigma=audio_scheduler.sigmas[index],
            )

        video_input = torch.cat([state.video] * 2) if pipeline.do_classifier_free_guidance else state.video
        video_input = video_input.to(prompt_context.connector_prompt_embeds.dtype)
        audio_input = torch.cat([state.audio] * 2) if pipeline.do_classifier_free_guidance else state.audio
        audio_input = audio_input.to(prompt_context.connector_prompt_embeds.dtype)
        ts = timestep.expand(video_input.shape[0])
        transformer_kwargs = pipeline._build_transformer_kwargs(
            forward_ctx,
            denoise_ctx,
            hidden_states=video_input,
            audio_hidden_states=audio_input,
            encoder_hidden_states=prompt_context.connector_prompt_embeds,
            audio_encoder_hidden_states=prompt_context.connector_audio_prompt_embeds,
            encoder_attention_mask=prompt_context.connector_attention_mask,
            audio_encoder_attention_mask=prompt_context.connector_attention_mask,
            ts=ts,
        )
        with pipeline._transformer_cache_context("cond_uncond"):
            video_pred, audio_pred = pipeline.transformer(**transformer_kwargs)
        video_pred = video_pred.float()
        audio_pred = audio_pred.float()

        if pipeline.do_classifier_free_guidance:
            video_uncond, video_cond = video_pred.chunk(2)
            video_pred = combine_velocity_via_x0(
                state.video,
                video_cond,
                video_uncond,
                pipeline.scheduler.sigmas[index],
                guidance_scale,
            )
            audio_uncond, audio_cond = audio_pred.chunk(2)
            audio_pred = combine_velocity_via_x0(
                state.audio,
                audio_cond,
                audio_uncond,
                audio_scheduler.sigmas[index],
                guidance_scale,
            )
        return video_pred, audio_pred


LTX_LEGACY_VELOCITY_GUIDANCE = LTXLegacyVelocityGuidance()
LTX_OFFICIAL_X0_GUIDANCE = LTXOfficialX0Guidance()
