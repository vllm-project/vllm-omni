# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Official multi-modal guidance for the LTX model family."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm_omni.diffusion.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)

if TYPE_CHECKING:
    from .ltx2_denoise import LTXDenoiseContext, LTXForwardContext
    from .ltx2_latents import LTXAVState


@dataclass(frozen=True)
class LTXModalityGuidance:
    """Guidance parameters applied to one LTX output modality."""

    cfg_scale: float = 1.0
    stg_scale: float = 0.0
    modality_scale: float = 1.0
    rescale_scale: float = 0.0
    stg_blocks: tuple[int, ...] = ()

    @property
    def do_cfg(self) -> bool:
        return not math.isclose(self.cfg_scale, 1.0)

    @property
    def do_stg(self) -> bool:
        return not math.isclose(self.stg_scale, 0.0) and bool(self.stg_blocks)

    @property
    def do_modality_guidance(self) -> bool:
        return not math.isclose(self.modality_scale, 1.0)


@dataclass(frozen=True)
class LTXGuidanceSpec:
    """Video and audio guidance requested for one denoise phase."""

    video: LTXModalityGuidance = LTXModalityGuidance()
    audio: LTXModalityGuidance = LTXModalityGuidance()

    @classmethod
    def positive_only(cls) -> LTXGuidanceSpec:
        return cls()

    @property
    def do_cfg(self) -> bool:
        return self.video.do_cfg or self.audio.do_cfg

    @property
    def do_stg(self) -> bool:
        return self.video.do_stg or self.audio.do_stg

    @property
    def do_modality_guidance(self) -> bool:
        return self.video.do_modality_guidance or self.audio.do_modality_guidance

    @property
    def do_rescale(self) -> bool:
        return self.video.rescale_scale != 0.0 or self.audio.rescale_scale != 0.0


@dataclass(frozen=True)
class LTXDenoisePass:
    """One conditioned Transformer evaluation in a guidance batch."""

    name: str
    negative_video_context: bool = False
    negative_audio_context: bool = False


@dataclass(frozen=True)
class LTXGuidancePlan:
    """Concrete Transformer passes derived from a guidance specification."""

    spec: LTXGuidanceSpec
    passes: tuple[LTXDenoisePass, ...]

    @classmethod
    def build(cls, spec: LTXGuidanceSpec) -> LTXGuidancePlan:
        passes = [LTXDenoisePass("cond")]
        if spec.do_cfg:
            passes.append(
                LTXDenoisePass(
                    "uncond",
                    negative_video_context=spec.video.do_cfg,
                    negative_audio_context=spec.audio.do_cfg,
                )
            )
        if spec.do_stg:
            passes.append(LTXDenoisePass("ptb"))
        if spec.do_modality_guidance:
            passes.append(LTXDenoisePass("mod"))
        return cls(spec=spec, passes=tuple(passes))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.passes)

    @property
    def cfg_parallel_compatible(self) -> bool:
        return self.names == ("cond", "uncond") and not self.spec.do_rescale


def x0_from_velocity(sample: torch.Tensor, velocity: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Convert velocity to x0 using the official fp32 arithmetic order."""
    sigma = sigma.to(torch.float32)
    return (sample.to(torch.float32) - velocity.to(torch.float32) * sigma).to(sample.dtype)


def velocity_from_x0(sample: torch.Tensor, x0: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Convert x0 back to velocity using the official fp32 arithmetic order."""
    sigma = sigma.to(torch.float32)
    return ((sample.to(torch.float32) - x0.to(torch.float32)) / sigma).to(sample.dtype)


def euler_step_from_velocity(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigmas: torch.Tensor,
    step_index: int,
) -> torch.Tensor:
    sigma = sigmas[step_index].to(torch.float32)
    sigma_next = sigmas[step_index + 1].to(torch.float32)
    return (sample.float() + velocity.float() * (sigma_next - sigma)).to(sample.dtype)


def combine_guided_x0(
    *,
    cond: torch.Tensor,
    uncond_text: torch.Tensor | float,
    uncond_perturbed: torch.Tensor | float,
    uncond_modality: torch.Tensor | float,
    guidance: LTXModalityGuidance,
) -> torch.Tensor:
    dtype = cond.dtype
    cond = cond.float()
    uncond_text = uncond_text.float() if isinstance(uncond_text, torch.Tensor) else uncond_text
    uncond_perturbed = uncond_perturbed.float() if isinstance(uncond_perturbed, torch.Tensor) else uncond_perturbed
    uncond_modality = uncond_modality.float() if isinstance(uncond_modality, torch.Tensor) else uncond_modality
    pred = cond + (guidance.cfg_scale - 1) * (cond - uncond_text)
    if guidance.do_stg:
        pred = pred + guidance.stg_scale * (cond - uncond_perturbed)
    pred = pred + (guidance.modality_scale - 1) * (cond - uncond_modality)
    if guidance.rescale_scale != 0.0:
        # The pinned official one-stage entry runs one generated sample and
        # uses its full-tensor standard deviation. In Omni, dimension 0 may
        # instead contain independently scheduled requests. Reduce each item
        # separately so its result cannot depend on unrelated co-batched work.
        reduce_dims = tuple(range(1, pred.ndim))
        cond_std = cond.std(dim=reduce_dims, keepdim=True)
        pred_std = pred.std(dim=reduce_dims, keepdim=True)
        eps = torch.finfo(pred.dtype).eps
        factor = cond_std / pred_std.clamp_min(eps)
        factor = torch.where(pred_std > eps, factor, torch.ones_like(factor))
        factor = guidance.rescale_scale * factor + (1 - guidance.rescale_scale)
        pred = pred * factor
    return pred.to(dtype)


def _repeat_batch(tensor: torch.Tensor, repeats: int) -> torch.Tensor:
    return tensor.repeat((repeats,) + (1,) * (tensor.ndim - 1))


def _mask_for_pass(
    pass_count: int,
    batch_size: int,
    pass_index: int,
    reference: torch.Tensor,
) -> torch.Tensor:
    mask = reference.new_ones((pass_count * batch_size, 1, 1))
    start = pass_index * batch_size
    mask[start : start + batch_size] = 0
    return mask


def build_perturbation_kwargs(
    plan: LTXGuidancePlan,
    batch_size: int,
    reference: torch.Tensor,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if "ptb" in plan.names:
        mask = _mask_for_pass(len(plan.passes), batch_size, plan.names.index("ptb"), reference)
        if plan.spec.video.do_stg:
            kwargs["video_self_attention_mask"] = mask
            kwargs["video_self_attention_blocks"] = plan.spec.video.stg_blocks
        if plan.spec.audio.do_stg:
            kwargs["audio_self_attention_mask"] = mask
            kwargs["audio_self_attention_blocks"] = plan.spec.audio.stg_blocks
    if "mod" in plan.names:
        mask = _mask_for_pass(len(plan.passes), batch_size, plan.names.index("mod"), reference)
        kwargs["a2v_cross_attention_mask"] = mask
        kwargs["v2a_cross_attention_mask"] = mask
    return kwargs


class LTXGuidanceExecutor:
    """Build and execute official LTX multi-guidance Transformer passes."""

    @staticmethod
    def validate_cfg_world_size(plan: LTXGuidancePlan, cfg_world_size: int) -> None:
        if cfg_world_size == 1:
            return
        if cfg_world_size != 2:
            raise ValueError(f"LTX CFG parallelism supports cfg_parallel_size 1 or 2, got {cfg_world_size}.")
        if not plan.cfg_parallel_compatible:
            raise ValueError("LTX CFG parallelism only supports CFG-only guidance without rescale.")

    @staticmethod
    def prepare_denoise_context(
        plan: LTXGuidancePlan,
        cfg_parallel_ready: bool,
        denoise_ctx: LTXDenoiseContext,
    ) -> LTXDenoiseContext:
        if len(plan.passes) > 1 and not cfg_parallel_ready:
            denoise_ctx.video_coords = _repeat_batch(denoise_ctx.video_coords, len(plan.passes))
            denoise_ctx.audio_coords = _repeat_batch(denoise_ctx.audio_coords, len(plan.passes))
        return denoise_ctx

    @staticmethod
    def timestep_kwargs(
        ts: torch.Tensor,
        video_token_count: int,
        audio_token_count: int,
    ) -> dict[str, torch.Tensor]:
        return {
            "timestep": ts.reshape(-1, 1).expand(-1, video_token_count),
            "audio_timestep": ts.reshape(-1, 1).expand(-1, audio_token_count),
            "sigma": ts,
            "audio_sigma": ts,
        }

    @staticmethod
    def combine_cfg_velocity(
        sample: torch.Tensor,
        positive_velocity: torch.Tensor,
        negative_velocity: torch.Tensor,
        sigma: torch.Tensor,
        guidance_scale: float,
        *,
        model_sigma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        guidance = LTXModalityGuidance(cfg_scale=guidance_scale)
        model_sigma = sigma if model_sigma is None else model_sigma
        guided_x0 = combine_guided_x0(
            cond=x0_from_velocity(sample, positive_velocity, model_sigma),
            uncond_text=x0_from_velocity(sample, negative_velocity, model_sigma),
            uncond_perturbed=0.0,
            uncond_modality=0.0,
            guidance=guidance,
        )
        return velocity_from_x0(sample, guided_x0, sigma)

    def predict_parallel_cfg(
        self,
        pipeline: Any,
        plan: LTXGuidancePlan,
        index: int,
        timestep: torch.Tensor,
        state: LTXAVState,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.validate_cfg_world_size(plan, get_classifier_free_guidance_world_size())
        prompt = forward_ctx.prompt_context
        negative = get_classifier_free_guidance_rank() == 1
        video_context = (
            prompt.negative_connector_prompt_embeds
            if negative and plan.spec.video.do_cfg
            else prompt.positive_connector_prompt_embeds
        )
        audio_context = (
            prompt.negative_connector_audio_prompt_embeds
            if negative and plan.spec.audio.do_cfg
            else prompt.positive_connector_audio_prompt_embeds
        )
        if video_context is None or audio_context is None:
            raise ValueError("Negative prompt context is required when LTX CFG is enabled.")
        video_input = state.video.to(prompt.positive_connector_prompt_embeds.dtype)
        audio_input = state.audio.to(prompt.positive_connector_audio_prompt_embeds.dtype)
        kwargs = pipeline._build_transformer_kwargs(
            forward_ctx,
            denoise_ctx,
            hidden_states=video_input,
            audio_hidden_states=audio_input,
            encoder_hidden_states=video_context,
            audio_encoder_hidden_states=audio_context,
            encoder_attention_mask=None,
            audio_encoder_attention_mask=None,
            ts=timestep.expand(video_input.shape[0]),
        )
        local_video, local_audio = pipeline.predict_noise(**kwargs)
        group = get_cfg_group()
        video = group.all_gather(local_video, separate_tensors=True)
        audio = group.all_gather(local_audio, separate_tensors=True)
        video_sigma = pipeline.scheduler.sigmas[index]
        return (
            self.combine_cfg_velocity(
                state.video,
                video[0],
                video[1],
                video_sigma,
                plan.spec.video.cfg_scale,
                model_sigma=pipeline._video_guidance_model_sigma(video_sigma, denoise_ctx),
            ),
            self.combine_cfg_velocity(
                state.audio,
                audio[0],
                audio[1],
                forward_ctx.audio_scheduler.sigmas[index],
                plan.spec.audio.cfg_scale,
            ),
        )

    def predict_noise(
        self,
        pipeline: Any,
        plan: LTXGuidancePlan,
        index: int,
        timestep: torch.Tensor,
        state: LTXAVState,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if forward_ctx.cfg_parallel_ready:
            return self.predict_parallel_cfg(pipeline, plan, index, timestep, state, forward_ctx, denoise_ctx)

        prompt = forward_ctx.prompt_context
        video_contexts: list[torch.Tensor] = []
        audio_contexts: list[torch.Tensor] = []
        for denoise_pass in plan.passes:
            video_context = (
                prompt.negative_connector_prompt_embeds
                if denoise_pass.negative_video_context
                else prompt.positive_connector_prompt_embeds
            )
            audio_context = (
                prompt.negative_connector_audio_prompt_embeds
                if denoise_pass.negative_audio_context
                else prompt.positive_connector_audio_prompt_embeds
            )
            if video_context is None or audio_context is None:
                raise ValueError("Negative prompt context is required when LTX CFG is enabled.")
            video_contexts.append(video_context)
            audio_contexts.append(audio_context)

        pass_count = len(plan.passes)
        video_input = _repeat_batch(state.video, pass_count).to(prompt.positive_connector_prompt_embeds.dtype)
        audio_input = _repeat_batch(state.audio, pass_count).to(prompt.positive_connector_audio_prompt_embeds.dtype)
        attention_kwargs = dict(forward_ctx.attention_kwargs or {})
        perturbations = build_perturbation_kwargs(plan, state.video.shape[0], video_input)
        if perturbations:
            attention_kwargs["ltx_perturbation_kwargs"] = perturbations
        kwargs = pipeline._build_transformer_kwargs(
            forward_ctx,
            denoise_ctx,
            hidden_states=video_input,
            audio_hidden_states=audio_input,
            encoder_hidden_states=torch.cat(video_contexts),
            audio_encoder_hidden_states=torch.cat(audio_contexts),
            encoder_attention_mask=None,
            audio_encoder_attention_mask=None,
            ts=timestep.expand(video_input.shape[0]),
            attention_kwargs=attention_kwargs,
        )
        with pipeline._transformer_cache_context("guided"):
            video_velocity, audio_velocity = pipeline.transformer(**kwargs)
        video_splits = dict(zip(plan.names, video_velocity.chunk(pass_count), strict=True))
        audio_splits = dict(zip(plan.names, audio_velocity.chunk(pass_count), strict=True))

        def guide_modality(
            sample: torch.Tensor,
            splits: dict[str, torch.Tensor],
            sigma: torch.Tensor,
            guidance: LTXModalityGuidance,
            *,
            model_sigma: torch.Tensor | None = None,
        ) -> torch.Tensor:
            model_sigma = sigma if model_sigma is None else model_sigma
            # Official guidance reduces contiguous BSC tensors. The fused
            # transformer output may be channel-major after splitting, which
            # changes fp32 reduction order and can cross bf16 rounding bounds.
            x0 = {name: x0_from_velocity(sample, value, model_sigma).contiguous() for name, value in splits.items()}
            guided = combine_guided_x0(
                cond=x0["cond"],
                uncond_text=x0.get("uncond", 0.0),
                uncond_perturbed=x0.get("ptb", 0.0),
                uncond_modality=x0.get("mod", 0.0),
                guidance=guidance,
            )
            return velocity_from_x0(sample, guided, sigma)

        return (
            guide_modality(
                state.video,
                video_splits,
                pipeline.scheduler.sigmas[index],
                plan.spec.video,
                model_sigma=pipeline._video_guidance_model_sigma(
                    pipeline.scheduler.sigmas[index],
                    denoise_ctx,
                ),
            ),
            guide_modality(
                state.audio,
                audio_splits,
                forward_ctx.audio_scheduler.sigmas[index],
                plan.spec.audio,
            ),
        )


LTX_GUIDANCE_EXECUTOR = LTXGuidanceExecutor()

# Kept as a narrow compatibility helper for CFG-parallel tests and extensions.
combine_velocity_via_x0 = LTXGuidanceExecutor.combine_cfg_velocity
