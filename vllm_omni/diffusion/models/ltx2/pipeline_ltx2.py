# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""One-stage entry points for the LTX model family."""

from __future__ import annotations

from typing import Any, ClassVar

import torch

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.dmd2 import DMD2PipelineMixin
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from .ltx2_components import (
    LTX2_COMPONENT_PROFILE,
    LTX23_COMPONENT_PROFILE,
    LTXComponentProfile,
)
from .ltx2_components import (
    get_ltx2_post_process_func as get_ltx2_post_process_func,  # noqa: F401
)
from .ltx2_conditioning import LTXI2VConditioningMixin
from .ltx2_guidance import (
    LTX_LEGACY_VELOCITY_GUIDANCE,
    LTX_OFFICIAL_X0_GUIDANCE,
)
from .ltx2_pipeline_runtime import LTXPipelineRuntime
from .ltx2_recipes import LTX2_ONE_STAGE_RECIPE, LTX23_ONE_STAGE_RECIPE, LTXOneStageRecipe
from .ltx2_request import LTXRequestInputs


class LTXOneStagePipeline(LTXPipelineRuntime):
    """Single execution path configured by model-version and task entries."""

    component_profile: ClassVar[LTXComponentProfile]
    one_stage_recipe: ClassVar[LTXOneStageRecipe]

    supports_request_batch = True
    supports_guidance_rescale = False

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
    ) -> DiffusionOutput | list[DiffusionOutput]:
        phase = self.run_phase(
            req,
            request_inputs,
            noise_scale=noise_scale,
            sigmas=sigmas,
            timesteps=timesteps,
            attention_kwargs=attention_kwargs,
            image=image,
        )
        return self.decode_phase(phase)

    @torch.no_grad()
    def forward(
        self,
        req: DiffusionRequestBatch,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        frame_rate: float | None = None,
        num_inference_steps: int | None = None,
        sigmas: list[float] | None = None,
        timesteps: list[int] | None = None,
        guidance_scale: float = 4.0,
        guidance_rescale: float = 0.0,
        noise_scale: float = 0.0,
        num_videos_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        decode_timestep: float | list[float] = 0.0,
        decode_noise_scale: float | list[float] | None = None,
        output_type: str = "np",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        max_sequence_length: int | None = None,
        *,
        image: Any | None = None,
    ) -> DiffusionOutput | list[DiffusionOutput]:
        del return_dict
        request_inputs = self._resolve_request_inputs(
            req,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            guidance_scale=self.one_stage_recipe.guidance_scale if guidance_scale is None else guidance_scale,
            guidance_rescale=guidance_rescale,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            latents=latents,
            audio_latents=audio_latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            output_type=output_type,
            max_sequence_length=max_sequence_length,
        )
        image = self._resolve_request_image(req, image, request_inputs)
        forward_kwargs = {
            "noise_scale": noise_scale,
            "sigmas": sigmas,
            "timesteps": timesteps,
            "attention_kwargs": attention_kwargs,
        }
        if self.support_image_input:
            forward_kwargs["image"] = image
        return self._forward_impl(
            req,
            request_inputs,
            **forward_kwargs,
        )


class LTX2Pipeline(LTXOneStagePipeline):
    """LTX2 one-stage text-to-video entry."""

    supports_guidance_rescale = True
    component_profile = LTX2_COMPONENT_PROFILE
    guidance_strategy = LTX_LEGACY_VELOCITY_GUIDANCE
    one_stage_recipe = LTX2_ONE_STAGE_RECIPE
    _dit_modules: ClassVar[list[str]] = list(component_profile.dit_modules)
    _encoder_modules: ClassVar[list[str]] = list(component_profile.encoder_modules)
    _vae_modules: ClassVar[list[str]] = list(component_profile.vae_modules)
    _resident_modules: ClassVar[list[str]] = list(component_profile.resident_modules)


class LTX23Pipeline(LTXOneStagePipeline):
    """LTX2.3 one-stage text-to-video entry."""

    connector_batches_cfg = True
    preserve_sp_padded_audio_duration = True
    reports_stage_durations = True
    component_profile = LTX23_COMPONENT_PROFILE
    guidance_strategy = LTX_OFFICIAL_X0_GUIDANCE
    one_stage_recipe = LTX23_ONE_STAGE_RECIPE
    _dit_modules: ClassVar[list[str]] = list(component_profile.dit_modules)
    _encoder_modules: ClassVar[list[str]] = list(component_profile.encoder_modules)
    _vae_modules: ClassVar[list[str]] = list(component_profile.vae_modules)
    _resident_modules: ClassVar[list[str]] = list(component_profile.resident_modules)

    @torch.no_grad()
    def forward(
        self,
        req: DiffusionRequestBatch,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        frame_rate: float | None = None,
        num_inference_steps: int | None = None,
        sigmas: list[float] | None = None,
        timesteps: list[int] | None = None,
        guidance_scale: float = 4.0,
        noise_scale: float = 0.0,
        num_videos_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        decode_timestep: float | list[float] = 0.0,
        decode_noise_scale: float | list[float] | None = None,
        output_type: str = "np",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        max_sequence_length: int | None = None,
        *,
        image: Any | None = None,
    ) -> DiffusionOutput | list[DiffusionOutput]:
        """Preserve the pre-refactor LTX2.3 positional argument order."""
        return super().forward(
            req,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            sigmas=sigmas,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            guidance_rescale=0.0,
            noise_scale=noise_scale,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            latents=latents,
            audio_latents=audio_latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            output_type=output_type,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
            max_sequence_length=max_sequence_length,
            image=image,
        )


class LTX2ImageToVideoPipeline(LTXI2VConditioningMixin, LTX2Pipeline):
    """LTX2 one-stage image-to-video entry."""


class LTX23ImageToVideoPipeline(LTXI2VConditioningMixin, LTX23Pipeline):
    """LTX2.3 one-stage image-to-video entry."""


class LTX2T2VDMD2Pipeline(DMD2PipelineMixin, LTX2Pipeline):
    """LTX2 T2V entry for FastGen DMD2-distilled models."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.__init_dmd2__()


class LTX2I2VDMD2Pipeline(DMD2PipelineMixin, LTX2ImageToVideoPipeline):
    """LTX2 I2V entry for FastGen DMD2-distilled models."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.__init_dmd2__()
