# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Two-stage entry points for the LTX model family."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, ClassVar

import torch
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from .ltx2_components import (
    LTX2_COMPONENT_PROFILE,
)
from .ltx2_components import (
    get_ltx2_post_process_func as get_ltx2_post_process_func,  # noqa: F401
)
from .ltx2_conditioning import LTXI2VConditioningMixin
from .ltx2_guidance import LTX_LEGACY_VELOCITY_GUIDANCE
from .ltx2_pipeline_runtime import LTXPipelineRuntime
from .ltx2_recipes import LTX2_ONE_STAGE_RECIPE
from .ltx2_request import LTXRequestInputs
from .pipeline_ltx2_latent_upsample import LTX2LatentUpsamplePipeline


class LTX2TwoStagesPipeline(LTXPipelineRuntime):
    """Legacy distilled-only LTX2 two-stage compatibility entry."""

    component_profile = LTX2_COMPONENT_PROFILE
    guidance_strategy = LTX_LEGACY_VELOCITY_GUIDANCE
    one_stage_recipe = LTX2_ONE_STAGE_RECIPE
    supports_request_batch = False
    support_image_input = False

    _dit_modules: ClassVar[list[str]] = list(component_profile.dit_modules)
    _encoder_modules: ClassVar[list[str]] = list(component_profile.encoder_modules)
    _vae_modules: ClassVar[list[str]] = list(component_profile.vae_modules)
    _resident_modules: ClassVar[list[str]] = list(component_profile.resident_modules)

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        model_path = od_config.model
        self.distilled = "distilled" in os.path.basename(os.path.normpath(model_path))
        if not self.distilled:
            raise NotImplementedError(f"{model_path} is not supported for {self.__class__.__name__}.")

        super().__init__(od_config=od_config, prefix=prefix)
        self.upsample_pipe = LTX2LatentUpsamplePipeline(vae=self.vae, od_config=od_config)

    def _run_two_stage(
        self,
        req: DiffusionRequestBatch,
        request_inputs: LTXRequestInputs,
        *,
        noise_scale: float,
        timesteps: list[int] | None,
        attention_kwargs: dict[str, Any] | None,
        image: Any | None = None,
    ) -> DiffusionOutput | list[DiffusionOutput]:
        stage1 = self.run_phase(
            req,
            request_inputs,
            noise_scale=noise_scale,
            sigmas=DISTILLED_SIGMA_VALUES if self.distilled else None,
            timesteps=timesteps,
            attention_kwargs=attention_kwargs,
            image=image,
        )
        upscaled_video_latent = self.upsample_pipe(
            latents=stage1.video,
            output_type="latent",
            return_dict=False,
        )[0]

        stage2_inputs = replace(
            request_inputs,
            num_inference_steps=3,
            guidance_scale=1.0,
            latents=upscaled_video_latent,
            audio_latents=stage1.audio,
            decode_timestep=0.0,
            decode_noise_scale=None,
            output_type="np",
        )
        stage2 = self.run_phase(
            req,
            stage2_inputs,
            noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
            sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
            timesteps=None,
            attention_kwargs=attention_kwargs,
            prompt_context=stage1.forward_context.prompt_context,
        )
        return self.decode_phase(stage2)

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
            guidance_scale=(
                getattr(getattr(self, "one_stage_recipe", None), "guidance_scale", 4.0)
                if guidance_scale is None
                else guidance_scale
            ),
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
        if self.support_image_input:
            image = self._resolve_request_image(req, image, request_inputs)
        return self._run_two_stage(
            req,
            request_inputs,
            noise_scale=noise_scale,
            timesteps=timesteps,
            attention_kwargs=attention_kwargs,
            image=image,
        )


class LTX2ImageToVideoTwoStagesPipeline(LTXI2VConditioningMixin, LTX2TwoStagesPipeline):
    """LTX2 two-stage image-to-video entry."""

    def forward(
        self,
        req: DiffusionRequestBatch,
        image: Any | None = None,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        frame_rate: float | None = None,
        num_inference_steps: int | None = None,
        sigmas: list[float] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> DiffusionOutput | list[DiffusionOutput]:
        """Preserve the legacy I2V sigma slot, which was ignored by this pipeline."""
        del sigmas
        return super().forward(
            req,
            image,
            prompt,
            negative_prompt,
            height,
            width,
            num_frames,
            frame_rate,
            num_inference_steps,
            *args,
            **kwargs,
        )
