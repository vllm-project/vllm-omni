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
from .ltx2_guidance import LTXGuidanceSpec
from .ltx2_pipeline_runtime import LTXRuntime
from .ltx2_recipes import LTX2_ONE_STAGE_RECIPE
from .ltx2_request import LTXRequestInputs
from .pipeline_ltx2_latent_upsample import LTX2LatentUpsamplePipeline


class LTX2TwoStagesPipeline(LTXRuntime):
    """Legacy distilled-only LTX2 two-stage compatibility entry."""

    component_profile = LTX2_COMPONENT_PROFILE
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
        image: Any | None = None,
    ) -> DiffusionOutput | list[DiffusionOutput]:
        stage1 = self.run_phase(
            req,
            request_inputs,
            noise_scale=0.0,
            sigmas=DISTILLED_SIGMA_VALUES if self.distilled else None,
            timesteps=None,
            attention_kwargs=None,
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
            guidance=LTXGuidanceSpec.positive_only(),
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
            attention_kwargs=None,
            prompt_context=stage1.forward_context.prompt_context,
        )
        return self.decode_phase(stage2)

    def _forward_request(
        self,
        req: DiffusionRequestBatch,
        *,
        image: Any | None = None,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        frame_rate: float | None = None,
        num_inference_steps: int | None = None,
        sigmas: list[float] | None = None,
        guidance_scale: float | None = None,
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
        max_sequence_length: int | None = None,
    ) -> DiffusionOutput | list[DiffusionOutput]:
        if sigmas is not None:
            raise ValueError(f"{self.__class__.__name__} uses fixed two-stage sigma schedules.")

        request_inputs = self._resolve_request_inputs(
            req,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
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
        return self._run_two_stage(
            req,
            request_inputs,
            image=image,
        )


class LTX2ImageToVideoTwoStagesPipeline(LTXI2VConditioningMixin, LTX2TwoStagesPipeline):
    """LTX2 two-stage image-to-video entry."""
