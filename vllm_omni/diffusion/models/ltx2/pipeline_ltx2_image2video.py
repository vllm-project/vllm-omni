# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import PIL.Image
import torch
from diffusers.video_processor import VideoProcessor
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.dmd2 import DMD2PipelineMixin
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from .ltx2_conditioning import LTXI2VConditioningMixin
from .ltx2_denoise import LTXDenoiseContext, LTXForwardContext
from .ltx2_recipes import LTX2_ONE_STAGE_RECIPE
from .pipeline_ltx2 import (
    LTX2Pipeline,
    LTX2TwoStagesPipeline,
)
from .pipeline_ltx2 import (
    get_ltx2_post_process_func as _get_ltx2_post_process_func,
)

if TYPE_CHECKING:
    from .ltx2_pipeline_base import LTXRequestInputs

logger = init_logger(__name__)


def get_ltx2_post_process_func(od_config: OmniDiffusionConfig):
    return _get_ltx2_post_process_func(od_config)


class LTX2ImageToVideoPipeline(LTXI2VConditioningMixin, LTX2Pipeline):
    supports_request_batch = False
    support_image_input = True

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__(od_config=od_config, prefix=prefix)
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio, resample="bilinear")

    def check_inputs(
        self,
        image,
        height,
        width,
        prompt,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
    ):
        if height % 32 != 0 or width % 32 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 32 but are {height} and {width}.")

        if image is None and latents is None:
            raise ValueError("Provide either `image` or `latents`. Cannot leave both undefined.")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        if prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        if prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")

        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise ValueError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            if prompt_attention_mask.shape != negative_prompt_attention_mask.shape:
                raise ValueError(
                    "`prompt_attention_mask` and `negative_prompt_attention_mask` must have the same shape when"
                    " passed directly, but"
                    f" got: `prompt_attention_mask` {prompt_attention_mask.shape} != `negative_prompt_attention_mask`"
                    f" {negative_prompt_attention_mask.shape}."
                )

    @staticmethod
    def _resolve_request_image(
        req: DiffusionRequestBatch,
        image: PIL.Image.Image | torch.Tensor | None,
        latents: torch.Tensor | None,
    ) -> Any:
        if image is not None or not req.prompts:
            return image

        images = []
        for prompt_item in req.prompts:
            if isinstance(prompt_item, str):
                raw_image = None
            else:
                multi_modal_data = prompt_item.get("multi_modal_data") or {}
                raw_image = multi_modal_data.get("image")
                if raw_image is None:
                    additional = prompt_item.get("additional_information") or {}
                    raw_image = (
                        additional.get("preprocessed_image")
                        or additional.get("pixel_values")
                        or additional.get("image")
                    )
            if isinstance(raw_image, list):
                raw_image = raw_image[0] if raw_image else None
            if isinstance(raw_image, str):
                raw_image = PIL.Image.open(raw_image).convert("RGB")
            images.append(raw_image)

        if any(item is None for item in images) and latents is None:
            raise ValueError("Image is required for LTX2 I2V generation.")
        if len(images) == 1:
            return images[0]
        return images

    def _check_forward_inputs(
        self,
        request_inputs: LTXRequestInputs,
        image: Any | None = None,
    ) -> None:
        self.check_inputs(
            image=image,
            height=request_inputs.height,
            width=request_inputs.width,
            prompt=request_inputs.prompt,
            latents=request_inputs.latents,
            prompt_embeds=request_inputs.prompt_embeds,
            negative_prompt_embeds=request_inputs.negative_prompt_embeds,
            prompt_attention_mask=request_inputs.prompt_attention_mask,
            negative_prompt_attention_mask=request_inputs.negative_prompt_attention_mask,
        )

    def _denoise_timestep_kwargs(
        self,
        ts: torch.Tensor,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> dict[str, torch.Tensor]:
        if denoise_ctx.conditioning_mask is None:
            raise ValueError("LTX2 I2V requires a video conditioning mask.")
        return {
            "timestep": ts.unsqueeze(-1) * (1 - denoise_ctx.conditioning_mask),
            "audio_timestep": ts,
        }

    @torch.no_grad()
    def forward(
        self,
        req: DiffusionRequestBatch,
        image: PIL.Image.Image | torch.Tensor | None = None,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        frame_rate: float | None = None,
        num_inference_steps: int | None = None,
        sigmas: list[float] | None = None,
        timesteps: list[int] | None = None,
        guidance_scale: float = LTX2_ONE_STAGE_RECIPE.guidance_scale,
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
    ) -> DiffusionOutput:
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
            guidance_scale=guidance_scale,
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
        image = self._resolve_request_image(req, image, request_inputs.latents)
        return self._forward_impl(
            req,
            request_inputs,
            noise_scale=noise_scale,
            sigmas=sigmas,
            timesteps=timesteps,
            attention_kwargs=attention_kwargs,
            image=image,
        )


class LTX2ImageToVideoTwoStagesPipeline(LTX2TwoStagesPipeline):
    """LTXImageToVideoTwoStagesPipeline is for two stages image to video generation"""

    support_image_input = True
    one_stage_pipeline_cls = LTX2ImageToVideoPipeline

    @torch.no_grad()
    def forward(
        self,
        req: DiffusionRequestBatch,
        image: PIL.Image.Image | torch.Tensor | None = None,
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
    ) -> DiffusionOutput:
        request_inputs = self.pipe._resolve_request_inputs(
            req,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
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
        image = self.pipe._resolve_request_image(req, image, request_inputs.latents)
        return self._run_two_stage(
            req,
            request_inputs,
            noise_scale=noise_scale,
            timesteps=timesteps,
            attention_kwargs=attention_kwargs,
            image=image,
        )


class LTX2I2VDMD2Pipeline(DMD2PipelineMixin, LTX2ImageToVideoPipeline):
    """LTX-2 I2V pipeline for FastGen DMD2-distilled models."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.__init_dmd2__()
