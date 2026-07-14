# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LTX-2.3 image-to-video pipeline."""

from __future__ import annotations

from typing import Any

import PIL.Image
import torch
from diffusers.video_processor import VideoProcessor

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from .ltx2_conditioning import LTXI2VConditioningMixin
from .ltx2_recipes import LTX23_ONE_STAGE_RECIPE
from .ltx2_request import LTXRequestInputs
from .pipeline_ltx2_3 import (
    LTX23Pipeline,
    get_ltx2_post_process_func,
)


class LTX23ImageToVideoPipeline(LTXI2VConditioningMixin, LTX23Pipeline):
    """LTX-2.3 image-to-video pipeline.

    This keeps the LTX-2.3 prompt connector, x0-space CFG, sigma prompt
    modulation, and audio branch semantics from ``LTX23Pipeline`` while
    reusing the existing LTX image-conditioning contract: the first video
    latent frame is encoded from the input image and remains fixed during
    denoising.
    """

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio, resample="bilinear")

    support_image_input = True

    @staticmethod
    def _resolve_single_prompt_image(raw_image: Any) -> Any:
        if isinstance(raw_image, list):
            if len(raw_image) != 1:
                raise ValueError(
                    "LTX-2.3 I2V prompt dictionaries support exactly one image per prompt. "
                    "Pass one image per prompt for batched I2V requests."
                )
            return raw_image[0]
        return raw_image

    @staticmethod
    def _resolve_additional_image(additional: dict[str, Any]) -> Any:
        raw_image = additional.get("preprocessed_image")
        if raw_image is None:
            raw_image = additional.get("pixel_values")
        if raw_image is None:
            raw_image = additional.get("image")
        return raw_image

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
        if image is None and latents is None:
            raise ValueError("Provide either `image` or `latents`. Cannot leave both undefined.")
        super().check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

    def _resolve_request_image(
        self,
        req: DiffusionRequestBatch,
        image: PIL.Image.Image | torch.Tensor | list[PIL.Image.Image | torch.Tensor] | None,
        request_inputs: LTXRequestInputs,
    ) -> PIL.Image.Image | torch.Tensor | list[PIL.Image.Image | torch.Tensor] | None:
        if image is not None or not req.prompts:
            return image

        raw_images = []
        for prompt_item in req.prompts:
            if isinstance(prompt_item, str):
                raw_image = None
            else:
                multi_modal_data = prompt_item.get("multi_modal_data") or {}
                raw_image = multi_modal_data.get("image")
                if raw_image is None:
                    additional = prompt_item.get("additional_information") or {}
                    raw_image = self._resolve_additional_image(additional)
            raw_image = self._resolve_single_prompt_image(raw_image)
            if isinstance(raw_image, str):
                raw_image = PIL.Image.open(raw_image).convert("RGB")
            raw_images.append(raw_image)

        if any(img is None for img in raw_images) and request_inputs.latents is None:
            raise ValueError("Image is required for LTX-2.3 I2V generation.")
        if len(raw_images) == 1:
            return raw_images[0]
        if raw_images:
            return raw_images
        return image

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

    @torch.no_grad()
    def forward(
        self,
        req: DiffusionRequestBatch,
        image: PIL.Image.Image | torch.Tensor | list[PIL.Image.Image | torch.Tensor] | None = None,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        frame_rate: float | None = None,
        num_inference_steps: int | None = None,
        sigmas: list[float] | None = None,
        timesteps: list[int] | None = None,
        guidance_scale: float = LTX23_ONE_STAGE_RECIPE.guidance_scale,
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
    ) -> list[DiffusionOutput]:
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
        return self._forward_impl(
            req,
            request_inputs,
            noise_scale=noise_scale,
            sigmas=sigmas,
            timesteps=timesteps,
            attention_kwargs=attention_kwargs,
            image=image,
        )


__all__ = [
    "LTX23ImageToVideoPipeline",
    "get_ltx2_post_process_func",
]
