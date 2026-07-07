# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""LTX-2.3 image-to-video pipeline."""

from __future__ import annotations

from typing import Any

import PIL.Image
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from .pipeline_ltx2_3 import (
    LTX23Pipeline,
    _LTX23DenoiseContext,
    _LTX23ForwardContext,
    _LTX23PromptContext,
    _LTX23RequestInputs,
    get_ltx2_post_process_func,
)
from .pipeline_ltx2_image2video import LTX2ImageToVideoPipeline, _I2VVideoAudioScheduler


class LTX23ImageToVideoPipeline(LTX23Pipeline):
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

    _normalize_latents = staticmethod(LTX2ImageToVideoPipeline._normalize_latents)
    _create_noised_state = staticmethod(LTX2ImageToVideoPipeline._create_noised_state)

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

    def prepare_latents(
        self,
        image: torch.Tensor | None = None,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare I2V latents and the first-frame conditioning mask.

        If caller-provided latents are used without an image, the latents must
        already represent the full video state including the conditioning first
        frame. Packed 3D latents are assumed to be in transformer token layout.
        """
        height = height // self.vae_spatial_compression_ratio
        width = width // self.vae_spatial_compression_ratio
        num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1

        shape = (batch_size, num_channels_latents, num_frames, height, width)
        mask_shape = (batch_size, 1, num_frames, height, width)

        if latents is not None:
            if latents.ndim == 5:
                batch_size, _, num_frames, height, width = latents.shape
                mask_shape = (batch_size, 1, num_frames, height, width)
                conditioning_mask = latents.new_zeros(mask_shape)
                conditioning_mask[:, :, 0] = 1.0

                latents = self._normalize_latents(
                    latents,
                    self.vae.latents_mean,
                    self.vae.latents_std,
                    self.vae.config.scaling_factor,
                )
                latents = self._create_noised_state(latents, noise_scale * (1 - conditioning_mask), generator)
                latents = self._pack_latents(
                    latents,
                    self.transformer_spatial_patch_size,
                    self.transformer_temporal_patch_size,
                )
            else:
                conditioning_mask = latents.new_zeros(mask_shape)
                conditioning_mask[:, :, 0] = 1.0

            conditioning_mask = self._pack_latents(
                conditioning_mask,
                self.transformer_spatial_patch_size,
                self.transformer_temporal_patch_size,
            ).squeeze(-1)
            if latents.ndim != 3 or latents.shape[:2] != conditioning_mask.shape:
                raise ValueError(
                    "Provided `latents` tensor has shape"
                    f" {latents.shape}, but the expected shape is {conditioning_mask.shape + (num_channels_latents,)}."
                )
            return latents.to(device=device, dtype=dtype), conditioning_mask

        if image is None:
            raise ValueError("`image` must be provided when `latents` is None.")

        image_batch_size = image.shape[0]
        if image_batch_size == 0:
            raise ValueError("`image` batch is empty.")
        if batch_size % image_batch_size != 0:
            raise ValueError(
                f"`batch_size` ({batch_size}) must be divisible by image batch size ({image_batch_size}) "
                "for image-to-video outputs."
            )
        num_videos_per_prompt = batch_size // image_batch_size

        if isinstance(generator, list):
            if len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective"
                    f" batch size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
            image_generators = [generator[i * num_videos_per_prompt] for i in range(image_batch_size)]
            init_latents = [
                retrieve_latents(self.vae.encode(image[i].unsqueeze(0).unsqueeze(2)), image_generators[i], "argmax")
                for i in range(image_batch_size)
            ]
        else:
            init_latents = [
                retrieve_latents(self.vae.encode(img.unsqueeze(0).unsqueeze(2)), generator, "argmax") for img in image
            ]

        init_latents = torch.cat(init_latents, dim=0).to(dtype)
        if num_videos_per_prompt > 1:
            init_latents = init_latents.repeat_interleave(num_videos_per_prompt, dim=0)
        init_latents = self._normalize_latents(
            init_latents,
            self.vae.latents_mean,
            self.vae.latents_std,
        )
        init_latents = init_latents.repeat(1, 1, num_frames, 1, 1)

        conditioning_mask = torch.zeros(mask_shape, device=device, dtype=dtype)
        conditioning_mask[:, :, 0] = 1.0

        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = init_latents * conditioning_mask + noise * (1 - conditioning_mask)

        conditioning_mask = self._pack_latents(
            conditioning_mask,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        ).squeeze(-1)
        latents = self._pack_latents(latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size)

        return latents, conditioning_mask

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

    _step_video_latents_i2v = LTX2ImageToVideoPipeline._step_video_latents_i2v

    def _resolve_request_image(
        self,
        req: DiffusionRequestBatch,
        image: PIL.Image.Image | torch.Tensor | list[PIL.Image.Image | torch.Tensor] | None,
        request_inputs: _LTX23RequestInputs,
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
        request_inputs: _LTX23RequestInputs,
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

    def _resolve_video_latent_dimensions(
        self,
        request_inputs: _LTX23RequestInputs,
    ) -> tuple[int, int, int]:
        latent_num_frames = (request_inputs.num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = request_inputs.height // self.vae_spatial_compression_ratio
        latent_width = request_inputs.width // self.vae_spatial_compression_ratio
        if request_inputs.latents is not None:
            if request_inputs.latents.ndim == 5:
                _, _, latent_num_frames, latent_height, latent_width = request_inputs.latents.shape
            elif request_inputs.latents.ndim != 3:
                raise ValueError(
                    f"Provided `latents` tensor has shape {request_inputs.latents.shape}, but the expected shape is "
                    "either [batch_size, seq_len, num_features] or "
                    "[batch_size, latent_dim, latent_frames, latent_height, latent_width]."
                )
        return latent_num_frames, latent_height, latent_width

    def _prepare_video_latents_stage(
        self,
        request_inputs: _LTX23RequestInputs,
        prompt_context: _LTX23PromptContext,
        *,
        device: torch.device,
        noise_scale: float,
        image: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if request_inputs.latents is None:
            if isinstance(image, torch.Tensor):
                if image.ndim == 3:
                    image = image.unsqueeze(0)
            elif isinstance(image, list) and image and isinstance(image[0], torch.Tensor):
                image = torch.stack(image, dim=0)
            else:
                image = self.video_processor.preprocess(
                    image,
                    height=request_inputs.height,
                    width=request_inputs.width,
                )
            image = image.to(device=device, dtype=prompt_context.positive_connector_prompt_embeds.dtype)

        num_channels_latents = self.transformer.config.in_channels
        return self.prepare_latents(
            image,
            prompt_context.batch_size * request_inputs.num_videos_per_prompt,
            num_channels_latents,
            request_inputs.height,
            request_inputs.width,
            request_inputs.num_frames,
            noise_scale,
            torch.float32,
            device,
            request_inputs.generator,
            request_inputs.latents,
        )

    def _make_video_audio_scheduler(
        self,
        audio_scheduler: Any,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> Any:
        return _I2VVideoAudioScheduler(
            self,
            audio_scheduler,
            latent_num_frames,
            latent_height,
            latent_width,
        )

    def _prepare_denoise_context_for_cfg(
        self,
        forward_ctx: _LTX23ForwardContext,
        denoise_ctx: _LTX23DenoiseContext,
    ) -> _LTX23DenoiseContext:
        denoise_ctx = super()._prepare_denoise_context_for_cfg(forward_ctx, denoise_ctx)
        if denoise_ctx.conditioning_mask is None:
            raise ValueError("I2V denoising requires a conditioning mask.")
        if self.do_classifier_free_guidance and not forward_ctx.cfg_parallel_ready:
            denoise_ctx.conditioning_mask_for_model = torch.cat(
                [denoise_ctx.conditioning_mask, denoise_ctx.conditioning_mask]
            )
        else:
            denoise_ctx.conditioning_mask_for_model = denoise_ctx.conditioning_mask
        return denoise_ctx

    def _denoise_timestep_kwargs(
        self,
        ts: torch.Tensor,
        forward_ctx: _LTX23ForwardContext,
        denoise_ctx: _LTX23DenoiseContext,
    ) -> dict[str, torch.Tensor]:
        conditioning_mask = (
            denoise_ctx.conditioning_mask if forward_ctx.cfg_parallel_ready else denoise_ctx.conditioning_mask_for_model
        )
        if conditioning_mask is None:
            raise ValueError("I2V denoising requires a conditioning mask.")
        return {
            "timestep": ts.unsqueeze(-1) * (1 - conditioning_mask),
            "audio_timestep": ts,
            "sigma": ts,
        }

    def _step_denoised_latents(
        self,
        forward_ctx: _LTX23ForwardContext,
        denoise_ctx: _LTX23DenoiseContext,
        noise_pred_video: torch.Tensor,
        noise_pred_audio: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latents, audio_latents = self.scheduler_step_maybe_with_cfg(
            (noise_pred_video, noise_pred_audio),
            (t, t),
            (denoise_ctx.latents, denoise_ctx.audio_latents),
            do_true_cfg=self.do_classifier_free_guidance,
            per_request_scheduler=forward_ctx.video_audio_scheduler,
        )
        return self._synchronize_cfg_parallel_step_output(
            (latents, audio_latents),
            do_true_cfg=self.do_classifier_free_guidance,
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
