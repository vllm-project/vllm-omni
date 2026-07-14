# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Task-specific conditioning shared by LTX model versions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents
from diffusers.utils.torch_utils import randn_tensor

from .ltx2_denoise import I2VVideoAudioScheduler

if TYPE_CHECKING:
    from .ltx2_pipeline_base import LTXPromptContext, LTXRequestInputs


class LTXI2VConditioningMixin:
    """First-frame conditioning behavior common to LTX2 and LTX2.3."""

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
        num_frames, height, width = self._resolve_video_latent_shape(
            height,
            width,
            num_frames,
            vae_spatial_compression_ratio=self.vae_spatial_compression_ratio,
            vae_temporal_compression_ratio=self.vae_temporal_compression_ratio,
        )
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
                latents = self._create_noised_state(
                    latents,
                    noise_scale * (1 - conditioning_mask),
                    generator,
                )
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
                retrieve_latents(
                    self.vae.encode(image[i].unsqueeze(0).unsqueeze(2)),
                    image_generators[i],
                    "argmax",
                )
                for i in range(image_batch_size)
            ]
        else:
            init_latents = [
                retrieve_latents(
                    self.vae.encode(img.unsqueeze(0).unsqueeze(2)),
                    generator,
                    "argmax",
                )
                for img in image
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
        latents = self._pack_latents(
            latents,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )
        return latents, conditioning_mask

    def _step_video_latents_i2v(
        self,
        noise_pred_video: torch.Tensor,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> torch.Tensor:
        noise_pred_video = self._unpack_latents(
            noise_pred_video,
            latent_num_frames,
            latent_height,
            latent_width,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )
        latents_unpacked = self._unpack_latents(
            latents,
            latent_num_frames,
            latent_height,
            latent_width,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )
        noise_pred_video = noise_pred_video[:, :, 1:]
        noise_latents = latents_unpacked[:, :, 1:]
        pred_latents = self.scheduler.step(
            noise_pred_video,
            timestep,
            noise_latents,
            return_dict=False,
        )[0]
        latents_unpacked = torch.cat([latents_unpacked[:, :, :1], pred_latents], dim=2)
        return self._pack_latents(
            latents_unpacked,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
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
            image = image.to(
                device=device,
                dtype=prompt_context.positive_connector_prompt_embeds.dtype,
            )

        return self.prepare_latents(
            image,
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

    def _make_video_audio_scheduler(
        self,
        audio_scheduler: Any,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> Any:
        return I2VVideoAudioScheduler(
            self,
            audio_scheduler,
            latent_num_frames,
            latent_height,
            latent_width,
        )
