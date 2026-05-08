# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
import os
from collections.abc import Iterable
from typing import Any, ClassVar

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.dmd2 import DMD2PipelineMixin
from vllm_omni.diffusion.models.interface import SupportsModuleOffload
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.lora.request import LoRARequest

from .pipeline_ltx2 import (
    LTX2Pipeline,
    _get_prompt_field,
    calculate_shift,
)
from .pipeline_ltx2 import (
    get_ltx2_post_process_func as _get_ltx2_post_process_func,
)
from .pipeline_ltx2_latent_upsample import LTX2LatentUpsamplePipeline

logger = init_logger(__name__)


def get_ltx2_post_process_func(od_config: OmniDiffusionConfig):
    return _get_ltx2_post_process_func(od_config)


class _I2VVideoAudioScheduler:
    """Composite scheduler for I2V: uses _step_video_latents_i2v for video, standard step for audio."""

    def __init__(self, pipeline, audio_scheduler, latent_num_frames, latent_height, latent_width):
        self.video_scheduler = pipeline.scheduler
        self.audio_scheduler = audio_scheduler
        self._pipeline = pipeline
        self._latent_num_frames = latent_num_frames
        self._latent_height = latent_height
        self._latent_width = latent_width

    def step(self, noise_pred, t, latents, return_dict=False, generator=None):
        video_out = self._pipeline._step_video_latents_i2v(
            noise_pred[0],
            latents[0],
            t[0],
            self._latent_num_frames,
            self._latent_height,
            self._latent_width,
        )
        audio_out = self.audio_scheduler.step(noise_pred[1], t[1], latents[1], return_dict=False, generator=generator)[
            0
        ]
        return ((video_out, audio_out),)


class LTX2ImageToVideoPipeline(LTX2Pipeline):
    support_image_input = True

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__(od_config=od_config, prefix=prefix)
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio, resample="bilinear")

    @staticmethod
    def _normalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * scaling_factor / latents_std
        return latents

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
        height = height // self.vae_spatial_compression_ratio
        width = width // self.vae_spatial_compression_ratio
        num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1

        shape = (batch_size, num_channels_latents, num_frames, height, width)
        mask_shape = (batch_size, 1, num_frames, height, width)

        if latents is not None:
            if latents.ndim == 5:
                # conditioning_mask needs to the same shape as latents in two stages generation.
                batch_size, _, num_frames, height, width = latents.shape
                mask_shape = (batch_size, 1, num_frames, height, width)
                conditioning_mask = latents.new_zeros(mask_shape)
                conditioning_mask[:, :, 0] = 1.0

                latents = self._normalize_latents(
                    latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
                )
                latents = self._create_noised_state(latents, noise_scale * (1 - conditioning_mask), generator)
                # latents are of shape [B, C, F, H, W], need to be packed
                latents = self._pack_latents(
                    latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
                )
            else:
                conditioning_mask = latents.new_zeros(mask_shape)
                conditioning_mask[:, :, 0] = 1.0

            conditioning_mask = self._pack_latents(
                conditioning_mask, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
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
        init_latents = self._normalize_latents(init_latents, self.vae.latents_mean, self.vae.latents_std)
        init_latents = init_latents.repeat(1, 1, num_frames, 1, 1)

        conditioning_mask = torch.zeros(mask_shape, device=device, dtype=dtype)
        conditioning_mask[:, :, 0] = 1.0

        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = init_latents * conditioning_mask + noise * (1 - conditioning_mask)

        conditioning_mask = self._pack_latents(
            conditioning_mask, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
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

    def _step_video_latents_i2v(
        self,
        noise_pred_video: torch.Tensor,
        latents: torch.Tensor,
        t: torch.Tensor,
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
        pred_latents = self.scheduler.step(noise_pred_video, t, noise_latents, return_dict=False)[0]

        latents_unpacked = torch.cat([latents_unpacked[:, :, :1], pred_latents], dim=2)
        latents = self._pack_latents(
            latents_unpacked, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
        )
        return latents

    @torch.no_grad()
    def forward(
        self,
        req: OmniDiffusionRequest,
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
        # Extract prompt/negative_prompt from request.
        # Input format: req.prompts is a list of str or dict with "prompt"/"negative_prompt" keys.
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in req.prompts] or prompt
        if all(isinstance(p, str) or p.get("negative_prompt") is None for p in req.prompts):
            negative_prompt = None
        elif req.prompts:
            negative_prompt = ["" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in req.prompts]

        height = req.sampling_params.height or height or 512
        width = req.sampling_params.width or width or 768
        num_frames = req.sampling_params.num_frames or num_frames or 121
        frame_rate = req.sampling_params.resolved_frame_rate or frame_rate or 24.0
        num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps or 40
        if timesteps is None:
            num_inference_steps = max(int(num_inference_steps), 2)
        elif len(timesteps) < 2:
            raise ValueError("`timesteps` must contain at least 2 values for FlowMatchEulerDiscreteScheduler.")
        num_videos_per_prompt = (
            req.sampling_params.num_outputs_per_prompt
            if req.sampling_params.num_outputs_per_prompt > 0
            else num_videos_per_prompt or 1
        )
        max_sequence_length = (
            req.sampling_params.max_sequence_length or max_sequence_length or self.tokenizer_max_length
        )

        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale
        if req.sampling_params.guidance_rescale is not None:
            guidance_rescale = req.sampling_params.guidance_rescale

        if generator is None:
            generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(req.sampling_params.seed)

        latents = req.sampling_params.latents if req.sampling_params.latents is not None else latents
        audio_latents = (
            req.sampling_params.audio_latents
            if req.sampling_params.audio_latents is not None
            else req.sampling_params.extra_args.get("audio_latents", audio_latents)
        )

        # Override with pre-computed embeddings if provided in request.
        req_prompt_embeds = [_get_prompt_field(p, "prompt_embeds") for p in req.prompts]
        if any(p is not None for p in req_prompt_embeds):
            prompt_embeds = torch.stack(req_prompt_embeds)  # type: ignore[arg-type]

        req_negative_prompt_embeds = [_get_prompt_field(p, "negative_prompt_embeds") for p in req.prompts]
        if any(p is not None for p in req_negative_prompt_embeds):
            negative_prompt_embeds = torch.stack(req_negative_prompt_embeds)  # type: ignore[arg-type]

        req_prompt_attention_masks = [
            _get_prompt_field(p, "prompt_attention_mask") or _get_prompt_field(p, "attention_mask") for p in req.prompts
        ]
        if any(m is not None for m in req_prompt_attention_masks):
            prompt_attention_mask = torch.stack(req_prompt_attention_masks)  # type: ignore[arg-type]

        req_negative_attention_masks = [
            _get_prompt_field(p, "negative_prompt_attention_mask") or _get_prompt_field(p, "negative_attention_mask")
            for p in req.prompts
        ]
        if any(m is not None for m in req_negative_attention_masks):
            negative_prompt_attention_mask = torch.stack(req_negative_attention_masks)  # type: ignore[arg-type]

        if req.sampling_params.decode_timestep is not None:
            decode_timestep = req.sampling_params.decode_timestep
        if req.sampling_params.decode_noise_scale is not None:
            decode_noise_scale = req.sampling_params.decode_noise_scale
        if req.sampling_params.output_type is not None:
            output_type = req.sampling_params.output_type

        if image is None and req.prompts:
            raw_images = []
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
                raw_images.append(raw_image)

            if any(img is None for img in raw_images):
                if latents is None:
                    raise ValueError("Image is required for LTX2 I2V generation.")
            if len(raw_images) == 1:
                image = raw_images[0]
            elif raw_images:
                image = raw_images

        self.check_inputs(
            image=image,
            height=height,
            width=width,
            prompt=prompt,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        self._current_timestep = None

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.device

        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        # Compute positive prompt connectors
        additive_attention_mask = (1 - prompt_attention_mask.to(prompt_embeds.dtype)) * -1000000.0
        connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask = self.connectors(
            prompt_embeds, additive_attention_mask, additive_mask=True
        )

        # Compute negative prompt connectors when CFG is enabled
        negative_connector_prompt_embeds = None
        negative_connector_audio_prompt_embeds = None
        negative_connector_attention_mask = None
        if self.do_classifier_free_guidance:
            negative_additive_attention_mask = (
                1 - negative_prompt_attention_mask.to(negative_prompt_embeds.dtype)
            ) * -1000000.0
            (
                negative_connector_prompt_embeds,
                negative_connector_audio_prompt_embeds,
                negative_connector_attention_mask,
            ) = self.connectors(
                negative_prompt_embeds,
                negative_additive_attention_mask,
                additive_mask=True,
            )

        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        if latents is not None:
            if latents.ndim == 5:
                logger.info(
                    "Got latents of shape [batch_size, latent_dim, latent_frames, latent_height, latent_width], `latent_num_frames`, `latent_height`, `latent_width` will be inferred."  # noqa
                )
                _, _, latent_num_frames, latent_height, latent_width = latents.shape  # [B, C, F, H, W]
            elif latents.ndim == 3:
                logger.warning(
                    f"You have supplied packed `latents` of shape {latents.shape}, so the latent dims cannot be"
                    f" inferred. Make sure the supplied `height`, `width`, and `num_frames` are correct."
                )
            else:
                raise ValueError(
                    f"Provided `latents` tensor has shape {latents.shape}, but the expected shape is either [batch_size, seq_len, num_features] or [batch_size, latent_dim, latent_frames, latent_height, latent_width]."  # noqa
                )
        video_sequence_length = latent_num_frames * latent_height * latent_width

        if latents is None:
            if isinstance(image, torch.Tensor):
                if image.ndim == 3:
                    image = image.unsqueeze(0)
            elif isinstance(image, list) and image and isinstance(image[0], torch.Tensor):
                image = torch.stack(image, dim=0)
            else:
                image = self.video_processor.preprocess(image, height=height, width=width)
            image = image.to(device=device, dtype=prompt_embeds.dtype)

        num_channels_latents = self.transformer.config.in_channels
        latents, conditioning_mask = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            noise_scale,
            torch.float32,
            device,
            generator,
            latents,
        )

        duration_s = num_frames / frame_rate
        audio_latents_per_second = (
            self.audio_sampling_rate / self.audio_hop_length / float(self.audio_vae_temporal_compression_ratio)
        )
        audio_num_frames = round(duration_s * audio_latents_per_second)
        if audio_latents is not None:
            if audio_latents.ndim == 4:
                logger.info(
                    "Got audio_latents of shape [batch_size, num_channels, audio_length, mel_bins], `audio_num_frames` will be inferred."  # noqa
                )
                _, _, audio_num_frames, _ = audio_latents.shape  # [B, C, L, M]
            elif audio_latents.ndim == 3:
                logger.warning(
                    f"You have supplied packed `audio_latents` of shape {audio_latents.shape}, so the latent dims"
                    f" cannot be inferred. Make sure the supplied `num_frames` and `frame_rate` are correct."
                )
            else:
                raise ValueError(
                    f"Provided `audio_latents` tensor has shape {audio_latents.shape}, but the expected shape is either [batch_size, seq_len, num_features] or [batch_size, num_channels, audio_length, mel_bins]."  # noqa
                )

        num_mel_bins = self.audio_vae.config.mel_bins if getattr(self, "audio_vae", None) is not None else 64
        latent_mel_bins = num_mel_bins // self.audio_vae_mel_compression_ratio
        num_channels_latents_audio = (
            self.audio_vae.config.latent_channels if getattr(self, "audio_vae", None) is not None else 8
        )
        audio_latents, original_audio_num_frames, padded_audio_num_frames = self.prepare_audio_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents_audio,
            audio_latent_length=audio_num_frames,
            num_mel_bins=num_mel_bins,
            noise_scale=noise_scale,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=audio_latents,
        )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        mu = calculate_shift(
            video_sequence_length,
            self.scheduler.config.get("base_image_seq_len", 1024),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.95),
            self.scheduler.config.get("max_shift", 2.05),
        )
        audio_scheduler = copy.deepcopy(self.scheduler)
        _ = retrieve_timesteps(
            audio_scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas=sigmas,
            mu=mu,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas=sigmas,
            mu=mu,
        )
        self._num_timesteps = len(timesteps)

        video_coords = self.transformer.rope.prepare_video_coords(
            latents.shape[0], latent_num_frames, latent_height, latent_width, latents.device, fps=frame_rate
        )
        audio_coords = self.transformer.audio_rope.prepare_audio_coords(
            audio_latents.shape[0], padded_audio_num_frames, audio_latents.device
        )

        i2v_scheduler = _I2VVideoAudioScheduler(
            pipeline=self,
            audio_scheduler=audio_scheduler,
            latent_num_frames=latent_num_frames,
            latent_height=latent_height,
            latent_width=latent_width,
        )
        # No coord duplication needed: mixin handles CFG via separate forward calls.

        with self.progress_bar(total=len(timesteps)) as pbar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                latent_model_input = latents.to(prompt_embeds.dtype)
                audio_latent_model_input = audio_latents.to(prompt_embeds.dtype)
                timestep = t.expand(latent_model_input.shape[0])
                video_timestep = timestep.unsqueeze(-1) * (1 - conditioning_mask)
                do_true_cfg = self.do_classifier_free_guidance

                positive_kwargs = {
                    "hidden_states": latent_model_input,
                    "audio_hidden_states": audio_latent_model_input,
                    "encoder_hidden_states": connector_prompt_embeds,
                    "audio_encoder_hidden_states": connector_audio_prompt_embeds,
                    "timestep": video_timestep,
                    "audio_timestep": timestep,
                    "encoder_attention_mask": connector_attention_mask,
                    "audio_encoder_attention_mask": connector_attention_mask,
                    "num_frames": latent_num_frames,
                    "height": latent_height,
                    "width": latent_width,
                    "fps": frame_rate,
                    "audio_num_frames": padded_audio_num_frames,
                    "video_coords": video_coords,
                    "audio_coords": audio_coords,
                    "attention_kwargs": attention_kwargs,
                    "return_dict": False,
                }
                negative_kwargs = (
                    {
                        **positive_kwargs,
                        "encoder_hidden_states": negative_connector_prompt_embeds,
                        "audio_encoder_hidden_states": negative_connector_audio_prompt_embeds,
                        "encoder_attention_mask": negative_connector_attention_mask,
                        "audio_encoder_attention_mask": negative_connector_attention_mask,
                    }
                    if do_true_cfg
                    else None
                )

                noise_pred_video, noise_pred_audio = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=False,
                )

                latents, audio_latents = self.scheduler_step_maybe_with_cfg(
                    (noise_pred_video, noise_pred_audio),
                    (t, t),
                    (latents, audio_latents),
                    do_true_cfg=do_true_cfg,
                    per_request_scheduler=i2v_scheduler,
                )
                latents, audio_latents = self._synchronize_cfg_parallel_step_output(
                    (latents, audio_latents),
                    do_true_cfg=do_true_cfg,
                )

                pbar.update()

        latents = self._unpack_latents(
            latents,
            latent_num_frames,
            latent_height,
            latent_width,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )
        latents = self._denormalize_latents(
            latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
        )

        audio_latents = self._unpad_audio_latents(audio_latents, original_audio_num_frames)
        audio_latents = self._denormalize_audio_latents(
            audio_latents, self.audio_vae.latents_mean, self.audio_vae.latents_std
        )
        audio_latents = self._unpack_audio_latents(
            audio_latents,
            original_audio_num_frames,
            num_mel_bins=latent_mel_bins,
        )

        if output_type == "latent":
            video = latents
            audio = audio_latents
        else:
            latents = latents.to(prompt_embeds.dtype)

            if not self.vae.config.timestep_conditioning:
                timestep = None
            else:
                noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
                if not isinstance(decode_timestep, list):
                    decode_timestep = [decode_timestep] * batch_size
                if decode_noise_scale is None:
                    decode_noise_scale = decode_timestep
                elif not isinstance(decode_noise_scale, list):
                    decode_noise_scale = [decode_noise_scale] * batch_size

                timestep = torch.tensor(decode_timestep, device=device, dtype=latents.dtype)
                decode_noise_scale = torch.tensor(decode_noise_scale, device=device, dtype=latents.dtype)[
                    :, None, None, None, None
                ]
                latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

            latents = latents.to(self.vae.dtype)
            video = self.vae.decode(latents, timestep, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)

            audio_latents = audio_latents.to(self.audio_vae.dtype)
            generated_mel_spectrograms = self.audio_vae.decode(audio_latents, return_dict=False)[0]
            audio = self.vocoder(generated_mel_spectrograms)

        if not return_dict:
            return DiffusionOutput(output=(video, audio))

        return DiffusionOutput(output=(video, audio))


class LTX2ImageToVideoTwoStagesPipeline(nn.Module, SupportsModuleOffload):
    """LTXImageToVideoTwoStagesPipeline is for two stages image to video generation"""

    support_image_input = True

    _dit_modules: ClassVar[list[str]] = ["pipe.transformer"]
    _encoder_modules: ClassVar[list[str]] = ["pipe.text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["pipe.vae", "pipe.audio_vae"]

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()

        self.device = get_local_device()
        self.dtype = getattr(od_config, "dtype", torch.bfloat16)
        self.model_path = od_config.model
        self.distilled = False
        # User provided model path may contain '/' in the end and basename function
        # will not return the expected directory name, so we need to remove it by normpath
        if "distilled" in os.path.basename(os.path.normpath(self.model_path)):
            self.distilled = True
        else:
            raise NotImplementedError(f"{self.model_path} is not supported for {self.__class__.__name__}.")

        self.pipe = LTX2ImageToVideoPipeline(od_config=od_config, prefix=prefix)
        self.upsample_pipe = LTX2LatentUpsamplePipeline(
            vae=self.pipe.vae,
            od_config=od_config,
        )

        self.lora_manager = DiffusionLoRAManager(
            pipeline=self.pipe,
            device=self.device,
            dtype=self.dtype,
            max_cached_adapters=od_config.max_cpu_loras,
        )

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="pipe.transformer.",
                fall_back_to_pt=True,
            ),
        ]

    @torch.no_grad()
    def forward(
        self,
        req: OmniDiffusionRequest,
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
    ):
        video_latent, audio_latent = self.pipe(
            req=req,
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            sigmas=DISTILLED_SIGMA_VALUES if self.distilled else None,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
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
            output_type="latent",
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
            max_sequence_length=max_sequence_length,
        ).output

        upscaled_video_latent = self.upsample_pipe(
            latents=video_latent,
            output_type="latent",
            return_dict=False,
        )[0]

        if not self.distilled:
            # Load Stage 2 distilled LoRA
            lora_path = f"{self.model_path}/ltx-2-19b-distilled-lora-384.safetensors"
            lora_request = LoRARequest(
                lora_name="stage_2_distilled",
                lora_int_id=1,
                lora_path=lora_path,
            )
            self.lora_manager.set_active_adapter(lora_request, lora_scale=1.0)

            # Change scheduler to use Stage 2 distilled sigmas as is
            new_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config,
                use_dynamic_shifting=False,
                shift_terminal=None,
            )
            self.pipe.scheduler = new_scheduler

        stage_2_req = copy.copy(req)
        stage_2_req.sampling_params = req.sampling_params.clone()
        stage_2_req.sampling_params.num_inference_steps = 3

        video, audio = self.pipe(
            req=stage_2_req,
            latents=upscaled_video_latent,
            audio_latents=audio_latent,
            prompt=prompt,
            negative_prompt=negative_prompt,
            noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
            sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
            guidance_scale=1.0,
            generator=generator,
            output_type="np",
            return_dict=False,
        ).output

        return DiffusionOutput(output=(video, audio))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


class LTX2I2VDMD2Pipeline(DMD2PipelineMixin, LTX2ImageToVideoPipeline):
    """LTX-2 I2V pipeline for FastGen DMD2-distilled models."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.__init_dmd2__()
