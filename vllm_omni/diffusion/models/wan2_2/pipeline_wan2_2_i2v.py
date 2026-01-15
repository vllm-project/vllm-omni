# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from collections.abc import Iterable

import numpy as np
import PIL.Image
import torch
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportImageInput
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    create_transformer_from_config,
    load_transformer_config,
    retrieve_latents,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest


def _load_model_index(model: str, local_files_only: bool) -> dict:
    """Load model_index.json from local path or HF Hub."""
    if local_files_only:
        model_index_path = os.path.join(model, "model_index.json")
        if os.path.exists(model_index_path):
            import json

            with open(model_index_path) as f:
                return json.load(f)
    else:
        try:
            import json

            from huggingface_hub import hf_hub_download

            model_index_path = hf_hub_download(model, "model_index.json")
            with open(model_index_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def get_wan22_i2v_post_process_func(
    od_config: OmniDiffusionConfig,
):
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(
        video: torch.Tensor,
        output_type: str = "np",
    ):
        if output_type == "latent":
            return video
        return video_processor.postprocess_video(video, output_type=output_type)

    return post_process_func


def get_wan22_i2v_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    """Pre-process function for I2V: load and resize input image."""
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def pre_process_func(requests: list[OmniDiffusionRequest]) -> list[OmniDiffusionRequest]:
        for req in requests:
            # Load image if path is provided
            if req.image_path is not None and req.pil_image is None:
                req.pil_image = PIL.Image.open(req.image_path).convert("RGB")

            if req.pil_image is not None:
                image = req.pil_image

                # Calculate dimensions based on aspect ratio if not provided
                if req.height is None or req.width is None:
                    # Default max area for 480P
                    max_area = 480 * 832
                    aspect_ratio = image.height / image.width

                    # Calculate dimensions maintaining aspect ratio
                    mod_value = 16  # Must be divisible by 16
                    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
                    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value

                    if req.height is None:
                        req.height = height
                    if req.width is None:
                        req.width = width

                # Resize image to target dimensions
                image = image.resize((req.width, req.height), PIL.Image.Resampling.LANCZOS)
                req.pil_image = image

                # Preprocess for VAE
                req.preprocessed_image = video_processor.preprocess(image, height=req.height, width=req.width)

        return requests

    return pre_process_func


class Wan22I2VPipeline(nn.Module, SupportImageInput):
    """
    Wan2.2 Image-to-Video Pipeline.

    Supports both Wan2.1-style I2V (with CLIP image embeddings) and
    Wan2.2-style I2V (with expand_timesteps for TI2V-5B).
    """

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config

        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        model = od_config.model
        local_files_only = os.path.exists(model)

        # Set up weights sources for transformer(s)
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
        ]

        # Load model_index.json to detect available components
        model_index = _load_model_index(model, local_files_only)

        # Check if this is a two-stage model (MoE with transformer_2)
        self.has_transformer_2 = "transformer_2" in model_index

        if self.has_transformer_2:
            self.weights_sources.append(
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=od_config.model,
                    subfolder="transformer_2",
                    revision=None,
                    prefix="transformer_2.",
                    fall_back_to_pt=True,
                )
            )

        # Text encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model, subfolder="text_encoder", torch_dtype=dtype, local_files_only=local_files_only
        ).to(self.device)

        # Image encoder (CLIP) - optional, for Wan2.1-style I2V
        self.has_image_encoder = "image_encoder" in model_index and model_index["image_encoder"][0] is not None

        if self.has_image_encoder:
            self.image_processor = CLIPImageProcessor.from_pretrained(
                model, subfolder="image_processor", local_files_only=local_files_only
            )
            self.image_encoder = CLIPVisionModel.from_pretrained(
                model, subfolder="image_encoder", torch_dtype=dtype, local_files_only=local_files_only
            ).to(self.device)
        else:
            self.image_processor = None
            self.image_encoder = None

        # VAE
        self.vae = AutoencoderKLWan.from_pretrained(
            model, subfolder="vae", torch_dtype=torch.float32, local_files_only=local_files_only
        ).to(self.device)

        # Transformers (weights loaded via load_weights)
        # Load config from model directory or HF Hub to get correct in_channels for I2V models
        transformer_config = load_transformer_config(model, "transformer", local_files_only)
        self.transformer = create_transformer_from_config(transformer_config)
        if self.has_transformer_2:
            transformer_2_config = load_transformer_config(model, "transformer_2", local_files_only)
            self.transformer_2 = create_transformer_from_config(transformer_2_config)
        else:
            self.transformer_2 = None

        # Scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )
        if od_config.flow_shift is not None:
            self.scheduler.config.flow_shift = od_config.flow_shift

        # VAE scale factors
        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if hasattr(self.vae, "config") else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if hasattr(self.vae, "config") else 8

        # MoE boundary ratio for two-stage denoising
        self.boundary_ratio = od_config.boundary_ratio

        # Whether to use expand_timesteps mode (for TI2V-5B style)
        self.expand_timesteps = getattr(od_config, "expand_timesteps", False)

        self._guidance_scale = None
        self._guidance_scale_2 = None
        self._num_timesteps = None
        self._current_timestep = None

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    def encode_image(
        self,
        image: PIL.Image.Image | list[PIL.Image.Image],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Encode image using CLIP image encoder."""
        device = device or self.device
        if self.image_encoder is None:
            raise ValueError("Image encoder not available for this model.")

        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device=device, dtype=self.image_encoder.dtype)
        image_embeds = self.image_encoder(pixel_values, output_hidden_states=True)
        return image_embeds.hidden_states[-2]

    @torch.no_grad()
    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        image: PIL.Image.Image | torch.Tensor | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | tuple[float, float] = 5.0,
        frame_num: int | None = None,
        output_type: str | None = "np",
        generator: torch.Generator | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        image_embeds: torch.Tensor | None = None,
        last_image: PIL.Image.Image | torch.Tensor | None = None,
        attention_kwargs: dict | None = None,
        **kwargs,
    ) -> DiffusionOutput:
        # Get parameters from request or arguments
        prompt = req.prompt if req.prompt is not None else prompt
        negative_prompt = req.negative_prompt if req.negative_prompt is not None else negative_prompt
        if prompt is None and prompt_embeds is None:
            raise ValueError("Prompt or prompt_embeds is required for Wan2.2 I2V generation.")

        # Get image from request
        if image is None:
            image = req.pil_image
        if image is None:
            raise ValueError("Image is required for I2V generation.")

        height = req.height or height or 480
        width = req.width or width or 832
        num_frames = req.num_frames if req.num_frames else frame_num or 81
        num_steps = req.num_inference_steps or num_inference_steps or 50

        # Handle guidance scales
        guidance_low = guidance_scale if isinstance(guidance_scale, (int, float)) else guidance_scale[0]
        guidance_high = (
            req.guidance_scale_2
            if req.guidance_scale_2 is not None
            else (
                guidance_scale[1]
                if isinstance(guidance_scale, (list, tuple)) and len(guidance_scale) > 1
                else guidance_low
            )
        )

        self._guidance_scale = guidance_low
        self._guidance_scale_2 = guidance_high

        # Validate inputs
        self.check_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image_embeds=image_embeds,
            guidance_scale_2=guidance_high if self.boundary_ratio is not None else None,
        )

        # Adjust num_frames to be compatible with VAE temporal scaling
        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        device = self.device
        dtype = self.transformer.dtype

        # Generator setup
        if generator is None:
            generator = req.generator
        if generator is None and req.seed is not None:
            generator = torch.Generator(device=device).manual_seed(req.seed)

        # Encode prompts
        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=guidance_low > 1.0 or guidance_high > 1.0,
                num_videos_per_prompt=req.num_outputs_per_prompt or 1,
                max_sequence_length=req.max_sequence_length or 512,
                device=device,
                dtype=dtype,
            )
        else:
            prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)

        batch_size = prompt_embeds.shape[0]

        # Encode image embeddings (for Wan2.1-style with CLIP)
        if self.has_image_encoder and self.transformer.config.image_dim is not None:
            if image_embeds is None:
                if last_image is None:
                    image_embeds = self.encode_image(image, device)
                else:
                    image_embeds = self.encode_image([image, last_image], device)
            image_embeds = image_embeds.repeat(batch_size, 1, 1)
            image_embeds = image_embeds.to(dtype)
        else:
            image_embeds = None

        # Timesteps
        self.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        boundary_timestep = None
        if self.boundary_ratio is not None:
            boundary_timestep = self.boundary_ratio * self.scheduler.config.num_train_timesteps

        # Prepare latents (use out_channels=16 for VAE latent, not in_channels=36)
        num_channels_latents = self.transformer.config.out_channels

        # Preprocess image for VAE
        from diffusers.video_processor import VideoProcessor

        video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        if isinstance(image, PIL.Image.Image):
            image_tensor = video_processor.preprocess(image, height=height, width=width)
        else:
            image_tensor = image
        image_tensor = image_tensor.to(device=device, dtype=torch.float32)

        # Handle last_image if provided
        if last_image is not None:
            if isinstance(last_image, PIL.Image.Image):
                last_image_tensor = video_processor.preprocess(last_image, height=height, width=width)
            else:
                last_image_tensor = last_image
            last_image_tensor = last_image_tensor.to(device=device, dtype=torch.float32)
        else:
            last_image_tensor = None

        latents, condition, first_frame_mask = self.prepare_latents(
            image=image_tensor,
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=req.latents,
            last_image=last_image_tensor,
        )

        if attention_kwargs is None:
            attention_kwargs = {}

        # Denoising loop
        for t in timesteps:
            self._current_timestep = t

            # Select model and guidance scale based on timestep
            current_model = self.transformer
            current_guidance_scale = guidance_low
            if boundary_timestep is not None and t < boundary_timestep and self.transformer_2 is not None:
                current_model = self.transformer_2
                current_guidance_scale = guidance_high

            # Prepare latent input
            if self.expand_timesteps:
                # TI2V-5B style: blend condition with latents using mask
                latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents
                latent_model_input = latent_model_input.to(dtype)

                # Expand timesteps for each patch
                temp_ts = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten()
                timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
            else:
                # Wan2.1 style: concatenate condition with latents
                latent_model_input = torch.cat([latents, condition], dim=1).to(dtype)
                timestep = t.expand(latents.shape[0])

            # Forward pass
            noise_pred = current_model(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_image=image_embeds,
                attention_kwargs=attention_kwargs,
                return_dict=False,
            )[0]

            # Classifier-free guidance
            if current_guidance_scale > 1.0 and negative_prompt_embeds is not None:
                noise_uncond = current_model(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        self._current_timestep = None

        # For expand_timesteps mode, blend final latents with condition
        if self.expand_timesteps:
            latents = (1 - first_frame_mask) * condition + first_frame_mask * latents

        # Decode
        if output_type == "latent":
            output = latents
        else:
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            output = self.vae.decode(latents, return_dict=False)[0]

        return DiffusionOutput(output=output)

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Encode text prompts using T5 text encoder."""
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_clean = [self._prompt_clean(p) for p in prompt]
        batch_size = len(prompt_clean)

        text_inputs = self.tokenizer(
            prompt_clean,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            neg_text_inputs = self.tokenizer(
                [self._prompt_clean(p) for p in negative_prompt],
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            ids_neg, mask_neg = neg_text_inputs.input_ids, neg_text_inputs.attention_mask
            seq_lens_neg = mask_neg.gt(0).sum(dim=1).long()
            negative_prompt_embeds = self.text_encoder(ids_neg.to(device), mask_neg.to(device)).last_hidden_state
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)
            negative_prompt_embeds = [u[:v] for u, v in zip(negative_prompt_embeds, seq_lens_neg)]
            negative_prompt_embeds = torch.stack(
                [
                    torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                    for u in negative_prompt_embeds
                ],
                dim=0,
            )
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    @staticmethod
    def _prompt_clean(text: str) -> str:
        return " ".join(text.strip().split())

    def prepare_latents(
        self,
        image: torch.Tensor,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype | None,
        device: torch.device | None,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None = None,
        last_image: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare latents for I2V generation.

        Returns:
            latents: Initial noise latents
            condition: Encoded image condition (concatenated with mask for non-expand mode)
            first_frame_mask: Mask for the first frame (1 for frames to denoise, 0 for condition)
        """
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        # Prepare image condition
        image = image.unsqueeze(2)  # [batch, channels, 1, height, width]

        if self.expand_timesteps:
            # TI2V-5B style: only use first frame as condition
            video_condition = image
        elif last_image is None:
            # Pad with zeros for remaining frames
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2
            )
        else:
            # First and last frame conditioning
            last_image = last_image.unsqueeze(2)
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 2, height, width), last_image],
                dim=2,
            )

        video_condition = video_condition.to(device=device, dtype=self.vae.dtype)

        # Encode through VAE
        latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
        latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

        # Normalize latents
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latent_condition.device, latent_condition.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latent_condition.device, latent_condition.dtype
        )
        latent_condition = (latent_condition - latents_mean) * latents_std
        latent_condition = latent_condition.to(dtype)

        if self.expand_timesteps:
            # TI2V-5B style: create mask where first frame is 0 (condition), rest is 1 (to denoise)
            first_frame_mask = torch.ones(
                1, 1, num_latent_frames, latent_height, latent_width, dtype=dtype, device=device
            )
            first_frame_mask[:, :, 0] = 0
            return latents, latent_condition, first_frame_mask

        # Wan2.1 style: create mask and concatenate with condition
        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width)

        if last_image is None:
            mask_lat_size[:, :, list(range(1, num_frames))] = 0
        else:
            mask_lat_size[:, :, list(range(1, num_frames - 1))] = 0

        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = torch.repeat_interleave(first_frame_mask, dim=2, repeats=self.vae_scale_factor_temporal)
        mask_lat_size = torch.concat([first_frame_mask, mask_lat_size[:, :, 1:, :]], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)

        # Concatenate mask with condition for channel dimension
        condition = torch.concat([mask_lat_size, latent_condition], dim=1)

        # For non-expand mode, first_frame_mask is not used in the same way
        first_frame_mask = torch.ones(1, 1, num_latent_frames, latent_height, latent_width, dtype=dtype, device=device)

        return latents, condition, first_frame_mask

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        image,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        image_embeds=None,
        guidance_scale_2=None,
    ):
        if image is None and image_embeds is None:
            raise ValueError("Provide either `image` or `image_embeds`. Cannot leave both undefined.")

        if image is not None and image_embeds is not None:
            raise ValueError("Cannot forward both `image` and `image_embeds`. Please provide only one.")

        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt` and `prompt_embeds`. Please provide only one.")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                "Cannot forward both `negative_prompt` and `negative_prompt_embeds`. Please provide only one."
            )

        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")

        if self.boundary_ratio is None and guidance_scale_2 is not None:
            raise ValueError("`guidance_scale_2` is only supported when `boundary_ratio` is set.")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights using AutoWeightsLoader for vLLM integration."""
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
