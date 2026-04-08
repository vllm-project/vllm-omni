# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
AniSora I2V Pipeline using CogVideoX architecture.

AniSora V1 (5B) is based on CogVideoX. The transformer uses vLLM-Omni's
parallelized layers (QKVParallelLinear, etc.) for tensor parallelism support.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable

import numpy as np
import PIL.Image
import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDDIMScheduler,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.models.transformers.cogvideox_transformer_3d import (
    CogVideoXTransformer3DModel as DiffusersCogVideoXTransformer,
)
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoTokenizer, T5EncoderModel
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.anisora.cogvideox_transformer import (
    CogVideoXTransformer3DModel,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniTextPrompt

logger = logging.getLogger(__name__)


def get_anisora_i2v_post_process_func(
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


def get_anisora_i2v_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    """Pre-process function for I2V: load and resize input image."""
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        for i, prompt in enumerate(request.prompts):
            multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else None
            raw_image = multi_modal_data.get("image", None) if multi_modal_data is not None else None
            if isinstance(prompt, str):
                prompt = OmniTextPrompt(prompt=prompt)
            if "additional_information" not in prompt:
                prompt["additional_information"] = {}

            if raw_image is None:
                raise ValueError(
                    "No image is provided. This model requires an image to run. "
                    'Please correctly set `"multi_modal_data": {"image": <an image object or file path>, …}`'
                )
            if not isinstance(raw_image, (str, PIL.Image.Image)):
                raise TypeError(
                    f"""Unsupported image format {raw_image.__class__}.""",
                    """Please correctly set `"multi_modal_data": {"image": <an image object or file path>, …}`""",
                )
            image = PIL.Image.open(raw_image).convert("RGB") if isinstance(raw_image, str) else raw_image

            # Calculate dimensions based on aspect ratio if not provided
            if request.sampling_params.height is None or request.sampling_params.width is None:
                # Default max area for 480P
                max_area = 480 * 832
                aspect_ratio = image.height / image.width

                # Calculate dimensions maintaining aspect ratio
                mod_value = 16  # Must be divisible by 16
                height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
                width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value

                if request.sampling_params.height is None:
                    request.sampling_params.height = height
                if request.sampling_params.width is None:
                    request.sampling_params.width = width

            # Resize image to target dimensions
            image = image.resize(
                (request.sampling_params.width, request.sampling_params.height),  # type: ignore # height/width set above
                PIL.Image.Resampling.LANCZOS,
            )
            prompt["multi_modal_data"]["image"] = image  # type: ignore # key existence checked above

            # Preprocess for VAE
            prompt["additional_information"]["preprocessed_image"] = video_processor.preprocess(
                image,
                height=request.sampling_params.height,
                width=request.sampling_params.width,
            )
            request.prompts[i] = prompt
        return request

    return pre_process_func


def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


class AniSoraI2VCogVideoXPipeline(nn.Module):
    """
    AniSora Image-to-Video Pipeline using CogVideoX architecture.

    The transformer uses vLLM-Omni's optimized layers (QKVParallelLinear, etc.)
    for tensor parallelism support. Compatible with the
    Disty0/Index-anisora-5B-diffusers model.
    """

    # vLLM uses this flag to decide whether to feed dummy images in warmup
    support_image_input = True

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
    ):
        super().__init__()
        self.device = get_local_device()
        self.dtype = od_config.dtype
        model_path = od_config.model

        local_files_only = os.path.exists(model_path) if isinstance(model_path, str) else False

        logger.info("Loading tokenizer from %s...", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            subfolder="tokenizer",
            local_files_only=local_files_only,
        )

        logger.info("Loading text encoder from %s...", model_path)
        self.text_encoder = T5EncoderModel.from_pretrained(
            model_path,
            subfolder="text_encoder",
            torch_dtype=self.dtype,
            local_files_only=local_files_only,
        )

        logger.info("Loading VAE from %s...", model_path)
        self.vae = AutoencoderKLCogVideoX.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=torch.float32,  # VAE in float32 for precision
            local_files_only=local_files_only,
        )

        # Load transformer config from pretrained, then build optimized model
        logger.info("Loading transformer config from %s...", model_path)
        tf_config = DiffusersCogVideoXTransformer.load_config(
            model_path, subfolder="transformer", local_files_only=local_files_only
        )
        self.transformer = CogVideoXTransformer3DModel(
            od_config=od_config,
            num_attention_heads=tf_config.get("num_attention_heads", 30),
            attention_head_dim=tf_config.get("attention_head_dim", 64),
            in_channels=tf_config.get("in_channels", 16),
            out_channels=tf_config.get("out_channels", 16),
            flip_sin_to_cos=tf_config.get("flip_sin_to_cos", True),
            freq_shift=tf_config.get("freq_shift", 0),
            time_embed_dim=tf_config.get("time_embed_dim", 512),
            ofs_embed_dim=tf_config.get("ofs_embed_dim", None),
            text_embed_dim=tf_config.get("text_embed_dim", 4096),
            num_layers=tf_config.get("num_layers", 30),
            dropout=tf_config.get("dropout", 0.0),
            attention_bias=tf_config.get("attention_bias", True),
            sample_width=tf_config.get("sample_width", 90),
            sample_height=tf_config.get("sample_height", 60),
            sample_frames=tf_config.get("sample_frames", 49),
            patch_size=tf_config.get("patch_size", 2),
            patch_size_t=tf_config.get("patch_size_t", None),
            temporal_compression_ratio=tf_config.get("temporal_compression_ratio", 4),
            max_text_seq_length=tf_config.get("max_text_seq_length", 226),
            activation_fn=tf_config.get("activation_fn", "gelu-approximate"),
            timestep_activation_fn=tf_config.get("timestep_activation_fn", "silu"),
            norm_elementwise_affine=tf_config.get("norm_elementwise_affine", True),
            norm_eps=tf_config.get("norm_eps", 1e-5),
            spatial_interpolation_scale=tf_config.get("spatial_interpolation_scale", 1.875),
            temporal_interpolation_scale=tf_config.get("temporal_interpolation_scale", 1.0),
            use_rotary_positional_embeddings=tf_config.get("use_rotary_positional_embeddings", False),
            use_learned_positional_embeddings=tf_config.get("use_learned_positional_embeddings", False),
            patch_bias=tf_config.get("patch_bias", True),
        )

        # Tell the framework to load transformer weights from checkpoint
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_path,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
        ]

        logger.info("Loading scheduler from %s...", model_path)
        self.scheduler = CogVideoXDDIMScheduler.from_pretrained(
            model_path,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )

        # Scale factors from VAE config
        self.vae_scale_factor_spatial = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.vae_scale_factor_temporal = self.vae.config.temporal_compression_ratio

        self._current_timestep = None
        logger.info("Pipeline loaded successfully!")

    def to(self, device):
        """Move pipeline to device."""
        self.device = device
        self.text_encoder = self.text_encoder.to(device)
        self.vae = self.vae.to(device)
        self.transformer = self.transformer.to(device)
        return self

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        max_sequence_length: int = 226,
    ):
        """Encode text prompts using T5."""
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)

        prompt_embeds = self.text_encoder(text_input_ids)[0]
        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)

        # Negative prompt
        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size

            uncond_input = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_tensors="pt",
            )
            negative_prompt_embeds = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.dtype, device=self.device)
        else:
            negative_prompt_embeds = None

        return prompt_embeds, negative_prompt_embeds

    @torch.no_grad()
    def encode_image(self, image: PIL.Image.Image | torch.Tensor, height: int, width: int):
        """Encode input image to latent space."""
        from diffusers.video_processor import VideoProcessor

        video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        # Preprocess if PIL Image
        if isinstance(image, PIL.Image.Image):
            image_tensor = video_processor.preprocess(image, height=height, width=width)
        else:
            image_tensor = image

        # Move to device and ensure correct dtype (VAE expects float32 for encoding)
        image_tensor = image_tensor.to(device=self.device, dtype=torch.float32)

        # Add frame dimension: [B, C, H, W] -> [B, C, F, H, W] with F=1
        # CogVideoX VAE expects [B, C, F, H, W]
        image_tensor = image_tensor.unsqueeze(2)  # [B, C, 1, H, W]

        # Encode
        latent = self.vae.encode(image_tensor).latent_dist.sample()

        # Scale by latent std if available
        if hasattr(self.vae.config, "scaling_factor"):
            latent = latent * self.vae.config.scaling_factor

        return latent  # [B, C, 1, H//8, W//8]

    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
    ):
        """Prepare 3D rotary positional embeddings."""
        grid_height = height // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        grid_width = width // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)

        p = self.transformer.config.patch_size
        p_t = self.transformer.config.patch_size_t

        base_size_width = self.transformer.config.sample_width // p
        base_size_height = self.transformer.config.sample_height // p

        if p_t is None:
            # CogVideoX 1.0 style
            grid_crops_coords = get_resize_crop_region_for_grid(
                (grid_height, grid_width), base_size_width, base_size_height
            )
            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.transformer.config.attention_head_dim,
                crops_coords=grid_crops_coords,
                grid_size=(grid_height, grid_width),
                temporal_size=num_frames,
                device=self.device,
            )
        else:
            # CogVideoX 1.5 style
            base_num_frames = (num_frames + p_t - 1) // p_t
            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.transformer.config.attention_head_dim,
                crops_coords=None,
                grid_size=(grid_height, grid_width),
                temporal_size=base_num_frames,
                grid_type="slice",
                max_size=(base_size_height, base_size_width),
                device=self.device,
            )

        return freqs_cos, freqs_sin

    @torch.no_grad()
    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] | None = None,
        image: PIL.Image.Image | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        generator: torch.Generator | None = None,
        output_type: str | None = None,
        **kwargs,
    ) -> DiffusionOutput:
        """
        Forward pass for vLLM framework integration.
        Extracts parameters from OmniDiffusionRequest and generates video.
        """
        # Ensure model weights are on the correct device before any compute
        self.to(self.device)
        if len(req.prompts) > 1:
            raise ValueError(
                "This model only supports a single prompt, not a batched request. "
                "Please pass in a single prompt object or string, or a single-item list."
            )

        # Extract text prompts from request if not explicitly provided
        if prompt is None:
            first_prompt = None
            if getattr(req, "prompts", None):
                first_prompt = req.prompts[0]
            if isinstance(first_prompt, str):
                prompt = first_prompt
            elif isinstance(first_prompt, dict):
                prompt = first_prompt.get("text") or first_prompt.get("prompt") or first_prompt.get("caption")

        if negative_prompt is None:
            neg = None
            first_prompt = None
            if getattr(req, "prompts", None):
                first_prompt = req.prompts[0]
            if isinstance(first_prompt, dict):
                neg = first_prompt.get("negative_text") or first_prompt.get("negative_prompt")
            negative_prompt = neg

        # Use preprocessed_image if available, otherwise use image from multi_modal_data
        if image is None:
            first_prompt = None
            if getattr(req, "prompts", None):
                first_prompt = req.prompts[0]
            if isinstance(first_prompt, dict):
                additional_info = first_prompt.get("additional_information", {})
                if isinstance(additional_info, dict) and "preprocessed_image" in additional_info:
                    image = additional_info["preprocessed_image"]
                elif "multi_modal_data" in first_prompt and isinstance(first_prompt["multi_modal_data"], dict):
                    image = first_prompt["multi_modal_data"].get("image")

        # Derive sampling parameters from explicit args, then from req.sampling_params, then defaults
        sampling_params = getattr(req, "sampling_params", None)

        def _get_sp_attr(name, *aliases, default=None):
            if sampling_params is None:
                return default
            for key in (name, *aliases):
                value = getattr(sampling_params, key, None)
                if value is not None:
                    return value
            return default

        if height is None:
            height = _get_sp_attr("height", default=480)
        if width is None:
            width = _get_sp_attr("width", default=832)
        if num_frames is None:
            num_frames = _get_sp_attr("num_frames", "frames", default=17)
        if num_inference_steps is None:
            num_inference_steps = _get_sp_attr("num_inference_steps", "steps", default=50)

        if getattr(sampling_params, "guidance_scale_provided", False):
            guidance_scale = sampling_params.guidance_scale
        elif guidance_scale is None:
            guidance_scale = _get_sp_attr("guidance_scale", "cfg_scale", default=6.0)
        output_type = output_type or "tensor"

        if prompt is None:
            raise ValueError("Prompt is required")
        if image is None:
            raise ValueError("Image is required for I2V generation")
        if isinstance(image, str):
            image = PIL.Image.open(image).convert("RGB")

        return self._generate(
            prompt=prompt,
            image=image,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type=output_type,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str],
        image: PIL.Image.Image,
        negative_prompt: str | list[str] | None = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 17,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        generator: torch.Generator | None = None,
        output_type: str = "tensor",
    ):
        """
        Direct call interface for standalone usage.

        Args:
            prompt: Text prompt(s)
            image: Input image
            negative_prompt: Negative prompt(s)
            height: Output height
            width: Output width
            num_frames: Number of output frames
            num_inference_steps: Denoising steps
            guidance_scale: Classifier-free guidance scale
            generator: Random generator for reproducibility
            output_type: "tensor", "np", or "pil"
        """
        return self._generate(
            prompt=prompt,
            image=image,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            output_type=output_type,
        )

    def _generate(
        self,
        prompt: str | list[str],
        image: PIL.Image.Image,
        negative_prompt: str | list[str] | None = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 17,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        generator: torch.Generator | None = None,
        output_type: str = "tensor",
    ) -> DiffusionOutput:
        # Default to empty negative prompt so CFG is not silently skipped
        if negative_prompt is None and guidance_scale > 1.0:
            negative_prompt = ""

        # Encode prompt
        logger.info("Encoding prompts...")
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt, negative_prompt)

        do_classifier_free_guidance = guidance_scale > 1.0 and negative_prompt_embeds is not None

        # Encode image
        logger.info("Encoding image...")
        image_latents = self.encode_image(image, height, width)

        # Prepare latent dimensions
        batch_size = prompt_embeds.shape[0]
        num_channels_latents = self.transformer.config.in_channels // 2  # 16 for noise, 16 for image
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        latent_num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # CogVideoX uses [B, F, C, H, W] format
        latent_shape = (
            batch_size,
            latent_num_frames,
            num_channels_latents,
            latent_height,
            latent_width,
        )

        # Initial noise
        logger.info("Preparing latents...")
        latents = randn_tensor(latent_shape, generator=generator, device=self.device, dtype=self.dtype)

        # Prepare image latents for conditioning
        # image_latents: [B, C, 1, H, W] -> [B, 1, C, H, W]
        image_latents = image_latents.permute(0, 2, 1, 3, 4).to(dtype=self.dtype)

        # Pad image latents to match num_frames: first frame is image, rest are zeros
        padding_shape = (
            batch_size,
            latent_num_frames - 1,
            num_channels_latents,
            latent_height,
            latent_width,
        )
        latent_padding = torch.zeros(padding_shape, device=self.device, dtype=self.dtype)
        image_latents_padded = torch.cat([image_latents, latent_padding], dim=1)  # [B, F, C, H, W]

        # Prepare rotary embeddings
        logger.info("Preparing rotary embeddings...")
        image_rotary_emb = self._prepare_rotary_positional_embeddings(height, width, latent_num_frames)

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # Scale initial noise
        latents = latents * self.scheduler.init_noise_sigma

        logger.info("Starting denoising loop (%d steps)...", num_inference_steps)
        for i, t in enumerate(timesteps):
            self._current_timestep = t

            # Expand latents for CFG
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Expand image latents for CFG
            latent_image_input = (
                torch.cat([image_latents_padded] * 2) if do_classifier_free_guidance else image_latents_padded
            )

            # Concatenate noise and image latents along channel dimension
            # [B, F, C, H, W] + [B, F, C, H, W] -> [B, F, 2C, H, W]
            latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)

            # Prepare prompt embeds for CFG
            if do_classifier_free_guidance:
                prompt_embeds_input = torch.cat([negative_prompt_embeds, prompt_embeds])
            else:
                prompt_embeds_input = prompt_embeds

            # Expand timestep to match batch dimension (for CFG)
            batch_size = latent_model_input.shape[0]
            timestep = t.expand(batch_size).to(latent_model_input.dtype)

            # Predict noise
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                encoder_hidden_states=prompt_embeds_input,
                timestep=timestep,
                image_rotary_emb=image_rotary_emb,
                return_dict=False,
            )[0]

            # CFG
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if (i + 1) % 10 == 0:
                logger.debug("Step %d/%d", i + 1, num_inference_steps)

        self._current_timestep = None
        logger.info("Decoding latents...")

        # Decode latents
        # CogVideoX VAE expects [B, C, F, H, W]
        latents = latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W] -> [B, C, F, H, W]

        # Unscale
        if hasattr(self.vae.config, "scaling_factor"):
            latents = latents / self.vae.config.scaling_factor

        latents = latents.to(dtype=torch.float32)
        video = self.vae.decode(latents).sample

        # video: [B, C, F, H, W] in range [-1, 1]
        logger.info("Output shape: %s", video.shape)
        logger.info("Output range: [%.3f, %.3f]", video.min().item(), video.max().item())

        return DiffusionOutput(output=video)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights using AutoWeightsLoader for vLLM integration."""
        loader = AutoWeightsLoader(self)
        loaded_weights = loader.load_weights(weights)
        # Record components already loaded via from_pretrained
        loaded_weights |= {f"text_encoder.{name}" for name, _ in self.text_encoder.named_parameters()}
        loaded_weights |= {f"vae.{name}" for name, _ in self.vae.named_parameters()}
        return loaded_weights
