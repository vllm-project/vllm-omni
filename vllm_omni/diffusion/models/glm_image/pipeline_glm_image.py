# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
GlmImagePipeline implementation for vLLM-Omni.

This pipeline implements GLM-Image text-to-image generation with:
- AR stage: GlmImageForConditionalGeneration generates prior tokens
- DiT stage: GlmImageTransformer2DModel performs diffusion denoising
- VAE: AutoencoderKL decodes latents to images
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
from collections.abc import Iterable

import numpy as np
import PIL.Image
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import (
    ByT5Tokenizer,
    GlmImageForConditionalGeneration,
    GlmImageProcessor,
    T5EncoderModel,
)

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.glm_image.glm_image_transformer import (
    GlmImageKVCache,
    GlmImageTransformer2DModel,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

logger = logging.getLogger(__name__)


def get_glm_image_pre_process_func(od_config: OmniDiffusionConfig):
    """Get pre-processing function for GLM-Image pipeline.

    Pre-processes condition images before they are sent to the pipeline.
    This is called by DiffusionEngine before batching requests.
    """
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])

    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        block_out_channels = vae_config.get("block_out_channels", [128, 256, 512, 512])
        vae_scale_factor = 2 ** (len(block_out_channels) - 1)

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    # GLM-Image uses patch_size=2 for transformer
    patch_size = 2

    def pre_process_func(requests: list[OmniDiffusionRequest]):
        """Pre-process condition images for Image Edit mode."""
        for req in requests:
            images = req.pil_image
            if images is None:
                # Text-to-image mode, no preprocessing needed
                continue

            if not isinstance(images, list):
                images = [images]

            preprocessed = []
            height, width = None, None

            for img in images:
                if isinstance(img, PIL.Image.Image):
                    img_h, img_w = img.size[::-1]  # PIL is (width, height)
                else:
                    img_h, img_w = img.shape[:2]

                # Align to multiple of vae_scale_factor * patch_size
                multiple_of = vae_scale_factor * patch_size
                img_h = (img_h // multiple_of) * multiple_of
                img_w = (img_w // multiple_of) * multiple_of

                processed = image_processor.preprocess(img, height=img_h, width=img_w)
                preprocessed.append(processed)

                # Use first image dimensions as default
                if height is None:
                    height, width = img_h, img_w

            # Store in request
            req.preprocessed_image = preprocessed
            req.prompt_image = images  # Keep original PIL images
            if req.height is None:
                req.height = height
            if req.width is None:
                req.width = width

        return requests

    return pre_process_func


def get_glm_image_post_process_func(od_config: OmniDiffusionConfig):
    """Get post-processing function for GLM-Image pipeline."""
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])

    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        block_out_channels = vae_config.get("block_out_channels", [128, 256, 512, 512])
        vae_scale_factor = 2 ** (len(block_out_channels) - 1)

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    def post_process_func(images: torch.Tensor) -> list[PIL.Image.Image]:
        return image_processor.postprocess(images, output_type="pil")

    return post_process_func


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    base_shift: float = 0.25,
    max_shift: float = 0.75,
) -> float:
    """Calculate timestep shift based on image sequence length."""
    m = (image_seq_len / base_seq_len) ** 0.5
    mu = m * max_shift + base_shift
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, int]:
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps.
    Handles custom timesteps and sigmas schedules.
    """
    accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
    accepts_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())

    if timesteps is not None and sigmas is not None:
        # Both provided - check if scheduler supports both
        if not accepts_timesteps and not accepts_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep or sigma schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif timesteps is not None:
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        if not accepts_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigma schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: torch.Generator | None = None,
    sample_mode: str = "sample",
) -> torch.Tensor:
    """Extract latents from VAE encoder output."""
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class GlmImagePipeline(nn.Module):
    """
    GLM-Image Pipeline for text-to-image and image-to-image generation.

    This pipeline integrates:
    - AR model (GlmImageForConditionalGeneration): Generates prior image tokens
    - Text encoder (T5EncoderModel): Encodes glyph/text embeddings
    - DiT model (GlmImageTransformer2DModel): Diffusion transformer
    - VAE (AutoencoderKL): Encodes/decodes images to/from latent space

    The pipeline flow:
    1. AR generates prior_token_ids from text prompt
    2. T5 encodes glyph text for text rendering
    3. DiT performs iterative denoising conditioned on prior tokens
    4. VAE decodes final latents to image
    """

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config
        self.device = get_local_device()

        model = od_config.model
        local_files_only = os.path.exists(model)

        if local_files_only:
            model_path = model
        else:
            model_path = download_weights_from_hf_specific(model, od_config.revision, ["*"])

        # Load scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path, subfolder="scheduler", local_files_only=True
        )

        # Load AR model (vision_language_encoder)
        logger.info("Loading GlmImageForConditionalGeneration (AR model)...")
        self.vision_language_encoder = GlmImageForConditionalGeneration.from_pretrained(
            model_path,
            subfolder="vision_language_encoder",
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.vision_language_encoder.eval()

        # Load processor for AR model
        self.processor = GlmImageProcessor.from_pretrained(model_path, subfolder="processor", local_files_only=True)

        # Load text encoder (T5 for glyph embeddings)
        logger.info("Loading T5EncoderModel (glyph encoder)...")
        self.text_encoder = T5EncoderModel.from_pretrained(
            model_path,
            subfolder="text_encoder",
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.text_encoder.eval()

        # Load tokenizer for glyph encoding
        self.tokenizer = ByT5Tokenizer.from_pretrained(model_path, subfolder="tokenizer", local_files_only=True)

        # Load VAE
        logger.info("Loading AutoencoderKL (VAE)...")
        self.vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae", local_files_only=True, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.vae.eval()

        # Load transformer (DiT)
        logger.info("Loading GlmImageTransformer2DModel (DiT)...")
        self.transformer = GlmImageTransformer2DModel(od_config=od_config)

        # Weight sources for DiT loading
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

        # Configure scale factors
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.default_sample_size = 128

        # Get transformer config for patch size
        self._patch_size = getattr(self.transformer, "patch_size", 2)

    # ==================== Input Validation ====================

    def check_inputs(
        self,
        prompt: str | list[str] | None,
        height: int | None,
        width: int | None,
        prompt_embeds: torch.Tensor | None = None,
    ) -> None:
        """Validate input arguments before generation."""
        # Check dimension alignment
        multiple_of = self.vae_scale_factor * self._patch_size
        if height is not None and height % multiple_of != 0:
            logger.warning(
                f"`height` should be divisible by {multiple_of} but is {height}. "
                "Dimensions will be adjusted accordingly."
            )
        if width is not None and width % multiple_of != 0:
            logger.warning(
                f"`width` should be divisible by {multiple_of} but is {width}. Dimensions will be adjusted accordingly."
            )

        # Check prompt/prompt_embeds mutual exclusivity
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. "
                "Please provide only one of the two."
            )
        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`. Cannot leave both undefined.")

        # Check prompt type
        if prompt is not None and not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` must be of type `str` or `list` but is {type(prompt)}")

    # ==================== AR Stage Methods ====================

    @staticmethod
    def _compute_generation_params(
        image_grid_thw: torch.Tensor,
        is_text_to_image: bool,
    ) -> tuple[int, int, int, int]:
        """
        Compute AR generation parameters from image grid.

        Args:
            image_grid_thw: Image grid tensor of shape [N, 3] where each row is [t, h, w]
            is_text_to_image: Whether this is text-to-image (vs image-to-image)

        Returns:
            Tuple of (max_new_tokens, large_image_start_offset, target_grid_h, target_grid_w)
        """
        grid_sizes = []
        grid_hw = []

        for i in range(image_grid_thw.shape[0]):
            t, h, w = image_grid_thw[i].tolist()
            grid_sizes.append(int(h * w))
            grid_hw.append((int(h), int(w)))

        if not is_text_to_image:
            # Image-to-image: only generate target image tokens
            max_new_tokens = grid_sizes[-1] + 1
            large_image_start_offset = 0
            target_grid_h, target_grid_w = grid_hw[-1]
        else:
            # Text-to-image: generate both small preview and large target
            total_tokens = sum(grid_sizes)
            max_new_tokens = total_tokens + 1
            large_image_start_offset = sum(grid_sizes[1:])
            target_grid_h, target_grid_w = grid_hw[0]

        return max_new_tokens, large_image_start_offset, target_grid_h, target_grid_w

    @staticmethod
    def _extract_large_image_tokens(
        outputs: torch.Tensor, input_length: int, large_image_start_offset: int, large_image_tokens: int
    ) -> torch.Tensor:
        """Extract large image tokens from AR output."""
        generated_tokens = outputs[0][input_length:]
        large_image_start = large_image_start_offset
        large_image_end = large_image_start + large_image_tokens
        return generated_tokens[large_image_start:large_image_end]

    @staticmethod
    def _upsample_token_ids(token_ids: torch.Tensor, token_h: int, token_w: int) -> torch.Tensor:
        """Upsample token IDs by 2x using nearest neighbor interpolation."""
        token_ids = token_ids.view(1, 1, token_h, token_w)
        token_ids = torch.nn.functional.interpolate(token_ids.float(), scale_factor=2, mode="nearest").to(
            dtype=torch.long
        )
        token_ids = token_ids.view(1, -1)
        return token_ids

    @torch.inference_mode()
    def generate_prior_tokens(
        self,
        prompt: str,
        height: int,
        width: int,
        image: list[PIL.Image.Image] | None = None,
        factor: int = 32,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """
        Generate prior tokens using the AR model.

        Args:
            prompt: Text prompt for generation
            height: Target image height
            width: Target image width
            image: Optional condition images for image-to-image
            factor: Token factor (default 32)

        Returns:
            Tuple of (prior_token_ids, prior_token_image_ids)
            prior_token_image_ids is a list of tensors, one per condition image
        """
        device = self.vision_language_encoder.device
        height = (height // factor) * factor
        width = (width // factor) * factor
        is_text_to_image = image is None or len(image) == 0

        # Build message content
        content = []
        if image is not None:
            for img in image:
                content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        # Apply chat template - processor will handle target dimensions and build grid
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            target_h=height,
            target_w=width,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        image_grid_thw = inputs.get("image_grid_thw")

        # Compute generation parameters from the full grid
        max_new_tokens, large_image_offset, token_h, token_w = self._compute_generation_params(
            image_grid_thw=image_grid_thw, is_text_to_image=is_text_to_image
        )

        # Process condition images if provided
        # Use image_grid_thw[:-1] to exclude the target image grid (last entry)
        prior_token_image_ids = None
        if image is not None and image_grid_thw is not None and len(image_grid_thw) > 1:
            # Get features only for condition images (exclude target image grid)
            condition_grid = image_grid_thw[:-1]
            prior_token_image_embed = self.vision_language_encoder.get_image_features(
                inputs["pixel_values"], condition_grid
            )
            prior_token_image_embed = torch.cat(prior_token_image_embed, dim=0)
            flat_prior_token_image_ids = self.vision_language_encoder.get_image_tokens(
                prior_token_image_embed, condition_grid
            )
            # Split by image grid sizes and convert to list
            split_sizes = (condition_grid.prod(dim=-1)).tolist()
            prior_token_image_ids_list = torch.split(flat_prior_token_image_ids, split_sizes, dim=0)
            # Convert to list with upsampling
            prior_token_image_ids = []
            for i, token_ids in enumerate(prior_token_image_ids_list):
                grid_t, grid_h, grid_w = condition_grid[i].tolist()
                token_ids = token_ids.view(1, -1)
                # Upsample 2x (from d32 to d64)
                token_ids_upsampled = self._upsample_token_ids(token_ids, grid_h, grid_w)
                prior_token_image_ids.append(token_ids_upsampled)

        # Generate with AR model
        outputs = self.vision_language_encoder.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
        )

        # Extract and upsample tokens
        large_image_tokens = token_h * token_w
        prior_token_ids_d32 = self._extract_large_image_tokens(
            outputs, inputs["input_ids"].shape[-1], large_image_offset, large_image_tokens
        )
        prior_token_ids = self._upsample_token_ids(prior_token_ids_d32, token_h, token_w)

        return prior_token_ids, prior_token_image_ids

    # ==================== Text Encoding Methods ====================

    def get_glyph_texts(self, prompt: str | list[str]) -> list[str]:
        """Extract text within quotes for glyph rendering."""
        prompt = prompt[0] if isinstance(prompt, list) else prompt
        ocr_texts = (
            re.findall(r"'([^']*)'", prompt)
            + re.findall(r"“([^“”]*)”", prompt)
            + re.findall(r'"([^"]*)"', prompt)
            + re.findall(r"「([^「」]*)」", prompt)
        )
        return ocr_texts

    def _get_glyph_embeds(
        self,
        prompt: str | list[str],
        max_sequence_length: int = 2048,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Get glyph embeddings from T5 encoder for text rendering."""
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        glyph_texts = self.get_glyph_texts(prompt)
        input_ids = self.tokenizer(
            glyph_texts if len(glyph_texts) > 0 else [""],
            max_length=max_sequence_length,
            truncation=True,
        ).input_ids

        # Pad to even length
        input_ids = [[self.tokenizer.pad_token_id] * ((len(ids) + 1) % 2) + ids for ids in input_ids]
        max_length = max(len(ids) for ids in input_ids)

        attention_mask = torch.tensor(
            [[1] * len(ids) + [0] * (max_length - len(ids)) for ids in input_ids],
            device=device,
        )
        input_ids = torch.tensor(
            [ids + [self.tokenizer.pad_token_id] * (max_length - len(ids)) for ids in input_ids],
            device=device,
        )

        outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
        glyph_embeds = outputs.last_hidden_state[attention_mask.bool()].unsqueeze(0)

        return glyph_embeds.to(device=device, dtype=dtype)

    def encode_prompt(
        self,
        prompt: str | list[str],
        do_classifier_free_guidance: bool = True,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        max_sequence_length: int = 2048,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode prompt into glyph embeddings for text rendering."""
        device = device or self.device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_glyph_embeds(prompt, max_sequence_length, device, dtype)

        seq_len = prompt_embeds.size(1)
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt = [""] * batch_size
            negative_prompt_embeds = self._get_glyph_embeds(negative_prompt, max_sequence_length, device, dtype)
            seq_len = negative_prompt_embeds.size(1)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    # ==================== Latent Preparation ====================

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Prepare random noise latents."""
        if latents is not None:
            return latents.to(device)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"Passed {len(generator)} generators but batch size is {batch_size}.")
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def diffuse(
        self,
        latents: torch.Tensor,
        prior_token_id: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        timesteps: torch.Tensor,
        target_size: torch.Tensor,
        crop_coords: torch.Tensor,
        guidance_scale: float,
        do_classifier_free_guidance: bool,
        kv_caches: GlmImageKVCache | None = None,
    ) -> torch.Tensor:
        """
        Denoising loop for diffusion process with CFG-Parallel support.

        Args:
            latents: Initial noise latents
            prior_token_id: Prior tokens generated by AR model
            prompt_embeds: Encoded positive prompt embeddings (glyph embeddings)
            negative_prompt_embeds: Encoded negative prompt embeddings
            timesteps: Denoising timesteps
            target_size: Target image size tensor [[height, width]]
            crop_coords: Crop coordinates tensor
            guidance_scale: CFG scale
            do_classifier_free_guidance: Whether to apply CFG
            kv_caches: Optional KV cache for Image Edit mode

        Returns:
            Denoised latents ready for VAE decode
        """
        # Prepare conditional/unconditional drop flags
        prior_token_drop_cond = torch.full_like(prior_token_id, False, dtype=torch.bool)
        prior_token_drop_uncond = torch.full_like(prior_token_id, True, dtype=torch.bool)

        transformer_dtype = self.transformer.dtype

        # Enable CFG-parallel: rank0 computes positive, rank1 computes negative
        cfg_parallel_ready = do_classifier_free_guidance and get_classifier_free_guidance_world_size() > 1

        for i, t in enumerate(timesteps):
            latent_model_input = latents.to(transformer_dtype)
            timestep = t.expand(latents.shape[0]) - 1

            if cfg_parallel_ready:
                cfg_group = get_cfg_group()
                cfg_rank = get_classifier_free_guidance_rank()

                if cfg_rank == 0:
                    # Rank 0: Compute positive (conditional) prediction
                    local_pred = self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        prior_token_id=prior_token_id,
                        prior_token_drop=prior_token_drop_cond,
                        timestep=timestep,
                        target_size=target_size,
                        crop_coords=crop_coords,
                        kv_cache=kv_caches,
                        return_dict=False,
                    )[0].float()
                else:
                    # Rank 1: Compute negative (unconditional) prediction
                    local_pred = self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=negative_prompt_embeds,
                        prior_token_id=prior_token_id,
                        prior_token_drop=prior_token_drop_uncond,
                        timestep=timestep,
                        target_size=target_size,
                        crop_coords=crop_coords,
                        kv_cache=kv_caches,
                        return_dict=False,
                    )[0].float()

                # All-gather predictions from all ranks
                gathered = cfg_group.all_gather(local_pred, separate_tensors=True)

                if cfg_rank == 0:
                    # Rank 0: Combine predictions and apply CFG
                    noise_pred_cond = gathered[0]
                    noise_pred_uncond = gathered[1]
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    # Scheduler step
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # Broadcast updated latents to all ranks
                cfg_group.broadcast(latents, src=0)

            else:
                # Sequential CFG (single GPU or no CFG)
                # Conditional forward pass
                noise_pred_cond = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    prior_token_id=prior_token_id,
                    prior_token_drop=prior_token_drop_cond,
                    timestep=timestep,
                    target_size=target_size,
                    crop_coords=crop_coords,
                    kv_cache=kv_caches,
                    return_dict=False,
                )[0].float()

                if do_classifier_free_guidance:
                    # Unconditional forward pass
                    noise_pred_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=negative_prompt_embeds,
                        prior_token_id=prior_token_id,
                        prior_token_drop=prior_token_drop_uncond,
                        timestep=timestep,
                        target_size=target_size,
                        crop_coords=crop_coords,
                        kv_cache=kv_caches,
                        return_dict=False,
                    )[0].float()

                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond

                # Scheduler step
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        return latents

    # ==================== Main Forward Pass ====================

    def _prepare_condition_image_kv_cache(
        self,
        condition_images: list[torch.Tensor],
        prior_token_image_ids: list[torch.Tensor],
        prompt_embeds: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> GlmImageKVCache:
        """
        Prepare KV cache by running condition images through transformer at timestep 0.

        This is used for Image Edit mode where we need to cache the condition image's
        KV states for cross-attention during denoising.

        Args:
            condition_images: List of preprocessed condition images
            prior_token_image_ids: Prior token IDs for each condition image from AR model
            prompt_embeds: Prompt embeddings (used to get dtype)
            generator: Optional random generator

        Returns:
            GlmImageKVCache with cached KV states from condition images
        """
        kv_caches = self.transformer.create_kv_cache()
        kv_caches.set_mode("write")

        # Prepare VAE normalization parameters
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.latent_channels, 1, 1)
            .to(device=self.device, dtype=prompt_embeds.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.latent_channels, 1, 1)
            .to(device=self.device, dtype=prompt_embeds.dtype)
        )

        # Process each condition image through transformer to populate KV cache
        for condition_image, condition_prior_token_id in zip(condition_images, prior_token_image_ids):
            condition_image = condition_image.to(device=self.device, dtype=prompt_embeds.dtype)

            # Encode condition image to latent space
            # Use argmax (mode) for deterministic encoding of condition images
            condition_latent = retrieve_latents(
                self.vae.encode(condition_image), generator=generator, sample_mode="argmax"
            )
            condition_latent = (condition_latent - latents_mean) / latents_std

            # Run forward pass at timestep 0 to cache KV states
            # Empty encoder_hidden_states since we only want to cache image features
            _ = self.transformer(
                hidden_states=condition_latent,
                encoder_hidden_states=torch.zeros_like(prompt_embeds)[:1, :0, ...],
                prior_token_id=condition_prior_token_id,
                prior_token_drop=torch.full_like(condition_prior_token_id, False, dtype=torch.bool),
                timestep=torch.zeros((1,), device=self.device),
                target_size=torch.tensor([condition_image.shape[-2:]], device=self.device, dtype=prompt_embeds.dtype),
                crop_coords=torch.zeros((1, 2), device=self.device, dtype=prompt_embeds.dtype),
                kv_cache=kv_caches,
                return_dict=False,
            )

        return kv_caches

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        """
        Main generation forward pass.

        Args:
            req: OmniDiffusionRequest with generation parameters

        Returns:
            DiffusionOutput containing generated image
        """
        prompt = req.prompt or ""
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""

        # Get pre-computed prompt embeddings if provided
        prompt_embeds = req.prompt_embeds if isinstance(req.prompt_embeds, torch.Tensor) else None

        # Get condition images for Image Edit mode
        # Use pre-processed images from pre_process_func
        preprocessed_images = req.preprocessed_image
        condition_images = getattr(req, "prompt_image", None)
        img_height = req.height
        img_width = req.width

        is_image_edit = preprocessed_images is not None

        # Use image dimensions as default if available
        height = req.height or img_height or self.default_sample_size * self.vae_scale_factor
        width = req.width or img_width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = req.num_inference_steps or 50
        guidance_scale = req.guidance_scale or 1.5

        # 0. Validate inputs
        self.check_inputs(prompt=prompt, height=height, width=width, prompt_embeds=prompt_embeds)

        batch_size = 1
        do_classifier_free_guidance = guidance_scale > 1.0

        # Set seed if provided
        generator = None
        if req.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(req.seed)

        # 1. Get prior tokens - either from external source (multistage) or generate internally
        # Check if prior_token_ids are provided externally (from AR stage in multistage mode)
        external_prior_tokens = req.extra.get("prior_token_ids") if req.extra else None
        external_prior_image_ids = req.extra.get("prior_token_image_ids") if req.extra else None

        if external_prior_tokens is not None:
            # Multistage mode: use externally provided prior tokens from vLLM AR stage
            logger.info("Using externally provided prior tokens from AR stage...")
            prior_token_id = external_prior_tokens
            if isinstance(prior_token_id, list):
                prior_token_id = torch.tensor(prior_token_id, dtype=torch.long, device=self.device)
            elif isinstance(prior_token_id, torch.Tensor):
                prior_token_id = prior_token_id.to(device=self.device, dtype=torch.long)
            # Ensure shape is [1, num_tokens] for batch processing
            if prior_token_id.dim() == 1:
                prior_token_id = prior_token_id.unsqueeze(0)
            prior_token_image_ids = external_prior_image_ids
        else:
            # Single-stage mode: generate prior tokens with internal AR model
            logger.info("Generating prior tokens with AR model...")
            prior_token_id, prior_token_image_ids = self.generate_prior_tokens(
                prompt=prompt,
                image=condition_images,
                height=height,
                width=width,
            )

        # 2. Encode prompt for glyph embeddings
        logger.info("Encoding prompt...")
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_images_per_prompt=1,
            prompt_embeds=prompt_embeds,
            device=self.device,
            dtype=self.transformer.dtype,
        )

        # 3. Prepare KV cache for Image Edit mode
        kv_caches = None
        if is_image_edit and prior_token_image_ids is not None:
            logger.info("Preparing KV cache for Image Edit mode...")
            kv_caches = self._prepare_condition_image_kv_cache(
                condition_images=preprocessed_images,
                prior_token_image_ids=prior_token_image_ids,
                prompt_embeds=prompt_embeds,
                generator=generator,
            )
            # Switch to read mode for denoising
            kv_caches.set_mode("read")

        # 4. Prepare latents
        latent_channels = self.transformer.in_channels
        latents = self.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=latent_channels,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=self.device,
            generator=generator,
        )

        # 5. Prepare timesteps
        image_seq_len = ((height // self.vae_scale_factor) * (width // self.vae_scale_factor)) // (self._patch_size**2)
        timesteps_array = np.linspace(self.scheduler.config.num_train_timesteps, 1.0, num_inference_steps + 1)[:-1]
        timesteps_array = timesteps_array.astype(np.int64).astype(np.float32)
        sigmas = timesteps_array / self.scheduler.config.num_train_timesteps

        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("base_shift", 0.25),
            self.scheduler.config.get("max_shift", 0.75),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, self.device, timesteps_array.tolist(), sigmas.tolist(), mu=mu
        )

        # 6. Prepare conditioning tensors
        target_size = torch.tensor([[height, width]], dtype=prompt_embeds.dtype, device=self.device)
        crop_coords = torch.zeros((1, 2), dtype=prompt_embeds.dtype, device=self.device)

        # 7. Denoising loop with CFG-parallel support
        logger.info(f"Starting denoising loop with {num_inference_steps} steps...")
        latents = self.diffuse(
            latents=latents,
            prior_token_id=prior_token_id,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            timesteps=timesteps,
            target_size=target_size,
            crop_coords=crop_coords,
            guidance_scale=guidance_scale,
            do_classifier_free_guidance=do_classifier_free_guidance,
            kv_caches=kv_caches,
        )

        # 8. VAE decode
        logger.info("Decoding latents with VAE...")
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.latent_channels, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.latent_channels, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents * latents_std + latents_mean
        image = self.vae.decode(latents, return_dict=False, generator=generator)[0]

        # 9. Leave post-process to vllm-omni pipeline

        return DiffusionOutput(output=image)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load transformer weights."""
        # Filter weights for transformer only
        transformer_weights = (
            (name.replace("transformer.", ""), weight) for name, weight in weights if name.startswith("transformer.")
        )
        return self.transformer.load_weights(transformer_weights)
