# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
AniSora V2/V3 Image-to-Video Pipeline using Wan2.1 architecture.

The transformer uses vLLM-Omni's optimized WanTransformer3DModel with
QKVParallelLinear for tensor parallelism support.

This pipeline uses a hybrid loading approach:
- VAE, Text Encoder, Scheduler from official Wan2.1-I2V-14B-Diffusers
- Transformer weights from AniSora (community conversion or official)

Supports:
- IndexTeam/Index-anisora (V2, V3.1, V3.2, anymask)
- aardsoul-music/Wan2.1-Anisora-14B
- ikusa/anisorav2

All these use WanModel architecture with in_dim=36 (16 noise + 16 image + 4 mask).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable

import numpy as np
import PIL.Image
import torch
from diffusers import AutoencoderKLWan
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import (
    AutoTokenizer,
    CLIPImageProcessor,
    CLIPVisionModel,
    UMT5EncoderModel,
)

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformer3DModel
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniTextPrompt

logger = logging.getLogger(__name__)

# Default Wan2.1 base repo for loading shared components (VAE, T5, CLIP).
# To use a local copy or a different repo, pass it via od_config.model_paths["wan_base"].
DEFAULT_WAN_BASE = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"


def get_anisora_v2_i2v_post_process_func(
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


def get_anisora_v2_i2v_pre_process_func(
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
                    f"Unsupported image format {raw_image.__class__}. "
                    'Please correctly set `"multi_modal_data": {"image": <an image object or file path>, …}`'
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


class AniSoraV2I2VPipeline(nn.Module):
    """
    AniSora V2/V3 Image-to-Video Pipeline using Wan2.1 architecture.

    This pipeline uses a hybrid loading approach for diffusers compatibility:
    - VAE, T5, CLIP, Scheduler from official Wan2.1 diffusers repo
    - Transformer weights from AniSora community conversions
    """

    # vLLM uses this flag to decide whether to feed dummy images in warmup
    support_image_input = True

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        """
        Args:
            od_config: OmniDiffusionConfig with model id/path, dtype, and runtime options.
            prefix: Reserved prefix string for compatibility (currently unused).
        """
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        self.dtype = od_config.dtype

        model_path = od_config.model
        wan_base_path = od_config.model_paths.get("wan_base", DEFAULT_WAN_BASE)

        flow_shift = od_config.flow_shift if od_config.flow_shift is not None else 5.0

        # Determine if local files
        local_wan = os.path.exists(wan_base_path)

        logger.info("=== AniSora V2 I2V Pipeline (Hybrid Loading) ===")
        logger.info("AniSora transformer: %s", model_path)
        logger.info("Wan2.1 base (VAE/T5): %s", wan_base_path)

        # Load tokenizer from Wan base
        logger.info("Loading tokenizer from Wan2.1...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            wan_base_path,
            subfolder="tokenizer",
            local_files_only=local_wan,
        )

        # Load T5 text encoder from Wan base
        logger.info("Loading T5 text encoder from Wan2.1...")
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            wan_base_path,
            subfolder="text_encoder",
            torch_dtype=self.dtype,
            local_files_only=local_wan,
        ).to(self.device)

        # Load CLIP image encoder from Wan base (for I2V conditioning)
        logger.info("Loading CLIP image encoder from Wan2.1...")
        try:
            self.image_processor = CLIPImageProcessor.from_pretrained(
                wan_base_path,
                subfolder="image_processor",
                local_files_only=local_wan,
            )
            self.image_encoder = CLIPVisionModel.from_pretrained(
                wan_base_path,
                subfolder="image_encoder",
                torch_dtype=self.dtype,
                local_files_only=local_wan,
            ).to(self.device)
            self.has_image_encoder = True
        except Exception as e:
            raise RuntimeError(
                f"Failed to load CLIP image encoder from {wan_base_path}. "
                "This is required for I2V (image-to-video) conditioning."
            ) from e

        # Load VAE from Wan base
        logger.info("Loading VAE from Wan2.1...")
        self.vae = AutoencoderKLWan.from_pretrained(
            wan_base_path,
            subfolder="vae",
            torch_dtype=torch.float32,  # VAE in float32 for precision
            local_files_only=local_wan,
        ).to(self.device)

        # Load transformer using vLLM-Omni's optimized WanTransformer3DModel
        logger.info("Loading transformer from AniSora: %s...", model_path)

        # Wan2.1 I2V config values (in_channels=36 for I2V: 16 noise + 16 image + 4 mask)
        self.transformer = WanTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=40,
            attention_head_dim=128,
            in_channels=36,
            out_channels=16,
            text_dim=4096,
            freq_dim=256,
            ffn_dim=13824,
            num_layers=40,
            cross_attn_norm=True,
            eps=1e-6,
            image_dim=1280,
            added_kv_proj_dim=5120,
            rope_max_seq_len=1024,
        )

        # Tell the framework to load transformer weights from AniSora checkpoint
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_path,
                subfolder=None,
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
        ]

        # Initialize scheduler
        logger.info("Initializing scheduler...")
        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=flow_shift,
            prediction_type="flow_prediction",
        )

        # VAE scale factors
        self.vae_scale_factor_temporal = getattr(self.vae.config, "temporal_compression_ratio", 4)
        self.vae_scale_factor_spatial = getattr(self.vae.config, "spatial_compression_ratio", 8)

        self._current_timestep = None
        self._device_moved = False
        logger.info("Pipeline loaded successfully!")

    def to(self, device):
        """Move pipeline to device."""
        self.device = device
        self.text_encoder = self.text_encoder.to(device)
        self.vae = self.vae.to(device)
        self.transformer = self.transformer.to(device)
        self.image_encoder = self.image_encoder.to(device)
        return self

    @staticmethod
    def _prompt_clean(text: str) -> str:
        """Clean prompt text."""
        return " ".join(text.strip().split())

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        max_sequence_length: int = 512,
    ):
        """Encode text prompts using T5."""
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)
        prompt_clean = [self._prompt_clean(p) for p in prompt]

        text_inputs = self.tokenizer(
            prompt_clean,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids = text_inputs.input_ids.to(self.device)
        mask = text_inputs.attention_mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(ids, mask).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=self.device)

        # Trim and pad to consistent length
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds],
            dim=0,
        )

        # Negative prompt
        negative_prompt_embeds = None
        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size

            neg_text_inputs = self.tokenizer(
                [self._prompt_clean(p) for p in negative_prompt],
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            ids_neg = neg_text_inputs.input_ids.to(self.device)
            mask_neg = neg_text_inputs.attention_mask.to(self.device)
            seq_lens_neg = mask_neg.gt(0).sum(dim=1).long()

            negative_prompt_embeds = self.text_encoder(ids_neg, mask_neg).last_hidden_state
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.dtype, device=self.device)
            negative_prompt_embeds = [u[:v] for u, v in zip(negative_prompt_embeds, seq_lens_neg)]
            negative_prompt_embeds = torch.stack(
                [
                    torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                    for u in negative_prompt_embeds
                ],
                dim=0,
            )

        return prompt_embeds, negative_prompt_embeds

    @torch.no_grad()
    def encode_image_clip(self, image: PIL.Image.Image) -> torch.Tensor:
        """Encode image using CLIP for conditioning."""
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        image_embeds = self.image_encoder(pixel_values, output_hidden_states=True)
        return image_embeds.hidden_states[-2]

    def prepare_latents(
        self,
        image: torch.Tensor,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None = None,
        last_image: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare latents for I2V generation.

        Returns:
            latents: Initial noise latents [B, C, F, H, W]
            condition: Encoded image condition with mask [B, C+4, F, H, W]
            first_frame_mask: Mask for conditioning
        """
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            latent_height,
            latent_width,
        )

        # Generate noise
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # Prepare image condition
        image = image.unsqueeze(2)  # [B, C, 1, H, W]

        if last_image is None:
            # Pad with zeros for remaining frames
            video_condition = torch.cat(
                [
                    image,
                    image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width),
                ],
                dim=2,
            )
        else:
            # First and last frame conditioning
            last_image = last_image.unsqueeze(2)
            video_condition = torch.cat(
                [
                    image,
                    image.new_zeros(image.shape[0], image.shape[1], num_frames - 2, height, width),
                    last_image,
                ],
                dim=2,
            )

        video_condition = video_condition.to(device=device, dtype=self.vae.dtype)

        # Encode through VAE
        latent_condition = self.vae.encode(video_condition).latent_dist.mode()
        latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

        # Normalize latents
        if hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None:
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, -1, 1, 1, 1)
                .to(latent_condition.device, latent_condition.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1).to(
                latent_condition.device, latent_condition.dtype
            )
            latent_condition = (latent_condition - latents_mean) * latents_std

        latent_condition = latent_condition.to(dtype)

        # Create mask: 1 for frames with condition, 0 for frames to generate
        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_height, latent_width, device=device)
        if last_image is None:
            mask_lat_size[:, :, 1:] = 0  # Only first frame is conditioned
        else:
            mask_lat_size[:, :, 1:-1] = 0  # First and last frames are conditioned

        # Compress mask temporally
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = first_frame_mask.repeat(1, 1, self.vae_scale_factor_temporal, 1, 1)
        mask_lat_size = torch.cat([first_frame_mask, mask_lat_size[:, :, 1:]], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1, self.vae_scale_factor_temporal, latent_height, latent_width)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        mask_lat_size = mask_lat_size.to(latent_condition.device)

        # Concatenate mask with condition [B, C+4, F, H, W]
        condition = torch.cat([mask_lat_size, latent_condition], dim=1)

        # Return placeholder for first_frame_mask (not used in this mode)
        first_frame_mask = torch.ones(
            1,
            1,
            num_latent_frames,
            latent_height,
            latent_width,
            dtype=dtype,
            device=device,
        )

        return latents, condition, first_frame_mask

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
        last_image: PIL.Image.Image | None = None,
        output_type: str | None = None,
        **kwargs,
    ) -> DiffusionOutput:
        """
        Forward pass for vLLM framework integration.
        Extracts parameters from OmniDiffusionRequest and generates video.
        """
        # Move to device once on first forward (after framework loads weights)
        if not self._device_moved:
            self.to(self.device)
            self._device_moved = True
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

        # Prefer PIL image from multi_modal_data for CLIP conditioning.
        if image is None:
            first_prompt = None
            if getattr(req, "prompts", None):
                first_prompt = req.prompts[0]
            if isinstance(first_prompt, dict):
                # First try the PIL image from multi_modal_data (needed for CLIP)
                if "multi_modal_data" in first_prompt and isinstance(first_prompt["multi_modal_data"], dict):
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
            num_frames = _get_sp_attr("num_frames", "frames", default=81)
        if num_inference_steps is None:
            num_inference_steps = _get_sp_attr("num_inference_steps", "steps", default=40)

        if getattr(sampling_params, "guidance_scale_provided", False):
            guidance_scale = sampling_params.guidance_scale
        elif guidance_scale is None:
            guidance_scale = _get_sp_attr("guidance_scale", "cfg_scale", default=5.0)
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
            last_image=last_image,
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
        num_frames: int = 81,
        num_inference_steps: int = 40,
        guidance_scale: float = 5.0,
        generator: torch.Generator | None = None,
        last_image: PIL.Image.Image | None = None,
        output_type: str = "tensor",
    ) -> DiffusionOutput:
        """
        Direct call interface for standalone usage.

        Args:
            prompt: Text prompt(s)
            image: Input image (first frame)
            negative_prompt: Negative prompt(s)
            height: Output height
            width: Output width
            num_frames: Number of output frames (should be 4n+1)
            num_inference_steps: Denoising steps (40 recommended for I2V)
            guidance_scale: Classifier-free guidance scale
            generator: Random generator for reproducibility
            last_image: Optional last frame for interpolation
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
            last_image=last_image,
            output_type=output_type,
        )

    def _generate(
        self,
        prompt: str | list[str],
        image: PIL.Image.Image,
        negative_prompt: str | list[str] | None = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 40,
        guidance_scale: float = 5.0,
        generator: torch.Generator | None = None,
        last_image: PIL.Image.Image | None = None,
        output_type: str = "tensor",
    ) -> DiffusionOutput:
        # Ensure num_frames is compatible with VAE temporal scaling
        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        # Default to empty negative prompt so CFG is not silently skipped
        if negative_prompt is None and guidance_scale > 1.0:
            negative_prompt = ""

        # Encode prompt
        logger.info("Encoding prompts...")
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt, negative_prompt)
        batch_size = prompt_embeds.shape[0]

        do_classifier_free_guidance = guidance_scale > 1.0 and negative_prompt_embeds is not None

        # Encode image with CLIP for additional conditioning
        logger.info("Encoding image with CLIP...")
        image_embeds = self.encode_image_clip(image)
        image_embeds = image_embeds.repeat(batch_size, 1, 1).to(self.dtype)

        # Preprocess image for VAE
        logger.info("Preprocessing image for VAE...")
        from diffusers.video_processor import VideoProcessor

        video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        image_tensor = video_processor.preprocess(image, height=height, width=width)
        image_tensor = image_tensor.to(device=self.device, dtype=torch.float32)

        # Handle last_image if provided
        last_image_tensor = None
        if last_image is not None:
            last_image_tensor = video_processor.preprocess(last_image, height=height, width=width)
            last_image_tensor = last_image_tensor.to(device=self.device, dtype=torch.float32)

        # Prepare latents
        logger.info("Preparing latents...")
        num_channels_latents = self.transformer.config.out_channels  # 16

        latents, condition, _ = self.prepare_latents(
            image=image_tensor,
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=torch.float32,
            device=self.device,
            generator=generator,
            last_image=last_image_tensor,
        )

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        logger.info("Starting denoising loop (%d steps)...", num_inference_steps)
        for i, t in enumerate(timesteps):
            self._current_timestep = t

            # Concatenate noise latents with condition [B, C, F, H, W] + [B, C+4, F, H, W]
            latent_model_input = torch.cat([latents, condition], dim=1).to(self.dtype)

            if do_classifier_free_guidance:
                # Batch conditional and unconditional in a single forward pass
                latent_model_input = torch.cat([latent_model_input] * 2)
                timestep = t.expand(latent_model_input.shape[0])
                prompt_embeds_input = torch.cat([negative_prompt_embeds, prompt_embeds])
                image_embeds_input = torch.cat([image_embeds, image_embeds]) if image_embeds is not None else None

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds_input,
                    encoder_hidden_states_image=image_embeds_input,
                    return_dict=False,
                )[0]

                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                timestep = t.expand(latent_model_input.shape[0])
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_hidden_states_image=image_embeds,
                    return_dict=False,
                )[0]

            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if (i + 1) % 10 == 0:
                logger.debug("Step %d/%d", i + 1, num_inference_steps)

        self._current_timestep = None
        logger.info("Decoding latents...")

        # Decode latents
        latents = latents.to(self.vae.dtype)

        # Denormalize
        if hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None:
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean

        video = self.vae.decode(latents, return_dict=False)[0]

        logger.info("Output shape: %s", video.shape)
        logger.info("Output range: [%.3f, %.3f]", video.min().item(), video.max().item())

        return DiffusionOutput(output=video)

    @staticmethod
    def _convert_anisora_key(key: str) -> str:
        """Convert a single AniSora weight key to diffusers naming convention."""
        new_key = key

        # Block-level: attention layer names
        new_key = new_key.replace(".self_attn.", ".attn1.")
        new_key = new_key.replace(".cross_attn.", ".attn2.")

        # Projection names within attention
        if ".attn1." in new_key or ".attn2." in new_key:
            new_key = new_key.replace(".k.", ".to_k.")
            new_key = new_key.replace(".q.", ".to_q.")
            new_key = new_key.replace(".v.", ".to_v.")
            new_key = new_key.replace(".o.", ".to_out.0.")

        # Image key/value projections for cross-attention
        new_key = new_key.replace(".k_img.", ".add_k_proj.")
        new_key = new_key.replace(".v_img.", ".add_v_proj.")
        new_key = new_key.replace(".norm_k_img.", ".norm_added_k.")

        # FFN: ffn.0 -> ffn.net.0.proj, ffn.2 -> ffn.net.2
        new_key = new_key.replace(".ffn.0.", ".ffn.net.0.proj.")
        new_key = new_key.replace(".ffn.2.", ".ffn.net.2.")

        # Normalization: norm3 -> norm2
        new_key = new_key.replace(".norm3.", ".norm2.")

        # Modulation -> scale_shift_table
        new_key = new_key.replace(".modulation", ".scale_shift_table")

        # Non-block (global) conversions
        new_key = new_key.replace("text_embedding.0.", "condition_embedder.text_embedder.linear_1.")
        new_key = new_key.replace("text_embedding.2.", "condition_embedder.text_embedder.linear_2.")
        new_key = new_key.replace("time_embedding.0.", "condition_embedder.time_embedder.linear_1.")
        new_key = new_key.replace("time_embedding.2.", "condition_embedder.time_embedder.linear_2.")
        new_key = new_key.replace("time_projection.1.", "condition_embedder.time_proj.")
        new_key = new_key.replace("head.head.", "proj_out.")
        if new_key == "head.scale_shift_table":
            new_key = "scale_shift_table"

        # Image embedder
        new_key = new_key.replace("img_emb.proj.0.", "condition_embedder.image_embedder.norm1.")
        new_key = new_key.replace("img_emb.proj.1.", "condition_embedder.image_embedder.ff.net.0.proj.")
        new_key = new_key.replace("img_emb.proj.3.", "condition_embedder.image_embedder.ff.net.2.")
        new_key = new_key.replace("img_emb.proj.4.", "condition_embedder.image_embedder.norm2.")

        return new_key

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights with AniSora→diffusers key conversion.

        The framework feeds weights from weights_sources (prefixed with
        'transformer.'). We strip the prefix, convert AniSora key names
        to diffusers format, then delegate to the transformer's load_weights()
        which handles QKV fusion and TP sharding.
        """
        prefix = "transformer."

        def _convert(weights_iter):
            for name, tensor in weights_iter:
                if name.startswith(prefix):
                    name = name[len(prefix) :]
                yield self._convert_anisora_key(name), tensor

        loaded = {f"transformer.{n}" for n in self.transformer.load_weights(_convert(weights))}
        logger.info("Loaded %d transformer weight entries", len(loaded))

        # Record components already loaded via from_pretrained
        loaded |= {f"text_encoder.{n}" for n, _ in self.text_encoder.named_parameters()}
        loaded |= {f"vae.{n}" for n, _ in self.vae.named_parameters()}
        loaded |= {f"image_encoder.{n}" for n, _ in self.image_encoder.named_parameters()}
        return loaded


# Note: This module is intended to be used via the OmniDiffusion entrypoints.
# A standalone __main__ test harness is intentionally omitted to avoid
# suggesting incorrect instantiation patterns for AniSoraV2I2VPipeline.
