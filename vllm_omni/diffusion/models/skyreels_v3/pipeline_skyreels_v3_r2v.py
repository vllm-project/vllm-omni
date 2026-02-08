# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
SkyReels-V3 Image-to-Video (R2V) Pipeline Implementation.

This pipeline supports generating videos from reference images using the
SkyReels-V3 multimodal video generation model.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import PIL.Image
import torch
from diffusers import AutoencoderKLWan
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportImageInput
from vllm_omni.diffusion.models.schedulers import FlowUniPCMultistepScheduler
from vllm_omni.diffusion.models.skyreels_v3.skyreels_v3_transformer import SkyReelsTransformer3DModel
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniTextPrompt

logger = logging.getLogger(__name__)


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: torch.Generator | None = None,
    sample_mode: str = "sample",
):
    """Retrieve latents from VAE encoder output."""
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def load_transformer_config(model_path: str, subfolder: str = "transformer", local_files_only: bool = True) -> dict:
    """Load transformer config from model directory or HF Hub."""
    if local_files_only:
        config_path = os.path.join(model_path, subfolder, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                return json.load(f)
    else:
        try:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=model_path,
                filename=f"{subfolder}/config.json",
            )
            with open(config_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def create_transformer_from_config(config: dict) -> SkyReelsTransformer3DModel:
    """Create SkyReelsTransformer3DModel from config dict."""
    kwargs = {}

    if "patch_size" in config:
        kwargs["patch_size"] = tuple(config["patch_size"])
    if "num_attention_heads" in config:
        kwargs["num_attention_heads"] = config["num_attention_heads"]
    if "attention_head_dim" in config:
        kwargs["attention_head_dim"] = config["attention_head_dim"]
    if "in_channels" in config:
        kwargs["in_channels"] = config["in_channels"]
    if "out_channels" in config:
        kwargs["out_channels"] = config["out_channels"]
    if "text_dim" in config:
        kwargs["text_dim"] = config["text_dim"]
    if "ffn_dim" in config:
        kwargs["ffn_dim"] = config["ffn_dim"]
    if "num_layers" in config:
        kwargs["num_layers"] = config["num_layers"]
    if "cross_attn_norm" in config:
        kwargs["cross_attn_norm"] = config["cross_attn_norm"]
    if "eps" in config:
        kwargs["eps"] = config["eps"]
    if "image_dim" in config:
        kwargs["image_dim"] = config["image_dim"]
    if "max_seq_len" in config:
        kwargs["max_seq_len"] = config["max_seq_len"]

    return SkyReelsTransformer3DModel(**kwargs)


def get_skyreels_v3_r2v_post_process_func(
    od_config: OmniDiffusionConfig,
):
    """Post-process function for R2V: convert latents to video frames."""
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


def get_skyreels_v3_r2v_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    """Pre-process function for R2V: load and resize input image."""
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
                (request.sampling_params.width, request.sampling_params.height),  # type: ignore
                PIL.Image.Resampling.LANCZOS,
            )
            prompt["multi_modal_data"]["image"] = image  # type: ignore

            # Preprocess for VAE
            prompt["additional_information"]["preprocessed_image"] = video_processor.preprocess(
                image, height=request.sampling_params.height, width=request.sampling_params.width
            )
            request.prompts[i] = prompt
        return request

    return pre_process_func


class SkyReelsV3R2VPipeline(nn.Module, SupportImageInput, CFGParallelMixin):
    """
    SkyReels-V3 Image-to-Video (R2V) Pipeline.

    Generates videos from reference images using text prompts.
    """

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
    ):
        super().__init__()
        self.od_config = od_config
        model = od_config.model

        # Load model components
        loader = DiffusersPipelineLoader(model, local_files_only=od_config.local_files_only)

        # Load VAE
        self.vae = loader.load_model(AutoencoderKLWan, "vae")
        self.vae.requires_grad_(False)
        self.vae.eval()

        # Load text encoder and tokenizer
        self.text_encoder = loader.load_model(UMT5EncoderModel, "text_encoder")
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        self.tokenizer = loader.load_tokenizer(AutoTokenizer, "tokenizer")

        # Load CLIP for image conditioning
        self.image_encoder = loader.load_model(CLIPVisionModel, "image_encoder")
        self.image_encoder.requires_grad_(False)
        self.image_encoder.eval()
        self.image_processor = loader.load_processor(CLIPImageProcessor, "image_processor")

        # Load or create transformer
        transformer_config = load_transformer_config(model, local_files_only=od_config.local_files_only)
        if transformer_config:
            self.transformer = create_transformer_from_config(transformer_config)
        else:
            # Default configuration for SkyReels-V3-R2V-14B
            self.transformer = SkyReelsTransformer3DModel(
                num_attention_heads=16,
                attention_head_dim=88,
                in_channels=16,
                num_layers=28,
                text_dim=4096,
                image_dim=1024,  # CLIP image embedding dimension
                patch_size=(1, 2, 2),
            )

        # Load transformer weights
        loader.load_module_weights(self.transformer, "transformer")

        # Load scheduler
        self.scheduler = loader.load_scheduler(FlowUniPCMultistepScheduler, "scheduler")

        # Move to device
        device = get_local_device()
        self.to(device)

        # Set VAE scaling factor
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device,
        num_videos_per_prompt: int = 1,
        do_classifier_free_guidance: bool = False,
        negative_prompt: str | list[str] | None = None,
    ) -> torch.Tensor:
        """Encode text prompt using UMT5."""
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # Tokenize
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)

        # Encode
        prompt_embeds = self.text_encoder(text_input_ids)[0]

        # Duplicate for multiple videos per prompt
        if num_videos_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_videos_per_prompt, dim=0)

        # Handle classifier-free guidance
        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = [""] * batch_size
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size

            uncond_tokens = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_input_ids = uncond_tokens.input_ids.to(device)
            negative_prompt_embeds = self.text_encoder(uncond_input_ids)[0]

            if num_videos_per_prompt > 1:
                negative_prompt_embeds = negative_prompt_embeds.repeat_interleave(num_videos_per_prompt, dim=0)

            # Concatenate for CFG
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def encode_image(
        self,
        image: PIL.Image.Image | torch.Tensor,
        device: torch.device,
        num_videos_per_prompt: int = 1,
    ) -> torch.Tensor:
        """Encode reference image using CLIP."""
        if isinstance(image, PIL.Image.Image):
            image = self.image_processor(images=image, return_tensors="pt").pixel_values
        image = image.to(device=device, dtype=self.image_encoder.dtype)

        # Encode
        image_embeds = self.image_encoder(image).pooler_output

        # Duplicate for multiple videos per prompt
        if num_videos_per_prompt > 1:
            image_embeds = image_embeds.repeat_interleave(num_videos_per_prompt, dim=0)

        return image_embeds

    @torch.no_grad()
    def forward(
        self,
        request: OmniDiffusionRequest,
    ) -> DiffusionOutput:
        """
        Generate video from image and text prompt.

        Args:
            request: Diffusion request containing prompts and parameters

        Returns:
            Generated video frames
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # Extract parameters
        prompt = [p["prompt"] if isinstance(p, dict) else p for p in request.prompts]
        batch_size = len(prompt)

        # Get sampling parameters
        height = request.sampling_params.height or 480
        width = request.sampling_params.width or 832
        num_frames = request.sampling_params.num_frames or 81
        num_inference_steps = request.sampling_params.num_inference_steps or 50
        guidance_scale = request.sampling_params.guidance_scale or 7.5
        num_videos_per_prompt = request.sampling_params.num_videos_per_prompt or 1

        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode text prompt
        prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_videos_per_prompt,
            do_classifier_free_guidance,
            request.sampling_params.negative_prompt,
        )

        # Encode reference image
        images = []
        for p in request.prompts:
            if isinstance(p, dict) and "additional_information" in p:
                img = p["additional_information"].get("preprocessed_image")
                if img is not None:
                    images.append(img)

        if not images:
            raise ValueError("No preprocessed images found in request")

        image_tensor = torch.cat(images, dim=0).to(device=device, dtype=dtype)
        image_embeds = self.encode_image(image_tensor, device, num_videos_per_prompt)

        # Prepare latents
        num_channels_latents = self.transformer.in_channels
        latents_shape = (
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            num_frames,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        generator = torch.Generator(device=device)
        if request.sampling_params.seed is not None:
            generator.manual_seed(request.sampling_params.seed)

        latents = randn_tensor(latents_shape, generator=generator, device=device, dtype=dtype)

        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Denoising loop
        for i, t in enumerate(timesteps):
            # Expand latents for CFG
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Expand timestep
            timestep = t.expand(latent_model_input.shape[0])

            # Predict noise
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                image_hidden_states=image_embeds if do_classifier_free_guidance else image_embeds.repeat(2, 1),
            ).sample

            # Perform CFG
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Compute previous noisy sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # Decode latents
        latents = latents / self.vae.config.scaling_factor
        video = self.vae.decode(latents).sample

        return DiffusionOutput(
            output=video,
            request_id=request.request_id,
        )
