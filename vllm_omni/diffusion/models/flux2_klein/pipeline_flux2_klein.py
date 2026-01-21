# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.flux2_klein.flux2_klein_transformer import (
    Flux2Transformer2DModel,
)
from vllm_omni.diffusion.models.interface import SupportImageInput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

logger = init_logger(__name__)


class Flux2ImageProcessor(VaeImageProcessor):
    """Image processor to preprocess the reference image for Flux2 klein."""

    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 16,
        vae_latent_channels: int = 32,
        do_normalize: bool = True,
        do_convert_rgb: bool = True,
    ):
        super().__init__(
            do_resize=do_resize,
            vae_scale_factor=vae_scale_factor,
            vae_latent_channels=vae_latent_channels,
            do_normalize=do_normalize,
            do_convert_rgb=do_convert_rgb,
        )

    @staticmethod
    def check_image_input(
        image: PIL.Image.Image,
        max_aspect_ratio: int = 8,
        min_side_length: int = 64,
        max_area: int = 1024 * 1024,
    ) -> PIL.Image.Image:
        if not isinstance(image, PIL.Image.Image):
            raise ValueError(f"Image must be a PIL.Image.Image, got {type(image)}")

        width, height = image.size
        if width < min_side_length or height < min_side_length:
            raise ValueError(f"Image too small: {width}x{height}. Both dimensions must be at least {min_side_length}px")

        aspect_ratio = max(width / height, height / width)
        if aspect_ratio > max_aspect_ratio:
            raise ValueError(
                f"Aspect ratio too extreme: {width}x{height} (ratio: {aspect_ratio:.1f}:1). "
                f"Maximum allowed ratio is {max_aspect_ratio}:1"
            )

        if width * height > max_area:
            logger.warning("Image area exceeds recommended maximum; resizing will be applied.")

        return image

    @staticmethod
    def _resize_to_target_area(image: PIL.Image.Image, target_area: int = 1024 * 1024) -> PIL.Image.Image:
        image_width, image_height = image.size
        scale = math.sqrt(target_area / (image_width * image_height))
        width = int(image_width * scale)
        height = int(image_height * scale)
        return image.resize((width, height), PIL.Image.Resampling.LANCZOS)

    @staticmethod
    def _resize_if_exceeds_area(image: PIL.Image.Image, target_area: int = 1024 * 1024) -> PIL.Image.Image:
        image_width, image_height = image.size
        if image_width * image_height <= target_area:
            return image
        return Flux2ImageProcessor._resize_to_target_area(image, target_area)

    def _resize_and_crop(self, image: PIL.Image.Image, width: int, height: int) -> PIL.Image.Image:
        image_width, image_height = image.size
        left = (image_width - width) // 2
        top = (image_height - height) // 2
        right = left + width
        bottom = top + height
        return image.crop((left, top, right, bottom))

    @staticmethod
    def concatenate_images(images: list[PIL.Image.Image]) -> PIL.Image.Image:
        if len(images) == 1:
            return images[0].copy()

        images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)
        background_color = (255, 255, 255)
        new_img = PIL.Image.new("RGB", (total_width, max_height), background_color)

        x_offset = 0
        for img in images:
            y_offset = (max_height - img.height) // 2
            new_img.paste(img, (x_offset, y_offset))
            x_offset += img.width

        return new_img


def get_flux2_klein_post_process_func(
    od_config: OmniDiffusionConfig,
):
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])

    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = 2 ** (len(vae_config["block_out_channels"]) - 1) if "block_out_channels" in vae_config else 8

    image_processor = Flux2ImageProcessor(vae_scale_factor=vae_scale_factor * 2)

    def post_process_func(images: torch.Tensor):
        return image_processor.postprocess(images)

    return post_process_func


# Copied from diffusers.pipelines.flux2.pipeline_flux2.compute_empirical_mu
def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


class Flux2KleinPipeline(nn.Module, SupportImageInput):
    """Flux2 klein pipeline for text-to-image generation."""

    support_image_input = True

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
        is_distilled: bool = False,
    ):
        super().__init__()
        self.od_config = od_config
        self.is_distilled = is_distilled
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

        self._execution_device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )
        self.text_encoder = Qwen3ForCausalLM.from_pretrained(
            model,
            subfolder="text_encoder",
            local_files_only=local_files_only,
        )
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            model,
            subfolder="tokenizer",
            local_files_only=local_files_only,
        )
        self.vae = AutoencoderKLFlux2.from_pretrained(
            model,
            subfolder="vae",
            local_files_only=local_files_only,
        ).to(self._execution_device)

        transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, Flux2Transformer2DModel)
        self.transformer = Flux2Transformer2DModel(**transformer_kwargs)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = Flux2ImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = 512
        self.default_sample_size = 128

        self._guidance_scale = None
        self._attention_kwargs = None
        self._num_timesteps = None
        self._current_timestep = None
        self._interrupt = False

    @staticmethod
    def _get_qwen3_prompt_embeds(
        text_encoder: Qwen3ForCausalLM,
        tokenizer: Qwen2TokenizerFast,
        prompt: str | list[str],
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        max_sequence_length: int = 512,
        hidden_states_layers: list[int] = (9, 18, 27),
    ):
        dtype = text_encoder.dtype if dtype is None else dtype
        device = text_encoder.device if device is None else device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        all_input_ids = []
        all_attention_masks = []

        for single_prompt in prompt:
            messages = [{"role": "user", "content": single_prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
            )

            all_input_ids.append(inputs["input_ids"])
            all_attention_masks.append(inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(device)

        # Forward pass through the model
        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Only use outputs from intermediate layers and stack them
        out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
        out = out.to(dtype=dtype, device=device)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

        return prompt_embeds

    @staticmethod
    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._prepare_text_ids
    def _prepare_text_ids(
        x: torch.Tensor,  # (B, L, D) or (L, D)
        t_coord: torch.Tensor | None = None,
    ):
        B, L, _ = x.shape
        out_ids = []

        for i in range(B):
            t = torch.arange(1) if t_coord is None else t_coord[i]
            h = torch.arange(1)
            w = torch.arange(1)
            seq_positions = torch.arange(L)

            coords = torch.cartesian_prod(t, h, w, seq_positions)
            out_ids.append(coords)

        return torch.stack(out_ids)

    @staticmethod
    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._prepare_latent_ids
    def _prepare_latent_ids(
        latents: torch.Tensor,  # (B, C, H, W)
    ):
        r"""
        Generates 4D position coordinates (T, H, W, L) for latent tensors.

        Args:
            latents (torch.Tensor):
                Latent tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor:
                Position IDs tensor of shape (B, H*W, 4) All batches share the same coordinate structure: T=0,
                H=[0..H-1], W=[0..W-1], L=0
        """

        batch_size, _, height, width = latents.shape

        t = torch.arange(1)  # [0] - time dimension
        h = torch.arange(height)
        w = torch.arange(width)
        layer_ids = torch.arange(1)  # [0] - layer dimension

        # Create position IDs: (H*W, 4)
        latent_ids = torch.cartesian_prod(t, h, w, layer_ids)

        # Expand to batch: (B, H*W, 4)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

        return latent_ids

    @staticmethod
    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._prepare_image_ids
    def _prepare_image_ids(
        image_latents: list[torch.Tensor],  # [(1, C, H, W), (1, C, H, W), ...]
        scale: int = 10,
    ):
        r"""
        Generates 4D time-space coordinates (T, H, W, L) for a sequence of image latents.

        This function creates a unique coordinate for every pixel/patch across all input latent with different
        dimensions.

        Args:
            image_latents (List[torch.Tensor]):
                A list of image latent feature tensors, typically of shape (C, H, W).
            scale (int, optional):
                A factor used to define the time separation (T-coordinate) between latents. T-coordinate for the i-th
                latent is: 'scale + scale * i'. Defaults to 10.

        Returns:
            torch.Tensor:
                The combined coordinate tensor. Shape: (1, N_total, 4) Where N_total is the sum of (H * W) for all
                input latents.

        Coordinate Components (Dimension 4):
            - T (Time): The unique index indicating which latent image the coordinate belongs to.
            - H (Height): The row index within that latent image.
            - W (Width): The column index within that latent image.
            - L (Seq. Length): A sequence length dimension, which is always fixed at 0 (size 1)
        """

        if not isinstance(image_latents, list):
            raise ValueError(f"Expected `image_latents` to be a list, got {type(image_latents)}.")

        # create time offset for each reference image
        t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
        t_coords = [t.view(-1) for t in t_coords]

        image_latent_ids = []
        for x, t in zip(image_latents, t_coords):
            x = x.squeeze(0)
            _, height, width = x.shape

            x_ids = torch.cartesian_prod(t, torch.arange(height), torch.arange(width), torch.arange(1))
            image_latent_ids.append(x_ids)

        image_latent_ids = torch.cat(image_latent_ids, dim=0)
        image_latent_ids = image_latent_ids.unsqueeze(0)

        return image_latent_ids

    @staticmethod
    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._patchify_latents
    def _patchify_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._unpatchify_latents
    def _unpatchify_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._pack_latents
    def _pack_latents(latents):
        """
        pack latents: (batch_size, num_channels, height, width) -> (batch_size, height * width, num_channels)
        """

        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)

        return latents

    @staticmethod
    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._unpack_latents_with_ids
    def _unpack_latents_with_ids(x: torch.Tensor, x_ids: torch.Tensor) -> list[torch.Tensor]:
        """
        using position ids to scatter tokens into place
        """
        x_list = []
        for data, pos in zip(x, x_ids):
            _, ch = data.shape  # noqa: F841
            h_ids = pos[:, 1].to(torch.int64)
            w_ids = pos[:, 2].to(torch.int64)

            h = torch.max(h_ids) + 1
            w = torch.max(w_ids) + 1

            flat_ids = h_ids * w + w_ids

            out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
            out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

            # reshape from (H * W, C) to (H, W, C) and permute to (C, H, W)

            out = out.view(h, w, ch).permute(2, 0, 1)
            x_list.append(out)

        return torch.stack(x_list, dim=0)

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int, ...] = (9, 18, 27),
    ):
        device = device or self._execution_device

        if prompt is None:
            prompt = ""

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_embeds = self._get_qwen3_prompt_embeds(
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                prompt=prompt,
                device=device,
                max_sequence_length=max_sequence_length,
                hidden_states_layers=text_encoder_out_layers,
            )

        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        text_ids = self._prepare_text_ids(prompt_embeds)
        text_ids = text_ids.to(device)
        return prompt_embeds, text_ids

    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._encode_vae_image
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if image.ndim != 4:
            raise ValueError(f"Expected image dims 4, got {image.ndim}.")

        image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")
        image_latents = self._patchify_latents(image_latents)

        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(image_latents.device, image_latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps)
        image_latents = (image_latents - latents_bn_mean) / latents_bn_std

        return image_latents

    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline.prepare_latents
    def prepare_latents(
        self,
        batch_size,
        num_latents_channels,
        height,
        width,
        dtype,
        device,
        generator: torch.Generator,
        latents: torch.Tensor | None = None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_latents_channels * 4, height // 2, width // 2)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        latent_ids = self._prepare_latent_ids(latents)
        latent_ids = latent_ids.to(device)

        latents = self._pack_latents(latents)  # [B, C, H, W] -> [B, H*W, C]
        return latents, latent_ids

    # Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline.prepare_image_latents
    def prepare_image_latents(
        self,
        images: list[torch.Tensor],
        batch_size,
        generator: torch.Generator,
        device,
        dtype,
    ):
        image_latents = []
        for image in images:
            image = image.to(device=device, dtype=dtype)
            imagge_latent = self._encode_vae_image(image=image, generator=generator)
            image_latents.append(imagge_latent)  # (1, 128, 32, 32)

        image_latent_ids = self._prepare_image_ids(image_latents)

        # Pack each latent and concatenate
        packed_latents = []
        for latent in image_latents:
            # latent: (1, 128, 32, 32)
            packed = self._pack_latents(latent)  # (1, 1024, 128)
            packed = packed.squeeze(0)  # (1024, 128) - remove batch dim
            packed_latents.append(packed)

        # Concatenate all reference tokens along sequence dimension
        image_latents = torch.cat(packed_latents, dim=0)  # (N*1024, 128)
        image_latents = image_latents.unsqueeze(0)  # (1, N*1024, 128)

        image_latents = image_latents.repeat(batch_size, 1, 1)
        image_latent_ids = image_latent_ids.repeat(batch_size, 1, 1)
        image_latent_ids = image_latent_ids.to(device)

        return image_latents, image_latent_ids

    def check_inputs(
        self,
        prompt,
        height,
        width,
        prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        guidance_scale=None,
    ):
        if (
            height is not None
            and height % (self.vae_scale_factor * 2) != 0
            or width is not None
            and width % (self.vae_scale_factor * 2) != 0
        ):
            logger.warning(
                "`height` and `width` have to be divisible by %s but are %s and %s. "
                "Dimensions will be resized accordingly",
                self.vae_scale_factor * 2,
                height,
                width,
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in ["latents", "prompt_embeds"] for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError("`callback_on_step_end_tensor_inputs` must be a subset of ['latents', 'prompt_embeds'].")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if guidance_scale > 1.0 and self.is_distilled:
            logger.warning(f"Guidance scale {guidance_scale} is ignored for step-wise distilled models.")

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1 and not self.is_distilled

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def forward(
        self,
        req: OmniDiffusionRequest,
        image: PIL.Image.Image | list[PIL.Image.Image] | None = None,
        prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float | None = 4.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int, ...] = (9, 18, 27),
    ) -> DiffusionOutput:
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, or list of these):
                `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
                numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
                or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
                list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
                latents as `image`, but if passing latents directly it is not encoded again.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality. For step-wise distilled models,
                `guidance_scale` is ignored.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Note that "" is used as the negative prompt in this pipeline.
                If not provided, will be generated from "".
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.qwenimage.QwenImagePipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.
            text_encoder_out_layers (`Tuple[int]`):
                Layer indices to use in the `text_encoder` to derive the final prompt embeddings.

        Examples:

        Returns:
            [`~pipelines.flux2.Flux2PipelineOutput`] or `tuple`: [`~pipelines.flux2.Flux2PipelineOutput`] if
            `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the
            generated images.
        """

        prompt = req.prompt if req.prompt is not None else prompt
        image = req.pil_image if req.pil_image is not None else image
        height = req.height or height
        width = req.width or width
        num_inference_steps = req.num_inference_steps or num_inference_steps
        guidance_scale = req.guidance_scale if req.guidance_scale is not None else guidance_scale
        generator = req.generator or generator
        req_num_outputs = getattr(req, "num_outputs_per_prompt", None)
        if req_num_outputs and req_num_outputs > 0:
            num_images_per_prompt = req_num_outputs

        if isinstance(req.prompt_embeds, torch.Tensor):
            prompt_embeds = req.prompt_embeds
        if isinstance(req.negative_prompt_embeds, torch.Tensor):
            negative_prompt_embeds = req.negative_prompt_embeds

        if req.max_sequence_length is not None:
            max_sequence_length = req.max_sequence_length
        if getattr(req, "text_encoder_out_layers", None) is not None:
            text_encoder_out_layers = req.text_encoder_out_layers

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            guidance_scale=guidance_scale,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. prepare text embeddings
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        if self.do_classifier_free_guidance:
            negative_prompt = ""
            if prompt is not None and isinstance(prompt, list):
                negative_prompt = [negative_prompt] * len(prompt)
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )

        # 4. process images
        if image is not None and not isinstance(image, list):
            image = [image]

        condition_images = None
        if image is not None:
            for img in image:
                self.image_processor.check_image_input(img)

            condition_images = []
            for img in image:
                image_width, image_height = img.size
                if image_width * image_height > 1024 * 1024:
                    img = self.image_processor._resize_to_target_area(img, 1024 * 1024)
                    image_width, image_height = img.size

                multiple_of = self.vae_scale_factor * 2
                image_width = (image_width // multiple_of) * multiple_of
                image_height = (image_height // multiple_of) * multiple_of
                img = self.image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
                condition_images.append(img)
                height = height or image_height
                width = width or image_width

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 5. prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        image_latents = None
        image_latent_ids = None
        if condition_images is not None:
            image_latents, image_latent_ids = self.prepare_image_latents(
                images=condition_images,
                batch_size=batch_size * num_images_per_prompt,
                generator=generator,
                device=device,
                dtype=self.vae.dtype,
            )

        # 6. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        self._num_timesteps = len(timesteps)

        # 7. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
        self.scheduler.set_begin_index(0)
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            self._current_timestep = t
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            latent_model_input = latents.to(self.transformer.dtype)
            latent_image_ids = latent_ids

            if image_latents is not None:
                latent_model_input = torch.cat([latents, image_latents], dim=1).to(self.transformer.dtype)
                latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=self.attention_kwargs,
                return_dict=False,
            )[0]

            noise_pred = noise_pred[:, : latents.size(1) :]

            if self.do_classifier_free_guidance:
                neg_noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=None,
                    encoder_hidden_states=negative_prompt_embeds,
                    txt_ids=negative_text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0]
                neg_noise_pred = neg_noise_pred[:, : latents.size(1) :]
                noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

        self._current_timestep = None

        latents = self._unpack_latents_with_ids(latents, latent_ids)

        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = latents * latents_bn_std + latents_bn_mean
        latents = self._unpatchify_latents(latents)
        if output_type == "latent":
            image = latents
        else:
            if latents.dtype != self.vae.dtype:
                latents = latents.to(self.vae.dtype)
            image = self.vae.decode(latents, return_dict=False)[0]

        return DiffusionOutput(output=image)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
