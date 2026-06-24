# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Krea 2 text-to-image pipeline for vLLM-Omni."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn as nn
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoTokenizer, Qwen3VLModel
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.krea2.krea2_transformer import Krea2Transformer2DModel
from vllm_omni.diffusion.models.krea2.preprocess_krea2 import (
    denormalize_latents,
    pack_latents,
    prepare_position_ids,
    unpack_latents,
)
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

logger = logging.getLogger(__name__)

DEFAULT_TEXT_ENCODER_SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)

PROMPT_TEMPLATE_PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n"
)
PROMPT_TEMPLATE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
PROMPT_TEMPLATE_START_IDX = 34
PROMPT_TEMPLATE_NUM_SUFFIX_TOKENS = 5


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 6400,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def get_krea2_post_process_func(od_config: OmniDiffusionConfig):
    if od_config.output_type == "latent":
        return lambda x: x

    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["*"])

    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = (
            2 ** len(vae_config["temporal_downsample"])
            if "temporal_downsample" in vae_config
            else 8
        )

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

    def post_process_func(images: torch.Tensor):
        return image_processor.postprocess(images)

    return post_process_func


class Krea2Pipeline(
    nn.Module,
    CFGParallelMixin,
    DiffusionPipelineProfilerMixin,
    SupportsComponentDiscovery,
):
    """Krea 2 text-to-image pipeline for vLLM-Omni."""

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.od_config = od_config
        self._execution_device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
        ]

        krea2_subfolders = ["scheduler", "text_encoder", "tokenizer", "vae"]
        prefetch_subfolders(model, krea2_subfolders, local_files_only=local_files_only)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )

        self.text_encoder = from_pretrained_with_prefetch(
            Qwen3VLModel.from_pretrained,
            model,
            subfolder="text_encoder",
            prefetch_list=krea2_subfolders,
            local_files_only=local_files_only,
        ).to(self._execution_device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model, subfolder="tokenizer", local_files_only=local_files_only
        )

        self.vae = from_pretrained_with_prefetch(
            AutoencoderKLQwenImage.from_pretrained,
            model,
            subfolder="vae",
            prefetch_list=krea2_subfolders,
            local_files_only=local_files_only,
        ).to(self._execution_device)

        transformer_kwargs = get_transformer_config_kwargs(
            od_config.tf_model_config, Krea2Transformer2DModel
        )
        self.transformer = Krea2Transformer2DModel(
            od_config=od_config,
            quant_config=od_config.quantization_config,
            **transformer_kwargs,
        )

        self.text_encoder_select_layers = DEFAULT_TEXT_ENCODER_SELECT_LAYERS

        model_config = od_config.model_config or {}
        self.is_distilled = model_config.get("is_distilled", False)
        self.patch_size = 2

        self.vae_scale_factor = (
            2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        )

        self._guidance_scale = 0.0
        self._current_timestep = None
        self._num_timesteps = None
        self._interrupt = False

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def get_text_hidden_states(
        self,
        prompt: str | list[str],
        max_sequence_length: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompt using the Krea 2 chat-template layout.

        Returns (hidden_states, attention_mask) of shapes
        (B, text_seq_len, num_text_layers, text_hidden_dim) and (B, text_seq_len).
        """
        device = self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prefix_idx = PROMPT_TEMPLATE_START_IDX

        text = [PROMPT_TEMPLATE_PREFIX + p for p in prompt]
        text_tokens = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_sequence_length + prefix_idx - PROMPT_TEMPLATE_NUM_SUFFIX_TOKENS,
            return_tensors="pt",
        ).to(device)

        suffix_tokens = self.tokenizer(
            [PROMPT_TEMPLATE_SUFFIX] * len(text), return_tensors="pt"
        ).to(device)

        input_ids = torch.cat([text_tokens.input_ids, suffix_tokens.input_ids], dim=1)
        attention_mask = torch.cat(
            [text_tokens.attention_mask, suffix_tokens.attention_mask], dim=1
        ).bool()

        position_ids = (attention_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden_states = torch.stack(
            [outputs.hidden_states[i] for i in self.text_encoder_select_layers], dim=2
        )

        hidden_states = hidden_states[:, prefix_idx:]
        attention_mask = attention_mask[:, prefix_idx:]
        return hidden_states, attention_mask

    def encode_prompt(
        self,
        prompt: str | list[str],
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_embeds, prompt_embeds_mask = self.get_text_hidden_states(
            prompt, max_sequence_length
        )
        batch_size, seq_len, num_text_layers, dim = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, num_text_layers, dim
        )
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt)
        prompt_embeds_mask = prompt_embeds_mask.view(
            batch_size * num_images_per_prompt, seq_len
        )
        return prompt_embeds, prompt_embeds_mask

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None = None,
    ) -> torch.Tensor:
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        shape = (batch_size, num_channels_latents, latent_height, latent_width)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return pack_latents(
            latents, batch_size, num_channels_latents, latent_height, latent_width, self.patch_size
        )

    def prepare_timesteps(
        self,
        num_inference_steps: int,
        sigmas: list[float] | None,
        image_seq_len: int,
    ) -> tuple[torch.Tensor, int]:
        if sigmas is None:
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

        if self.is_distilled:
            mu = 1.15
        else:
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 6400),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )

        accepts_sigmas = "sigmas" in set(
            __import__("inspect").signature(self.scheduler.set_timesteps).parameters.keys()
        )
        if accepts_sigmas:
            self.scheduler.set_timesteps(sigmas=sigmas, mu=mu)
        else:
            self.scheduler.set_timesteps(num_inference_steps, mu=mu)
        timesteps = self.scheduler.timesteps
        return timesteps, len(timesteps)

    def combine_cfg_noise(
        self,
        positive_noise_pred: torch.Tensor | tuple[torch.Tensor, ...],
        negative_noise_pred: torch.Tensor | tuple[torch.Tensor, ...],
        true_cfg_scale: float,
        cfg_normalize: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Krea 2 non-standard CFG: v_cond + scale * (v_cond - v_uncond)."""
        if isinstance(positive_noise_pred, tuple):
            pos = positive_noise_pred[0]
            neg = negative_noise_pred[0] if isinstance(negative_noise_pred, tuple) else negative_noise_pred
        else:
            pos = positive_noise_pred
            neg = negative_noise_pred[0] if isinstance(negative_noise_pred, tuple) else negative_noise_pred
        result = pos + true_cfg_scale * (pos - neg)
        if isinstance(positive_noise_pred, tuple):
            return (result,)
        return result

    def diffuse(
        self,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds_mask: torch.Tensor | None,
        latents: torch.Tensor,
        position_ids: torch.Tensor,
        timesteps: torch.Tensor,
        do_cfg: bool,
        guidance_scale: float,
    ) -> torch.Tensor:
        self.scheduler.set_begin_index(0)

        for i, t in enumerate(timesteps):
            if self._interrupt:
                continue

            self._current_timestep = t
            timestep = (t / self.scheduler.config.num_train_timesteps).expand(
                latents.shape[0]
            ).to(dtype=latents.dtype, device=latents.device)

            positive_kwargs = {
                "hidden_states": latents,
                "encoder_hidden_states": prompt_embeds,
                "timestep": timestep,
                "position_ids": position_ids,
                "encoder_attention_mask": prompt_embeds_mask,
                "return_dict": False,
            }

            if do_cfg:
                negative_kwargs = {
                    "hidden_states": latents,
                    "encoder_hidden_states": negative_prompt_embeds,
                    "timestep": timestep,
                    "position_ids": position_ids,
                    "encoder_attention_mask": negative_prompt_embeds_mask,
                    "return_dict": False,
                }
            else:
                negative_kwargs = None

            noise_pred = self.predict_noise_maybe_with_cfg(
                do_cfg,
                guidance_scale,
                positive_kwargs,
                negative_kwargs,
            )
            if isinstance(noise_pred, tuple):
                noise_pred = noise_pred[0]

            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 28,
        sigmas: list[float] | None = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        max_sequence_length: int = 512,
        **kwargs: Any,
    ) -> DiffusionOutput:
        prompt = (
            [p if isinstance(p, str) else (p.get("prompt") or "") for p in req.prompts]
            or prompt
        )
        if all(isinstance(p, str) or p.get("negative_prompt") is None for p in req.prompts):
            negative_prompt = None
        elif req.prompts:
            negative_prompt = [
                "" if isinstance(p, str) else (p.get("negative_prompt") or "")
                for p in req.prompts
            ]

        default_size = 1024
        height = req.sampling_params.height or height or default_size
        width = req.sampling_params.width or width or default_size
        num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps
        sigmas = req.sampling_params.sigmas or sigmas
        guidance_scale = (
            req.sampling_params.guidance_scale
            if req.sampling_params.guidance_scale_provided
            else guidance_scale
        )
        generator = req.sampling_params.generator or generator
        num_images_per_prompt = (
            req.sampling_params.num_outputs_per_prompt
            if req.sampling_params.num_outputs_per_prompt > 0
            else num_images_per_prompt
        )

        multiple = self.vae_scale_factor * self.patch_size
        if height % multiple != 0 or width % multiple != 0:
            height = ((height + multiple - 1) // multiple) * multiple
            width = ((width + multiple - 1) // multiple) * multiple

        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = 1

        do_cfg = guidance_scale > 0

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        negative_prompt_embeds = None
        negative_prompt_embeds_mask = None
        if do_cfg:
            if negative_prompt is None:
                negative_prompt = [""] * batch_size
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        num_channels_latents = self.transformer.in_channels // (self.patch_size**2)
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self._execution_device,
            generator,
        )

        grid_height = height // (self.vae_scale_factor * self.patch_size)
        grid_width = width // (self.vae_scale_factor * self.patch_size)
        position_ids = prepare_position_ids(
            prompt_embeds.shape[1], grid_height, grid_width, self._execution_device
        )

        timesteps, num_inference_steps = self.prepare_timesteps(
            num_inference_steps, sigmas, latents.shape[1]
        )
        self._num_timesteps = len(timesteps)

        latents = self.diffuse(
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            latents,
            position_ids,
            timesteps,
            do_cfg,
            guidance_scale,
        )

        self._current_timestep = None

        latents = unpack_latents(latents, height, width, self.vae_scale_factor, self.patch_size)
        latents = latents.to(self.vae.dtype)

        latents_mean = torch.tensor(self.vae.config.latents_mean)
        latents_std = torch.tensor(self.vae.config.latents_std)
        latents = denormalize_latents(latents, latents_mean, latents_std)

        image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]

        return DiffusionOutput(output=image)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
