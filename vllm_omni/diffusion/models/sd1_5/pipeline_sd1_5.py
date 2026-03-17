# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Stable Diffusion v1.5 pipeline for vLLM-Omni.

SD v1.5 uses a UNet2DConditionModel backbone (not a DiT), so the UNet is loaded
directly from diffusers rather than being ported to vllm-omni's Attention layer.
"""

from __future__ import annotations

import json
import logging
import os

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import PNDMScheduler
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

logger = logging.getLogger(__name__)


def get_sd15_image_post_process_func(od_config: OmniDiffusionConfig):
    model_name = od_config.model
    if os.path.exists(model_name):
        model_path = model_name
    else:
        model_path = download_weights_from_hf_specific(model_name, None, ["vae/config.json"])
    vae_config_path = os.path.join(model_path, "vae/config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
        vae_scale_factor = 2 ** (len(vae_config["block_out_channels"]) - 1) if "block_out_channels" in vae_config else 8

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    def post_process_func(images: torch.Tensor):
        return image_processor.postprocess(images)

    return post_process_func


class StableDiffusion15Pipeline(nn.Module, CFGParallelMixin, ProgressBarMixin):
    """Stable Diffusion v1.5 pipeline.

    Uses UNet2DConditionModel loaded directly from diffusers. All weights are
    loaded eagerly in __init__ (UNet, VAE, text encoder), so load_weights is
    a no-op.
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
        dtype = od_config.dtype

        model = od_config.model
        local_files_only = os.path.exists(model)

        self.tokenizer = CLIPTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)
        variant = "fp16"
        self.text_encoder = CLIPTextModel.from_pretrained(
            model,
            subfolder="text_encoder",
            torch_dtype=dtype,
            variant=variant,
            local_files_only=local_files_only,
        ).to(self.device)

        self.vae = AutoencoderKL.from_pretrained(
            model,
            subfolder="vae",
            torch_dtype=dtype,
            variant=variant,
            local_files_only=local_files_only,
        ).to(self.device)

        self.unet = UNet2DConditionModel.from_pretrained(
            model,
            subfolder="unet",
            torch_dtype=dtype,
            variant=variant,
            local_files_only=local_files_only,
        ).to(self.device)
        self.transformer = self.unet

        self.scheduler = PNDMScheduler.from_pretrained(model, subfolder="scheduler", local_files_only=local_files_only)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self._guidance_scale = None

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale is not None and self._guidance_scale > 1.0

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        prompt_embeds = self.text_encoder(text_input_ids)[0].to(dtype=dtype)

        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            if isinstance(negative_prompt, str):
                negative_prompt = batch_size * [negative_prompt]
            uncond_tokens = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            negative_prompt_embeds = self.text_encoder(uncond_tokens.input_ids.to(device))[0].to(dtype=dtype)

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self,
        batch_size: int,
        num_channels: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        shape = (
            batch_size,
            num_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def forward(
        self,
        req: OmniDiffusionRequest,
    ) -> DiffusionOutput:
        if len(req.prompts) > 1:
            raise ValueError("This model only supports a single prompt per request.")

        first_prompt = req.prompts[0]
        prompt = first_prompt if isinstance(first_prompt, str) else (first_prompt.get("prompt") or "")
        negative_prompt = None if isinstance(first_prompt, str) else first_prompt.get("negative_prompt")

        sp = req.sampling_params
        height = sp.height or 512
        width = sp.width or 512
        num_steps = sp.num_inference_steps or 50
        guidance_scale = sp.guidance_scale if sp.guidance_scale_provided else 7.5

        self._guidance_scale = guidance_scale

        device = self.device
        dtype = self.unet.dtype

        generator = sp.generator
        if generator is None and sp.seed is not None:
            generator = torch.Generator(device=device).manual_seed(sp.seed)

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            device=device,
            dtype=dtype,
        )

        latents = self.prepare_latents(
            batch_size=1,
            num_channels=self.unet.config.in_channels,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
        )

        self.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.scheduler.timesteps

        with self.progress_bar(total=len(timesteps)) as pbar:
            for t in timesteps:
                latent_model_input = latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                do_cfg = self.do_classifier_free_guidance and negative_prompt_embeds is not None

                positive_kwargs = {
                    "sample": latent_model_input,
                    "timestep": t,
                    "encoder_hidden_states": prompt_embeds,
                }
                if do_cfg:
                    negative_kwargs = {
                        "sample": latent_model_input,
                        "timestep": t,
                        "encoder_hidden_states": negative_prompt_embeds,
                    }
                else:
                    negative_kwargs = None

                noise_pred = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=do_cfg,
                    true_cfg_scale=guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=False,
                )

                latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_cfg)
                pbar.update()

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]

        return DiffusionOutput(output=image)

    def predict_noise(self, **kwargs) -> torch.Tensor:
        return self.unet(**kwargs).sample

    def load_weights(self, weights):
        pass
