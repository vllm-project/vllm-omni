# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import os
from collections.abc import Iterable

import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl import DistributedAutoencoderKL
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.sdxl.sdxl_unet import SDXLUNet2DConditionModel
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

logger = logging.getLogger(__name__)


def get_sdxl_image_post_process_func(od_config: OmniDiffusionConfig):
    if od_config.output_type == "latent":
        return lambda x: x
    image_processor = VaeImageProcessor(vae_scale_factor=8)

    def post_process_func(images: torch.Tensor):
        if images.device.type != "cpu":
            images = images.cpu()
        return image_processor.postprocess(images)

    return post_process_func


class StableDiffusionXLPipeline(
    nn.Module, CFGParallelMixin, DiffusionPipelineProfilerMixin, SupportsComponentDiscovery
):
    _dit_modules: list[str] = ["unet"]
    _encoder_modules: list[str] = ["text_encoder", "text_encoder_2"]
    _vae_modules: list[str] = ["vae"]
    _resident_modules: list[str] = []

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="unet",
                revision=None,
                prefix="unet.",
                fall_back_to_pt=True,
            )
        ]

        self.device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)
        dtype = od_config.dtype

        sdxl_subfolders = [
            "scheduler",
            "tokenizer",
            "tokenizer_2",
            "text_encoder",
            "text_encoder_2",
            "vae",
        ]
        prefetch_subfolders(model, sdxl_subfolders, local_files_only=local_files_only)

        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
            torch_dtype=torch.float32,
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            model, subfolder="tokenizer_2", local_files_only=local_files_only
        )
        self.text_encoder = from_pretrained_with_prefetch(
            CLIPTextModel.from_pretrained,
            model,
            subfolder="text_encoder",
            prefetch_list=sdxl_subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        )
        self.text_encoder_2 = from_pretrained_with_prefetch(
            CLIPTextModelWithProjection.from_pretrained,
            model,
            subfolder="text_encoder_2",
            prefetch_list=sdxl_subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        )
        self.unet = SDXLUNet2DConditionModel(od_config=od_config)
        self.vae = from_pretrained_with_prefetch(
            DistributedAutoencoderKL.from_pretrained,
            model,
            subfolder="vae",
            prefetch_list=sdxl_subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(self.device)

        self.vae_scale_factor = 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = 77
        self.default_sample_size = 128  # 1024 / 8
        self.output_type = self.od_config.output_type
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    def _get_clip_prompt_embeds(
        self,
        prompt: str | list[str],
        tokenizer: CLIPTokenizer,
        text_encoder: nn.Module,
        num_images_per_prompt: int = 1,
        return_pooled: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        text_encoder_device = next(text_encoder.parameters()).device
        outputs = text_encoder(text_input_ids.to(text_encoder_device), output_hidden_states=True)
        prompt_embeds = outputs.hidden_states[-2].to(dtype=self.od_config.dtype, device=self.device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = None
        if return_pooled:
            pooled_prompt_embeds = outputs[0].to(dtype=self.od_config.dtype, device=self.device)
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(num_images_per_prompt, 1)
            pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds

    def encode_prompt(
        self,
        prompt: str | list[str],
        num_images_per_prompt: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Encode with first CLIP (ViT-L/14, 768-dim hidden states)
        prompt_embeds_1, _ = self._get_clip_prompt_embeds(
            prompt,
            self.tokenizer,
            self.text_encoder,
            num_images_per_prompt,
            return_pooled=False,
        )
        # Encode with second CLIP (OpenCLIP ViT-bigG/14, 1280-dim hidden states + pooled)
        prompt_embeds_2, pooled_prompt_embeds = self._get_clip_prompt_embeds(
            prompt,
            self.tokenizer_2,
            self.text_encoder_2,
            num_images_per_prompt,
            return_pooled=True,
        )

        # Concatenate along feature dimension: 768 + 1280 = 2048
        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)

        return prompt_embeds, pooled_prompt_embeds

    def _get_add_time_ids(
        self,
        original_size: tuple[int, int],
        crops_coords_top_left: tuple[int, int],
        target_size: tuple[int, int],
        batch_size: int,
    ) -> torch.Tensor:
        add_time_ids = torch.tensor(
            [list(original_size) + list(crops_coords_top_left) + list(target_size)],
            dtype=torch.float32,
            device=self.device,
        )
        return add_time_ids.repeat(batch_size, 1)

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=self.device, dtype=self.od_config.dtype)

        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        latents = randn_tensor(shape, generator=generator, device=self.device, dtype=self.od_config.dtype)
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def predict_noise(self, **kwargs):
        result = self.unet(**kwargs)
        return result[0]

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

    def diffuse(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        added_cond_kwargs: dict,
        negative_prompt_embeds: torch.Tensor | None,
        negative_added_cond_kwargs: dict | None,
        do_cfg: bool,
        guidance_scale: float,
    ) -> torch.Tensor:
        for _, t in enumerate(timesteps):
            if self.interrupt:
                continue
            self._current_timestep = t

            # Scale model input
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            timestep = t.expand(latent_model_input.shape[0]).to(device=self.device, dtype=self.od_config.dtype)

            positive_kwargs = {
                "hidden_states": latent_model_input,
                "timestep": timestep,
                "encoder_hidden_states": prompt_embeds,
                "added_cond_kwargs": added_cond_kwargs,
                "return_dict": False,
            }
            if do_cfg:
                negative_kwargs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": negative_prompt_embeds,
                    "added_cond_kwargs": negative_added_cond_kwargs,
                    "return_dict": False,
                }
            else:
                negative_kwargs = None

            noise_pred = self.predict_noise_maybe_with_cfg(
                do_cfg,
                guidance_scale,
                positive_kwargs,
                negative_kwargs,
                cfg_normalize=False,
            )

            latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_cfg)

        return latents

    def forward(
        self,
        req: DiffusionRequestBatch,
        prompt: str | list[str] = "",
        negative_prompt: str | list[str] = "",
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> DiffusionOutput:
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in req.prompts] or prompt
        negative_prompt = [
            "" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in req.prompts
        ] or negative_prompt

        height = req.sampling_params.height or self.default_sample_size * self.vae_scale_factor
        width = req.sampling_params.width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps
        generator = req.sampling_params.generator or generator
        num_images_per_prompt = (
            req.sampling_params.num_outputs_per_prompt
            if req.sampling_params.num_outputs_per_prompt > 0
            else num_images_per_prompt
        )

        self._guidance_scale = req.sampling_params.guidance_scale
        self._current_timestep = None
        self._interrupt = False

        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)

        # Encode prompt
        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
        )

        # Encode negative prompt for CFG
        do_cfg = self.guidance_scale > 1.0
        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None
        if do_cfg:
            negative_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(
                prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
            )

        # Prepare added conditioning
        original_size = (height, width)
        target_size = (height, width)
        crops_coords_top_left = (0, 0)

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            batch_size * num_images_per_prompt,
        )
        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": add_time_ids,
        }
        negative_added_cond_kwargs = None
        if do_cfg:
            negative_added_cond_kwargs = {
                "text_embeds": negative_pooled_prompt_embeds,
                "time_ids": add_time_ids,
            }

        # XPU workaround: set_timesteps uses numpy internally and needs CPU tensors.
        for attr in ("alphas_cumprod", "sigmas", "timesteps"):
            if hasattr(self.scheduler, attr):
                val = getattr(self.scheduler, attr)
                if hasattr(val, "device") and val.device.type != "cpu":
                    setattr(self.scheduler, attr, val.cpu())
        self.scheduler.set_timesteps(num_inference_steps)
        self.scheduler.sigmas = self.scheduler.sigmas.to(self.device)
        self.scheduler.timesteps = self.scheduler.timesteps.to(self.device)
        timesteps = self.scheduler.timesteps

        # Prepare latents (uses scheduler.init_noise_sigma which depends on set_timesteps)
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            self.unet.in_channels,
            height,
            width,
            generator,
            latents,
        )
        self._num_timesteps = len(timesteps)

        # Denoising loop
        latents = self.diffuse(
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_added_cond_kwargs=negative_added_cond_kwargs,
            do_cfg=do_cfg,
            guidance_scale=self.guidance_scale,
        )

        self._current_timestep = None

        if self.output_type == "latent":
            image = latents
        else:
            latents = latents.to(self.vae.dtype)
            latents = latents / self.vae.config.scaling_factor
            image = self.vae.decode(latents, return_dict=False)[0]

        return DiffusionOutput(
            output=image,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
