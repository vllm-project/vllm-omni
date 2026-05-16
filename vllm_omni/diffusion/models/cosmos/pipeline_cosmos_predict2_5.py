# Copyright 2025 The NVIDIA Team and The HuggingFace Team. All rights reserved.
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

# Adapted from Diffusers implementation:
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cosmos/pipeline_cosmos2_5_predict.py

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from torch import nn
import torchvision.transforms.functional

from diffusers import AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.cosmos.cosmos_transformer import CosmosTransformer3DModel
from vllm_omni.diffusion.models.cosmos.utils import DEFAULT_NEGATIVE_PROMPT, CosmosSafetyChecker, retrieve_latents
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.platforms import current_omni_platform

logger = logging.getLogger(__name__)


def load_transformer_config(
    model_path: str, subfolder: str = "transformer", local_files_only: bool = True, revision: str | None = None
) -> dict:
    """Load transformer config from model directory or HF Hub."""
    if local_files_only:
        config_path = os.path.join(model_path, subfolder, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                return json.load(f)
    else:
        # Try to download config from HF Hub
        try:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=model_path,
                filename=f"{subfolder}/config.json",
                revision=revision,
            )
            with open(config_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def get_cosmos_predict25_post_process_func(
    od_config: OmniDiffusionConfig,
):

    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(
        video: torch.Tensor,
        output_type: str = "np",
    ):
        if output_type == "latent":
            return video
        return video_processor.postprocess_video(video, output_type=output_type)

    return post_process_func


class CosmosPredict25Pipeline(nn.Module):
    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()

        # Initialize safety checker (required by NVIDIA Open Model License Agreement)
        self.safety_checker = CosmosSafetyChecker()

        self.od_config = od_config

        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        model = od_config.model
        local_files_only = os.path.exists(model)

        # Set up weights sources for transformer(s)
        self.weights_sources = []
        self.weights_sources.append(
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            subfolder="tokenizer",
            local_files_only=local_files_only,
            revision=od_config.revision,
            use_fast=False,
        )
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model,
            subfolder="text_encoder",
            torch_dtype=dtype,
            revision=od_config.revision,
            local_files_only=local_files_only,
        ).to(self.device)

        self.vae = AutoencoderKLWan.from_pretrained(
            model,
            subfolder="vae",
            torch_dtype=dtype,
            revision=od_config.revision,
            local_files_only=local_files_only,
        ).to(self.device)

        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial, resample="bilinear")

        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).float()
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).float()
        self.latents_mean = latents_mean
        self.latents_std = 1.0 / latents_std

        transformer_config = load_transformer_config(
            model,
            subfolder="transformer",
            local_files_only=local_files_only,
            revision=od_config.revision,
        )
        self.transformer = CosmosTransformer3DModel(
            od_config=od_config,
            **transformer_config,
        )

        self.scheduler = UniPCMultistepScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
            revision=od_config.revision,
        )

        if hasattr(self.scheduler, "alphas_cumprod") and isinstance(self.scheduler.alphas_cumprod, torch.Tensor):
            if self.scheduler.alphas_cumprod.is_cuda:
                self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.cpu()
        if hasattr(self.scheduler, "betas") and isinstance(self.scheduler.betas, torch.Tensor):
            if self.scheduler.betas.is_cuda:
                self.scheduler.betas = self.scheduler.betas.cpu()

        self._guidance_scale = None
        self._num_timesteps = None                

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

    def create_condition_mask(
        self,
        latent_shape: tuple,
        device: torch.device,
        dtype: torch.dtype,
        num_cond_latent_frames: int | list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, C, T, H, W = latent_shape
        cond_indicator = torch.zeros(bsz, 1, T, 1, 1, dtype=dtype, device=device)
        if isinstance(num_cond_latent_frames, int):
            num_cond_latent_frames = [num_cond_latent_frames] * bsz
        for idx in range(bsz):
            cond_indicator[idx, :, : num_cond_latent_frames[idx], :, :] = 1.0
        cond_mask = cond_indicator.expand(-1, -1, -1, H, W)
        return cond_indicator, cond_mask

    def _get_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        input_ids_batch = []

        for sample_idx in range(len(prompt)):
            conversations = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant who will provide prompts to an image generator.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt[sample_idx],
                        }
                    ],
                },
            ]
            input_ids = self.tokenizer.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=False,
                add_vision_id=False,
                max_length=max_sequence_length,
                truncation=True,
                padding="max_length",
            )
            input_ids = (
                input_ids["input_ids"] if not isinstance(input_ids, list) and "input_ids" in input_ids else input_ids
            )
            input_ids = torch.LongTensor(input_ids)
            input_ids_batch.append(input_ids)

        input_ids_batch = torch.stack(input_ids_batch, dim=0)

        outputs = self.text_encoder(
            input_ids_batch.to(device),
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states

        normalized_hidden_states = []
        for layer_idx in range(1, len(hidden_states)):
            normalized_state = (hidden_states[layer_idx] - hidden_states[layer_idx].mean(dim=-1, keepdim=True)) / (
                hidden_states[layer_idx].std(dim=-1, keepdim=True) + 1e-8
            )
            normalized_hidden_states.append(normalized_state)

        prompt_embeds = torch.cat(normalized_hidden_states, dim=-1)
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds

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


        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        prompt_embeds = self._get_prompt_embeds(
            prompt=prompt, max_sequence_length=max_sequence_length, device=device, dtype=dtype
        )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or DEFAULT_NEGATIVE_PROMPT
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            negative_prompt_embeds = self._get_prompt_embeds(
                prompt=negative_prompt, max_sequence_length=max_sequence_length, device=device, dtype=dtype
            )

            _, seq_len, _ = negative_prompt_embeds.shape
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self,
        video: torch.Tensor | None,        
        batch_size: int,
        num_channels_latents: int = 16,        
        height: int = 704,
        width: int = 1280,
        num_frames_in: int = 93,
        num_frames_out: int = 93,                
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,        
        latents: torch.Tensor | None = None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f"Generator list length {len(generator)} does not match batch size {batch_size}.")
                
        B = batch_size
        C = num_channels_latents
        T = (num_frames_out - 1) // self.vae_scale_factor_temporal + 1
        H = height // self.vae_scale_factor_spatial
        W = width // self.vae_scale_factor_spatial
        shape = (B, C, T, H, W)

        if num_frames_in == 0:
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                latents = latents.to(device=device, dtype=dtype)

            cond_indicator, cond_mask = self.create_condition_mask(shape, device, dtype, 0)
            cond_latents = torch.zeros_like(latents)

            return latents, cond_latents, cond_mask, cond_indicator

        else:
            if video is None:
                raise ValueError("`video` must be provided when `num_frames_in` is greater than 0.")
            needs_preprocessing = not (isinstance(video, torch.Tensor) and video.ndim == 5 and video.shape[1] == 3)
            if needs_preprocessing:
                video = self.video_processor.preprocess_video(video, height, width)
            video = video.to(device=device, dtype=self.vae.dtype)

            if isinstance(generator, list):
                cond_latents = [
                    retrieve_latents(
                        self.vae.encode(video[i].unsqueeze(0)), generator=generator[i], sample_mode="argmax"
                    )
                    for i in range(batch_size)
                ]
            else:
                cond_latents = [
                    retrieve_latents(self.vae.encode(vid.unsqueeze(0)), generator, sample_mode="argmax")
                    for vid in video
                ]

            cond_latents = torch.cat(cond_latents, dim=0).to(dtype)

            latents_mean = self.latents_mean.to(device=device, dtype=dtype)
            latents_std = self.latents_std.to(device=device, dtype=dtype)
            cond_latents = (cond_latents - latents_mean) * latents_std

            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                latents = latents.to(device=device, dtype=dtype)

            num_cond_latent_frames = (num_frames_in - 1) // self.vae_scale_factor_temporal + 1
            cond_indicator, cond_mask = self.create_condition_mask(shape, device, dtype, num_cond_latent_frames)

            return latents, cond_latents, cond_mask, cond_indicator

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and "
                f"`negative_prompt_embeds`: {negative_prompt_embeds}. "
                "Please make sure to only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")


    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        height: int = 704,
        width: int = 1280,
        num_inference_steps: int = 36,
        guidance_scale: float = 7.0,
        frame_num: int = 93,
        output_type: str | None = "np",
        generator: torch.Generator | list[torch.Generator] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        # Cosmos-specific
        image: torch.Tensor | None = None,
        video: torch.Tensor | None = None,
        num_latent_conditional_frames: int = 2,
        conditional_frame_timestep: float = 0.0001,
        **kwargs,
    ) -> DiffusionOutput:
        if self.safety_checker is None:
            raise ValueError(
                f"You have disabled the safety checker for {self.__class__}. This is in violation of the "
                "[NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). "
                f"Please ensure that you are compliant with the license agreement."
            )

        extra = getattr(req.sampling_params, "extra_args", {}) or {}
        image = extra.get("image", image)
        video = extra.get("video", video)
        num_latent_conditional_frames = extra.get("num_latent_conditional_frames", num_latent_conditional_frames)
        conditional_frame_timestep = extra.get("conditional_frame_timestep", conditional_frame_timestep)        

        if image is not None and video is not None:
            raise ValueError("image and video cannot be provided simultaneously")
        
        if len(req.prompts) == 1:  # If req.prompt is empty, default to prompt & neg_prompt in param list
            prompt = req.prompts[0] if isinstance(req.prompts[0], str) else req.prompts[0].get("prompt")
            negative_prompt = None if isinstance(req.prompts[0], str) else req.prompts[0].get("negative_prompt")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Prompt or prompt_embeds is required for Cosmos Predict 2.5 generation.")

        device = self.device
        dtype = self.transformer.dtype if self.transformer is not None else self.text_encoder.dtype

        if isinstance(self.safety_checker, CosmosSafetyChecker):
            self.safety_checker.to(device, dtype=dtype)
            if prompt is not None:
                prompt_list = [prompt] if isinstance(prompt, str) else prompt
                for p in prompt_list:
                    if not self.safety_checker.check_text_safety(p):
                        raise ValueError(
                            f"Cosmos Guardrail detected unsafe text in the prompt: {p}. Please ensure that the "
                            f"prompt abides by the NVIDIA Open Model License Agreement."
                        )

        height = req.sampling_params.height or height
        width = req.sampling_params.width or width
        num_frames = req.sampling_params.num_frames if req.sampling_params.num_frames else frame_num
        num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps
        num_videos_per_prompt = req.sampling_params.num_outputs_per_prompt or 1
        max_sequence_length=req.sampling_params.max_sequence_length or 512

        # Ensure dimensions are compatible with VAE and patch size
        patch_size = self.transformer.config.patch_size
        mod_value = self.vae_scale_factor_spatial * patch_size[1]
        height = (height // mod_value) * mod_value
        width = (width // mod_value) * mod_value

        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale

        self._guidance_scale = guidance_scale

        self.check_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        # Seed / generator
        if generator is None:
            generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=device).manual_seed(req.sampling_params.seed)

        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        else:
            prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
            elif self.do_classifier_free_guidance:
                raise ValueError(
                    "negative_prompt_embeds must be provided when prompt_embeds are given and guidance_scale > 1."
                )

        batch_size = prompt_embeds.shape[0]

        is_image = image is not None
        is_video = video is not None

        if is_image:
            image = torchvision.transforms.functional.to_tensor(image).unsqueeze(0)
            video = torch.cat([image, torch.zeros_like(image).repeat(num_frames - 1, 1, 1, 1)], dim=0)
            video = video.unsqueeze(0)
            video = self.video_processor.preprocess_video(video, height, width)
            num_frames_in = 1

        elif is_video:
            if batch_size != 1:
                raise ValueError(f"batch_size must be 1 for video input (given {batch_size})")

            if num_latent_conditional_frames not in [1, 2]:
                raise ValueError(
                    f"num_latent_conditional_frames must be 1 or 2, but got {num_latent_conditional_frames}"
                )

            needs_preprocessing = not (isinstance(video, torch.Tensor) and video.ndim == 5 and video.shape[1] == 3)
            if needs_preprocessing:
                video = self.video_processor.preprocess_video(video, height, width)

            frames_to_extract = 4 * (num_latent_conditional_frames - 1) + 1
            total_input_frames = video.shape[2]
            if total_input_frames < frames_to_extract:
                raise ValueError(
                    f"Input video has only {total_input_frames} frames but Video2World requires at least "
                    f"{frames_to_extract} frames for conditioning."
                )

            video = video[:, :, -frames_to_extract:, :, :]
            if video.shape[2] < num_frames:
                n_pad_frames = num_frames - video.shape[2]
                last_frame = video[:, :, -1:, :, :]
                pad_frames = last_frame.repeat(1, 1, n_pad_frames, 1, 1)
                video = torch.cat((video, pad_frames), dim=2)

            num_frames_in = frames_to_extract

        else:
            video = torch.zeros(batch_size, 3, num_frames, height, width, dtype=torch.uint8)
            num_frames_in = 0

        num_frames_out = num_frames
        assert num_frames_in <= num_frames_out, f"expected ({num_frames_in=}) <= ({num_frames_out=})"
        video = video.to(device=device, dtype=self.vae.dtype)
                           
        num_channels_latents = self.transformer.config.in_channels - 1
        latents, cond_latent, cond_mask, cond_indicator = self.prepare_latents(
            video=video,
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,            
            height=height,
            width=width,
            num_frames_in=num_frames_in,
            num_frames_out=num_frames,                                   
            dtype=torch.float32,
            device=self.device,
            generator=generator,
            latents=req.sampling_params.latents,
        )

        padding_mask = latents.new_zeros(1, 1, height, width, dtype=dtype)
        cond_mask = cond_mask.to(dtype)

        # Timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        gt_velocity = (latents - cond_latent) * cond_mask

        for i, t in enumerate(timesteps):
            self._current_timestep = t

            sigma_t = self.scheduler.sigmas[i].expand(batch_size).to(device=device, dtype=torch.float32)

            if conditional_frame_timestep >= 0:
                in_timestep = cond_indicator * conditional_frame_timestep + (1 - cond_indicator) * sigma_t.view(
                    batch_size, 1, 1, 1, 1
                )
            else:
                in_timestep = sigma_t
            
            in_latents = cond_mask * cond_latent + (1 - cond_mask) * latents
            in_latents = in_latents.to(dtype)

            noise_pred = self.transformer(
                hidden_states=in_latents,
                condition_mask=cond_mask,
                timestep=in_timestep,
                encoder_hidden_states=prompt_embeds,
                padding_mask=padding_mask,
                return_dict=False,
            )[0]            

            noise_pred = gt_velocity + noise_pred * (1 - cond_mask)

            if self.do_classifier_free_guidance:
                noise_pred_neg = self.transformer(
                    hidden_states=in_latents,
                    condition_mask=cond_mask,
                    timestep=in_timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0]                

                noise_pred_neg = gt_velocity + noise_pred_neg * (1 - cond_mask)
                noise_pred = noise_pred + self.guidance_scale * (noise_pred - noise_pred_neg)

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Empty cache before VAE decoding to avoid OOM errors
        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()
        self._current_timestep = None

        # Decode
        if output_type == "latent":
            video = latents
        else:
            latents_mean = self.latents_mean.to(latents.device, latents.dtype)
            latents_std = self.latents_std.to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]

            if isinstance(self.safety_checker, CosmosSafetyChecker):
                self.safety_checker.to(device, dtype=torch.float32)
                video_np = self.video_processor.postprocess_video(video, output_type="np")
                video_np = (video_np * 255).astype(np.uint8)
                video_batch = []
                for vid in video_np:
                    vid = self.safety_checker.check_video_safety(vid)
                    video_batch.append(vid)
                video = np.stack(video_batch).astype(np.float32) / 255.0 * 2 - 1
                video = torch.from_numpy(video).permute(0, 4, 1, 2, 3).to(device)

        return DiffusionOutput(output=video)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Load weights using AutoWeightsLoader for vLLM transformer
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
