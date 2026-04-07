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
from diffusers import AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.utils import is_cosmos_guardrail_available
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from torch import nn
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.cosmos.cosmos_transformer import CosmosTransformer3DModel
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.platforms import current_omni_platform

if is_cosmos_guardrail_available():
    from cosmos_guardrail import CosmosSafetyChecker
else:

    class CosmosSafetyChecker:
        def __init__(self, *args, **kwargs):
            message = (
                "`cosmos_guardrail` is not installed. Please install it to use "
                "the safety checker for Cosmos: `pip install cosmos_guardrail`."
            )
            raise ImportError(message)


DEFAULT_NEGATIVE_PROMPT = (
    "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, "
    "over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, "
    "underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, "
    "jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, "
    "fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. "
    "Overall, the video is of poor quality."
)

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


class CosmosPredict25Pipeline(nn.Module, CFGParallelMixin):
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
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).float()
            if getattr(self.vae.config, "latents_mean", None) is not None
            else None
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).float()
            if getattr(self.vae.config, "latents_std", None) is not None
            else None
        )
        self.latents_mean = latents_mean
        self.latents_std = latents_std

        if self.latents_mean is None or self.latents_std is None:
            raise ValueError("VAE configuration must define both `latents_mean` and `latents_std`.")

        # Load vLLM-omni transformer with config
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

        # self.transformer = create_transformer_from_config(transformer_config)

        # Store the active transformer config
        # self.transformer_config = self.transformer.config

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

    def _build_input_ids_batch(
        self,
        prompt_texts: list[str],
        max_sequence_length: int,
    ) -> torch.Tensor:
        batch = []

        for text in prompt_texts:
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
                            "text": text,
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
            batch.append(torch.tensor(input_ids, dtype=torch.long))

        input_ids_batch = torch.stack(batch, dim=0)
        return input_ids_batch

    def _hidden_states_to_prompt_embeds(
        self,
        hidden_states: tuple[torch.Tensor, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        normalized_hidden_states = []
        for h in hidden_states[1:]:
            normalized = (h - h.mean(dim=-1, keepdim=True)) / (h.std(dim=-1, keepdim=True) + 1e-8)
            normalized_hidden_states.append(normalized)

        prompt_embeds = torch.cat(normalized_hidden_states, dim=-1)
        return prompt_embeds.to(dtype=dtype, device=device)

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
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        input_ids_batch = self._build_input_ids_batch(prompt, max_sequence_length)

        with torch.no_grad():
            outputs = self.text_encoder(
                input_ids_batch.to(device),
                output_hidden_states=True,
            )
        hidden_states = outputs.hidden_states

        prompt_embeds = self._hidden_states_to_prompt_embeds(
            hidden_states,
            device=device,
            dtype=dtype,
        )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        negative_prompt_embeds = None
        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or DEFAULT_NEGATIVE_PROMPT
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            neg_input_ids_batch = self._build_input_ids_batch(negative_prompt, max_sequence_length)

            with torch.no_grad():
                neg_outputs = self.text_encoder(
                    neg_input_ids_batch.to(device),
                    output_hidden_states=True,
                )
            neg_hidden_states = neg_outputs.hidden_states

            negative_prompt_embeds = self._hidden_states_to_prompt_embeds(
                neg_hidden_states,
                device=device,
                dtype=dtype,
            )

            _, seq_len, _ = negative_prompt_embeds.shape
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self,
        batch_size: int,
        num_frames_out: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
    ):
        B = batch_size
        C = 16
        T = (num_frames_out - 1) // self.vae_scale_factor_temporal + 1
        H = height // self.vae_scale_factor_spatial
        W = width // self.vae_scale_factor_spatial
        shape = (B, C, T, H, W)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        cond_mask = torch.zeros((B, 1, T, H, W), dtype=latents.dtype, device=latents.device)
        cond_indicator = torch.zeros((B, 1, T, 1, 1), dtype=latents.dtype, device=latents.device)
        cond_latents = torch.zeros_like(latents)

        return latents, cond_latents, cond_mask, cond_indicator

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        num_frames,
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

    def predict_noise(
        self,
        cond_mask: torch.Tensor | None = None,
        gt_velocity: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Predict noise with velocity replacement for conditioning.

        Args:
            cond_mask: Conditioning mask (B, 1, T, H, W)
            gt_velocity: Ground truth velocity for conditioning frames
            **kwargs: Arguments to pass to transformer

        Returns:
            Predicted noise with velocity replacement applied
        """
        noise_pred_raw = self.transformer(**kwargs)

        if cond_mask is not None and gt_velocity is not None:
            noise_pred = gt_velocity + noise_pred_raw * (1 - cond_mask)
        else:
            noise_pred = noise_pred_raw

        return noise_pred

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        height: int = 704,
        width: int = 1280,
        num_inference_steps: int = 36,
        guidance_scale: float | tuple[float, float] = 7.0,
        frame_num: int = 93,
        output_type: str | None = "np",
        generator: torch.Generator | list[torch.Generator] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        conditional_frame_timestep: float = 0.1,
        **kwargs,
    ) -> DiffusionOutput:
        # Enforce safety checker requirement (NVIDIA Open Model License Agreement)
        if self.safety_checker is None:
            message = (
                f"You have disabled the safety checker for {self.__class__}. This is in violation of the "
                "[NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/"
                "enterprise-software/nvidia-open-model-license). "
                f"Please ensure that you are compliant with the license agreement."
            )
            raise ValueError(message)

        # Get parameters from request or arguments
        if len(req.prompts) == 1:  # If req.prompt is empty, default to prompt & neg_prompt in param list
            prompt = req.prompts[0] if isinstance(req.prompts[0], str) else req.prompts[0].get("prompt")
            negative_prompt = None if isinstance(req.prompts[0], str) else req.prompts[0].get("negative_prompt")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Prompt or prompt_embeds is required for Cosmos Predict 2.5 generation.")

        height = req.sampling_params.height or height
        width = req.sampling_params.width or width
        num_frames = req.sampling_params.num_frames if req.sampling_params.num_frames else frame_num

        # Ensure dimensions are compatible with VAE and patch size
        # patch_size = self.transformer_config.patch_size
        mod_value = 32  # self.vae_scale_factor_spatial * patch_size[1]  # 16*2=32 for TI2V, 8*2=16 for I2V
        height = (height // mod_value) * mod_value
        width = (width // mod_value) * mod_value
        num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps

        # num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps
        guidance_scale = (
            req.sampling_params.guidance_scale if req.sampling_params.guidance_scale_provided else guidance_scale
        )
        num_videos_per_prompt = 1  # req.sampling_params.num_outputs_per_prompt or 1,

        self._guidance_scale = guidance_scale
        self._current_timestep = None

        device = self.device
        dtype = self.transformer.dtype if self.transformer is not None else self.text_encoder.dtype

        # Seed / generator
        if generator is None:
            generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=device).manual_seed(req.sampling_params.seed)

        do_classifier_free_guidance = guidance_scale > 1.0

        # Check text safety before generation (required by NVIDIA Open Model License Agreement)
        if self.safety_checker is not None:
            self.safety_checker.to(device, dtype=torch.float32)
            if prompt is not None:
                prompt_list = [prompt] if isinstance(prompt, str) else prompt
                for p in prompt_list:
                    if not self.safety_checker.check_text_safety(p):
                        raise ValueError(
                            f"Cosmos Guardrail detected unsafe text in the prompt: {p}. Please ensure that the "
                            f"prompt abides by the NVIDIA Open Model License Agreement."
                        )

        self.check_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if prompt_embeds is None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                device=device,
                dtype=dtype,
            )
        else:
            prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
            elif do_classifier_free_guidance:
                raise ValueError(
                    "negative_prompt_embeds must be provided when prompt_embeds are given and guidance_scale > 1."
                )

        batch_size = prompt_embeds.shape[0]
        latents, cond_latent, cond_mask, cond_indicator = self.prepare_latents(
            batch_size=batch_size * num_videos_per_prompt,
            num_frames_out=num_frames,
            height=height,
            width=width,
            dtype=torch.float32,
            device=self.device,
            generator=generator,
        )

        transformer_dtype = self.transformer.dtype

        cond_timestep = torch.ones_like(cond_indicator) * conditional_frame_timestep
        cond_mask = cond_mask.to(transformer_dtype)

        padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        gt_velocity = (latents - cond_latent) * cond_mask

        for i, t in enumerate(timesteps):
            self._current_timestep = t
            sigma_t = (
                torch.tensor(self.scheduler.sigmas[i].item())
                .unsqueeze(0)
                .to(device=self.device, dtype=transformer_dtype)
            )

            in_latents = cond_mask * cond_latent + (1 - cond_mask) * latents
            in_latents = in_latents.to(transformer_dtype)
            in_timestep = cond_indicator * cond_timestep + (1 - cond_indicator) * sigma_t

            do_true_cfg = do_classifier_free_guidance and negative_prompt_embeds is not None
            positive_kwargs = {
                "cond_mask": cond_mask,
                "gt_velocity": gt_velocity,
                "hidden_states": in_latents,
                "condition_mask": cond_mask,
                "timestep": in_timestep,
                "encoder_hidden_states": prompt_embeds,
                "padding_mask": padding_mask,
            }

            if do_true_cfg:
                negative_kwargs = {
                    "cond_mask": cond_mask,
                    "gt_velocity": gt_velocity,
                    "hidden_states": in_latents,
                    "condition_mask": cond_mask,
                    "timestep": in_timestep,
                    "encoder_hidden_states": negative_prompt_embeds,
                    "padding_mask": padding_mask,
                }
            else:
                negative_kwargs = None

            noise_pred = self.predict_noise_maybe_with_cfg(
                do_true_cfg=do_true_cfg,
                true_cfg_scale=guidance_scale,
                positive_kwargs=positive_kwargs,
                negative_kwargs=negative_kwargs,
                cfg_normalize=False,
            )

            latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg)

        # Empty cache before VAE decoding to avoid OOM errors
        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()
        self._current_timestep = None

        # Decode
        if output_type == "latent":
            video = latents
        else:
            # Denormalize latents
            latents_mean = self.latents_mean.to(latents.device, latents.dtype)
            latents_std = self.latents_std.to(latents.device, latents.dtype)
            latents = latents * latents_std + latents_mean

            # Decode to video
            video = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]

            # Check video safety after decode (required by NVIDIA Open Model License Agreement)
            assert self.safety_checker is not None
            self.safety_checker.to(device, dtype=torch.float32)
            video_np = self.video_processor.postprocess_video(video, output_type="np")
            video_np = (video_np * 255).astype(np.uint8)

            # Check safety for each video in batch
            video_batch = []
            for vid in video_np:
                vid = self.safety_checker.check_video_safety(vid)
                video_batch.append(vid)

            video = np.stack(video_batch).astype(np.float32) / 255.0 * 2 - 1
            video = torch.from_numpy(video).permute(0, 4, 1, 2, 3).to(device)

        return DiffusionOutput(output=video)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
