# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import json
import logging
import math
import os
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.parallel_state import get_classifier_free_guidance_world_size
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.ultraflux import UltraFluxTransformer2DModel
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

logger = logging.getLogger(__name__)


def get_ultraflux_post_process_func(
    od_config: OmniDiffusionConfig,
):
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
        vae_scale_factor = 2 ** (len(vae_config["block_out_channels"]) - 1) if "block_out_channels" in vae_config else 8

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2)

    def post_process_func(images: torch.Tensor):
        return image_processor.postprocess(images)

    return post_process_func


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
) -> float:
    """
    Calculate shift parameter for timestep sampling based on image sequence length.

    Optimized for UltraFlux to handle various resolutions better:
    - 1024x1024: ~256 seq_len -> shift ~0.5
    - 2048x2048: ~1024 seq_len -> shift ~0.83
    - 4096x4096: ~4096 seq_len -> shift ~1.16

    Args:
        image_seq_len: Current image sequence length
        base_seq_len: Base sequence length (typically 256 for 1024x1024)
        max_seq_len: Maximum sequence length (typically 4096 for 4K)
        base_shift: Base shift value
        max_shift: Maximum shift value

    Returns:
        Calculated shift value (mu)
    """
    # Improved interpolation with better handling for intermediate resolutions
    if image_seq_len <= base_seq_len:
        return base_shift
    elif image_seq_len >= max_seq_len:
        return max_shift

    # Use logarithmic interpolation for better intermediate resolution handling
    # This gives more appropriate shift values for 2048x2048
    log_ratio = math.log(image_seq_len / base_seq_len) / math.log(max_seq_len / base_seq_len)
    mu = base_shift + (max_shift - base_shift) * log_ratio

    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, int]:
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class UltraFluxPipeline(nn.Module, CFGParallelMixin):
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
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

        self.device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model, subfolder="text_encoder", local_files_only=local_files_only
        )
        self.text_encoder_2 = T5EncoderModel.from_pretrained(
            model, subfolder="text_encoder_2", local_files_only=local_files_only
        )
        self.vae = AutoencoderKL.from_pretrained(model, subfolder="vae", local_files_only=local_files_only).to(
            self.device
        )

        self.transformer = UltraFluxTransformer2DModel(od_config=od_config)

        self.tokenizer = CLIPTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)
        self.tokenizer_2 = T5TokenizerFast.from_pretrained(
            model, subfolder="tokenizer_2", local_files_only=local_files_only
        )

        self.stage = None

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8

        if not hasattr(self.vae.config, "scaling_factor") or self.vae.config.scaling_factor is None:
            self.vae.config.scaling_factor = 0.3611
        if not hasattr(self.vae.config, "shift_factor") or self.vae.config.shift_factor is None:
            self.vae.config.shift_factor = 0.1159
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 128

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} "
                f"but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. "
                "Make sure to generate `pooled_prompt_embeds` from same text encoder "
                "that was used to generate `prompt_embeds`."
            )
        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. "
                "Make sure to generate `negative_pooled_prompt_embeds` from same text encoder "
                "that was used to generate `negative_prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    def _get_t5_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        dtype: torch.dtype | None = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer_2)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(text_input_ids.to(self.device), output_hidden_states=False)[0]

        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=self.device)

        _, seq_len, _ = prompt_embeds.shape

        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: str | list[str],
        num_images_per_prompt: int = 1,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length}: {removed_text}"
            )
        prompt_embeds = self.text_encoder(text_input_ids.to(self.device), output_hidden_states=False)

        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=self.device)

        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: str | list[str],
        prompt_2: str | list[str],
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.FloatTensor | None = None,
        pooled_prompt_embeds: torch.FloatTensor | None = None,
        max_sequence_length: int = 512,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=self.device,
            )

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3).to(device=self.device, dtype=dtype)
        text_ids = text_ids.repeat(num_images_per_prompt, 1, 1)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
        latent_image_ids = latent_image_ids.reshape(
            batch_size, latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        return latents, latent_image_ids

    def prepare_timesteps(self, num_inference_steps, sigmas, image_seq_len):
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.16),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            sigmas=sigmas,
            mu=mu,
        )
        return timesteps, num_inference_steps

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

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
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_pooled_prompt_embeds: torch.Tensor,
        latents: torch.Tensor,
        latent_image_ids: torch.Tensor,
        text_ids: torch.Tensor,
        negative_text_ids: torch.Tensor,
        timesteps: torch.Tensor,
        do_true_cfg: bool,
        guidance: torch.Tensor,
        true_cfg_scale: float,
        cfg_normalize: bool = False,
    ) -> torch.Tensor:
        self.scheduler.set_begin_index(0)
        self.transformer.do_true_cfg = do_true_cfg
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            self._current_timestep = t
            timestep = t.expand(latents.shape[0]).to(device=latents.device, dtype=latents.dtype)

            positive_kwargs = {
                "hidden_states": latents,
                "timestep": timestep / 1000,
                "guidance": guidance,
                "pooled_projections": pooled_prompt_embeds,
                "encoder_hidden_states": prompt_embeds,
                "txt_ids": text_ids,
                "img_ids": latent_image_ids,
                "joint_attention_kwargs": self.joint_attention_kwargs,
                "return_dict": False,
            }

            if do_true_cfg:
                negative_kwargs = {
                    "hidden_states": latents,
                    "timestep": timestep / 1000,
                    "guidance": guidance,
                    "pooled_projections": negative_pooled_prompt_embeds,
                    "encoder_hidden_states": negative_prompt_embeds,
                    "txt_ids": negative_text_ids,
                    "img_ids": latent_image_ids,
                    "joint_attention_kwargs": self.joint_attention_kwargs,
                    "return_dict": False,
                }
            else:
                negative_kwargs = None

            noise_pred = self.predict_noise_maybe_with_cfg(
                do_true_cfg,
                true_cfg_scale,
                positive_kwargs,
                negative_kwargs,
                cfg_normalize,
            )

            latents = self.scheduler_step_maybe_with_cfg(noise_pred, t, latents, do_true_cfg)

        return latents

    def check_cfg_parallel_validity(self, true_cfg_scale: float, has_neg_prompt: bool):
        if get_classifier_free_guidance_world_size() == 1:
            return True

        if true_cfg_scale <= 1:
            logger.warning("CFG parallel is NOT working correctly when true_cfg_scale <= 1.")
            return False

        if not has_neg_prompt:
            logger.warning(
                "CFG parallel is NOT working correctly when there is no negative prompt or negative prompt embeddings."
            )
            return False
        return True

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] | None = None,
        prompt_2: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        negative_prompt_2: str | list[str] | None = None,
        true_cfg_scale: float = 1.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 28,
        sigmas: list[float] | None = None,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.FloatTensor | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        pooled_prompt_embeds: torch.FloatTensor | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
        negative_pooled_prompt_embeds: torch.FloatTensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in req.prompts] or prompt
        if all(isinstance(p, str) or p.get("negative_prompt") is None for p in req.prompts):
            negative_prompt = None
        elif req.prompts:
            negative_prompt = ["" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in req.prompts]

        height = req.sampling_params.height or self.default_sample_size * self.vae_scale_factor
        width = req.sampling_params.width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps
        sigmas = req.sampling_params.sigmas or sigmas
        guidance_scale = (
            req.sampling_params.guidance_scale if req.sampling_params.guidance_scale is not None else guidance_scale
        )
        generator = req.sampling_params.generator or generator
        true_cfg_scale = req.sampling_params.true_cfg_scale or true_cfg_scale
        num_images_per_prompt = (
            req.sampling_params.num_outputs_per_prompt
            if req.sampling_params.num_outputs_per_prompt > 0
            else num_images_per_prompt
        )

        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        if negative_prompt is not None:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )
        else:
            negative_prompt_embeds = None
            negative_pooled_prompt_embeds = None
            negative_text_ids = None

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        image_seq_len = latents.shape[1]
        timesteps, num_inference_steps = self.prepare_timesteps(num_inference_steps, sigmas, image_seq_len)
        self._num_timesteps = len(timesteps)

        do_true_cfg = self.check_cfg_parallel_validity(true_cfg_scale, negative_prompt is not None)

        if self.transformer.config.guidance_embeds:
            guidance = torch.tensor([guidance_scale], device=device)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        latents = self.diffuse(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            latents=latents,
            latent_image_ids=latent_image_ids,
            text_ids=text_ids,
            negative_text_ids=negative_text_ids,
            timesteps=timesteps,
            do_true_cfg=do_true_cfg,
            guidance=guidance,
            true_cfg_scale=true_cfg_scale,
            cfg_normalize=False,
        )

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            decoded = self.vae.decode(latents, return_dict=False)
            image = decoded[0] if isinstance(decoded, (list, tuple)) else decoded
            image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
            image = image_processor.postprocess(image, output_type=output_type)

        if not return_dict:
            return (image,)

        return DiffusionOutput(output=image)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_params = set()
        for name, loaded_weight in weights:
            if name.startswith("transformer."):
                param_name = name.replace("transformer.", "")
                for param in self.transformer.parameters():
                    if param_name in param.__dict__.get("_name", ""):
                        param.data.copy_(loaded_weight)
                        loaded_params.add(name)
                        break
        return loaded_params
