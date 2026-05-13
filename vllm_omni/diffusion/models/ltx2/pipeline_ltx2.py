# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
import inspect
import json
import os
from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any, ClassVar

import numpy as np
import torch
from diffusers import AutoencoderKLLTX2Audio, AutoencoderKLLTX2Video, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.ltx2 import LTX2TextConnectors
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg, retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from torch import nn
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.parallel_state import (
    get_classifier_free_guidance_world_size,
)
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.dmd2 import DMD2PipelineMixin
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.lora.request import LoRARequest

from .ltx2_transformer import LTX2VideoTransformer3DModel
from .pipeline_ltx2_latent_upsample import LTX2LatentUpsamplePipeline

logger = init_logger(__name__)


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


def create_transformer_from_config(config: dict) -> LTX2VideoTransformer3DModel:
    """Create LTX2VideoTransformer3DModel from config dict."""
    if not config:
        return LTX2VideoTransformer3DModel()

    signature = inspect.signature(LTX2VideoTransformer3DModel.__init__)
    allowed_keys = set(signature.parameters.keys())
    kwargs = {k: v for k, v in config.items() if k in allowed_keys}
    return LTX2VideoTransformer3DModel(**kwargs)


def get_ltx2_post_process_func(
    od_config: OmniDiffusionConfig,
):
    def post_process_func(output: tuple[torch.Tensor, torch.Tensor] | torch.Tensor):
        if isinstance(output, tuple) and len(output) == 2:
            video, audio = output
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu()
            return {"video": video, "audio": audio}
        return output

    return post_process_func


def _unwrap_request_tensor(value: Any) -> Any:
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _get_prompt_field(prompt: Any, key: str) -> Any:
    if isinstance(prompt, str):
        return None
    value = prompt.get(key)
    if value is None:
        additional = prompt.get("additional_information")
        if isinstance(additional, dict):
            value = additional.get(key)
    return _unwrap_request_tensor(value)


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


class _VideoAudioScheduler:
    """Composite scheduler dispatching to video and audio schedulers."""

    def __init__(self, video_scheduler, audio_scheduler):
        self.video_scheduler = video_scheduler
        self.audio_scheduler = audio_scheduler

    def step(self, noise_pred, t, latents, return_dict=False, generator=None):
        video_out = self.video_scheduler.step(
            noise_pred[0],
            t[0],
            latents[0],
            return_dict=False,
            generator=generator,
        )[0]
        audio_out = self.audio_scheduler.step(
            noise_pred[1],
            t[1],
            latents[1],
            return_dict=False,
            generator=generator,
        )[0]
        return ((video_out, audio_out),)


class LTX2Pipeline(nn.Module, CFGParallelMixin, ProgressBarMixin):
    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config

        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        model = od_config.model
        local_files_only = os.path.exists(model)

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
        ]

        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            subfolder="tokenizer",
            local_files_only=local_files_only,
        )
        # prefer mmap loading as default device is cuda, and the output of text encoder
        # could be deterministic.
        with torch.device("cpu"):
            self.text_encoder = Gemma3ForConditionalGeneration.from_pretrained(
                model,
                subfolder="text_encoder",
                torch_dtype=dtype,
                local_files_only=local_files_only,
            ).to(self.device)
        self.connectors = LTX2TextConnectors.from_pretrained(
            model,
            subfolder="connectors",
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ).to(self.device)

        self.vae = AutoencoderKLLTX2Video.from_pretrained(
            model,
            subfolder="vae",
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ).to(self.device)
        self.audio_vae = AutoencoderKLLTX2Audio.from_pretrained(
            model,
            subfolder="audio_vae",
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ).to(self.device)
        self.vocoder = LTX2Vocoder.from_pretrained(
            model,
            subfolder="vocoder",
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ).to(self.device)

        transformer_config = load_transformer_config(model, "transformer", local_files_only)
        self.transformer = create_transformer_from_config(transformer_config)

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model, subfolder="scheduler", local_files_only=local_files_only
        )

        self.vae_spatial_compression_ratio = (
            self.vae.spatial_compression_ratio if getattr(self, "vae", None) is not None else 32
        )
        self.vae_temporal_compression_ratio = (
            self.vae.temporal_compression_ratio if getattr(self, "vae", None) is not None else 8
        )
        self.audio_vae_mel_compression_ratio = (
            self.audio_vae.mel_compression_ratio if getattr(self, "audio_vae", None) is not None else 4
        )
        self.audio_vae_temporal_compression_ratio = (
            self.audio_vae.temporal_compression_ratio if getattr(self, "audio_vae", None) is not None else 4
        )
        self.transformer_spatial_patch_size = (
            self.transformer.config.patch_size if getattr(self, "transformer", None) is not None else 1
        )
        self.transformer_temporal_patch_size = (
            self.transformer.config.patch_size_t if getattr(self, "transformer", None) is not None else 1
        )

        self.audio_sampling_rate = (
            self.audio_vae.config.sample_rate if getattr(self, "audio_vae", None) is not None else 16000
        )
        self.audio_hop_length = (
            self.audio_vae.config.mel_hop_length if getattr(self, "audio_vae", None) is not None else 160
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)
        tokenizer_max_length = 1024
        if getattr(self, "tokenizer", None) is not None:
            tokenizer_max_length = self.tokenizer.model_max_length
            if tokenizer_max_length is None or tokenizer_max_length > 100000:
                encoder_config = getattr(self.text_encoder, "config", None)
                config_max_len = getattr(encoder_config, "max_position_embeddings", None)
                if config_max_len is None:
                    config_max_len = getattr(encoder_config, "max_seq_len", None)
                tokenizer_max_length = config_max_len or 1024
        self.tokenizer_max_length = int(tokenizer_max_length)

        self._guidance_scale = None
        self._guidance_rescale = None
        self._attention_kwargs = None
        self._interrupt = False
        self._num_timesteps = None
        self._current_timestep = None

    @staticmethod
    def _pack_text_embeds(
        text_hidden_states: torch.Tensor,
        sequence_lengths: torch.Tensor,
        device: str | torch.device,
        padding_side: str = "left",
        scale_factor: int = 8,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_dim, num_layers = text_hidden_states.shape
        original_dtype = text_hidden_states.dtype

        token_indices = torch.arange(seq_len, device=device).unsqueeze(0)
        if padding_side == "right":
            mask = token_indices < sequence_lengths[:, None]
        elif padding_side == "left":
            start_indices = seq_len - sequence_lengths[:, None]
            mask = token_indices >= start_indices
        else:
            raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")
        mask = mask[:, :, None, None]

        masked_text_hidden_states = text_hidden_states.masked_fill(~mask, 0.0)
        num_valid_positions = (sequence_lengths * hidden_dim).view(batch_size, 1, 1, 1)
        masked_mean = masked_text_hidden_states.sum(dim=(1, 2), keepdim=True) / (num_valid_positions + eps)

        x_min = text_hidden_states.masked_fill(~mask, float("inf")).amin(dim=(1, 2), keepdim=True)
        x_max = text_hidden_states.masked_fill(~mask, float("-inf")).amax(dim=(1, 2), keepdim=True)

        normalized_hidden_states = (text_hidden_states - masked_mean) / (x_max - x_min + eps)
        normalized_hidden_states = normalized_hidden_states * scale_factor

        normalized_hidden_states = normalized_hidden_states.flatten(2)
        mask_flat = mask.squeeze(-1).expand(-1, -1, hidden_dim * num_layers)
        normalized_hidden_states = normalized_hidden_states.masked_fill(~mask_flat, 0.0)
        normalized_hidden_states = normalized_hidden_states.to(dtype=original_dtype)
        return normalized_hidden_states

    def _get_gemma_prompt_embeds(
        self,
        prompt: str | list[str],
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 1024,
        scale_factor: int = 8,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self.device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if getattr(self, "tokenizer", None) is not None:
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        prompt = [p.strip() for p in prompt]
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        text_input_ids = text_input_ids.to(device)
        prompt_attention_mask = prompt_attention_mask.to(device)

        text_encoder_outputs = self.text_encoder(
            input_ids=text_input_ids, attention_mask=prompt_attention_mask, output_hidden_states=True
        )
        text_encoder_hidden_states = text_encoder_outputs.hidden_states
        text_encoder_hidden_states = torch.stack(text_encoder_hidden_states, dim=-1)
        sequence_lengths = prompt_attention_mask.sum(dim=-1)

        prompt_embeds = self._pack_text_embeds(
            text_encoder_hidden_states,
            sequence_lengths,
            device=device,
            padding_side=self.tokenizer.padding_side,
            scale_factor=scale_factor,
        )
        prompt_embeds = prompt_embeds.to(dtype=dtype)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_videos_per_prompt, 1)

        return prompt_embeds, prompt_attention_mask

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        max_sequence_length: int = 1024,
        scale_factor: int = 8,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self.device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_gemma_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                scale_factor=scale_factor,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds, negative_prompt_attention_mask = self._get_gemma_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                scale_factor=scale_factor,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    def check_inputs(
        self,
        prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
    ):
        if height % 32 != 0 or width % 32 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 32 but are {height} and {width}.")

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

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")

        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise ValueError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            if prompt_attention_mask.shape != negative_prompt_attention_mask.shape:
                raise ValueError(
                    "`prompt_attention_mask` and `negative_prompt_attention_mask` must have the same shape when "
                    "passed directly, but got: `prompt_attention_mask` "
                    f"{prompt_attention_mask.shape} != `negative_prompt_attention_mask` "
                    f"{negative_prompt_attention_mask.shape}."
                )

    @staticmethod
    def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = latents.shape
        post_patch_num_frames = num_frames // patch_size_t
        post_patch_height = height // patch_size
        post_patch_width = width // patch_size
        latents = latents.reshape(
            batch_size,
            -1,
            post_patch_num_frames,
            patch_size_t,
            post_patch_height,
            patch_size,
            post_patch_width,
            patch_size,
        )
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        return latents

    @staticmethod
    def _unpack_latents(
        latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
    ) -> torch.Tensor:
        batch_size = latents.size(0)
        latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
        latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return latents

    @staticmethod
    def _normalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        # Normalize latents across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) * scaling_factor / latents_std
        return latents

    @staticmethod
    def _normalize_audio_latents(latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor):
        latents_mean = latents_mean.to(latents.device, latents.dtype)
        latents_std = latents_std.to(latents.device, latents.dtype)
        return (latents - latents_mean) / latents_std

    @staticmethod
    def _denormalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std / scaling_factor + latents_mean
        return latents

    @staticmethod
    def _denormalize_audio_latents(latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor):
        latents_mean = latents_mean.to(latents.device, latents.dtype)
        latents_std = latents_std.to(latents.device, latents.dtype)
        return (latents * latents_std) + latents_mean

    @staticmethod
    def _create_noised_state(
        latents: torch.Tensor, noise_scale: float | torch.Tensor, generator: torch.Generator | None = None
    ):
        noise = randn_tensor(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
        noised_latents = noise_scale * noise + (1 - noise_scale) * latents
        return noised_latents

    @staticmethod
    def _pack_audio_latents(
        latents: torch.Tensor, patch_size: int | None = None, patch_size_t: int | None = None
    ) -> torch.Tensor:
        if patch_size is not None and patch_size_t is not None:
            batch_size, num_channels, latent_length, latent_mel_bins = latents.shape
            post_patch_latent_length = latent_length / patch_size_t
            post_patch_mel_bins = latent_mel_bins / patch_size
            latents = latents.reshape(
                batch_size, -1, post_patch_latent_length, patch_size_t, post_patch_mel_bins, patch_size
            )
            latents = latents.permute(0, 2, 4, 1, 3, 5).flatten(3, 5).flatten(1, 2)
        else:
            latents = latents.transpose(1, 2).flatten(2, 3)
        return latents

    @staticmethod
    def _unpack_audio_latents(
        latents: torch.Tensor,
        latent_length: int,
        num_mel_bins: int,
        patch_size: int | None = None,
        patch_size_t: int | None = None,
    ) -> torch.Tensor:
        if patch_size is not None and patch_size_t is not None:
            batch_size = latents.size(0)
            latents = latents.reshape(batch_size, latent_length, num_mel_bins, -1, patch_size_t, patch_size)
            latents = latents.permute(0, 3, 1, 4, 2, 5).flatten(4, 5).flatten(2, 3)
        else:
            latents = latents.unflatten(2, (-1, num_mel_bins)).transpose(1, 2)
        return latents

    @staticmethod
    def _unpad_audio_latents(latents: torch.Tensor, num_frames: int) -> torch.Tensor:
        return latents[:, :num_frames]

    def prepare_latents(
        self,
        batch_size: int = 1,
        num_channels_latents: int = 128,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        noise_scale: float = 0.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            if latents.ndim == 5:
                latents = self._normalize_latents(
                    latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
                )
                # latents are of shape [B, C, F, H, W], need to be packed
                latents = self._pack_latents(
                    latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size
                )
            if latents.ndim != 3:
                raise ValueError(
                    f"Provided `latents` tensor has shape {latents.shape}, but the expected shape is [batch_size, num_seq, num_features]."  # noqa
                )
            latents = self._create_noised_state(latents, noise_scale, generator)
            return latents.to(device=device, dtype=dtype)

        height = height // self.vae_spatial_compression_ratio
        width = width // self.vae_spatial_compression_ratio
        num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1

        shape = (batch_size, num_channels_latents, num_frames, height, width)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, self.transformer_spatial_patch_size, self.transformer_temporal_patch_size)

        return latents

    def prepare_audio_latents(
        self,
        batch_size: int = 1,
        num_channels_latents: int = 8,
        audio_latent_length: int = 1,  # 1 is just a dummy value
        num_mel_bins: int = 64,
        noise_scale: float = 0.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int, int]:
        original_latent_length = audio_latent_length
        padded_latent_length = original_latent_length

        latent_mel_bins = num_mel_bins // self.audio_vae_mel_compression_ratio

        sp_size = getattr(self.od_config.parallel_config, "sequence_parallel_size", 1)
        if sp_size > 1:
            padded_latent_length += (sp_size - (original_latent_length % sp_size)) % sp_size

        if latents is not None:
            if latents.ndim == 4:
                # latents are of shape [B, C, L, M], need to be packed
                latents = self._pack_audio_latents(latents)
            if latents.ndim != 3:
                raise ValueError(
                    f"Provided `latents` tensor has shape {latents.shape}, but the expected shape is "
                    "[batch_size, num_seq, num_features] or [batch_size, num_channels, audio_length, mel_bins]."
                )
            latents = self._normalize_audio_latents(latents, self.audio_vae.latents_mean, self.audio_vae.latents_std)
            latents = self._create_noised_state(latents, noise_scale, generator)

            if latents.shape[1] not in {original_latent_length, padded_latent_length}:
                raise ValueError(
                    "Provided `audio_latents` has incompatible audio frame count "
                    f"{latents.shape[1]}; expected {original_latent_length} or {padded_latent_length}."
                )

            if latents.shape[1] == original_latent_length and padded_latent_length > original_latent_length:
                padding = torch.zeros(
                    latents.shape[0],
                    padded_latent_length - original_latent_length,
                    latents.shape[2],
                    dtype=latents.dtype,
                    device=latents.device,
                )
                latents = torch.cat([latents, padding], dim=1)

            return latents.to(device=device, dtype=dtype), original_latent_length, padded_latent_length

        shape = (batch_size, num_channels_latents, padded_latent_length, latent_mel_bins)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_audio_latents(latents)
        return latents, original_latent_length, padded_latent_length

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    def _transformer_cache_context(self, context_name: str):
        cache_context = getattr(self.transformer, "cache_context", None)
        if callable(cache_context):
            return cache_context(context_name)
        return nullcontext()

    def predict_noise(self, **kwargs):
        with self._transformer_cache_context("cond_uncond"):
            noise_pred_video, noise_pred_audio = self.transformer(**kwargs)
        return noise_pred_video.float(), noise_pred_audio.float()

    def combine_cfg_noise(self, positive_noise_pred, negative_noise_pred, true_cfg_scale, cfg_normalize=False):
        """Per-element CFG combine with guidance_rescale support."""
        (video_pos, audio_pos) = positive_noise_pred
        (video_neg, audio_neg) = negative_noise_pred
        video_combined = super().combine_cfg_noise(video_pos, video_neg, true_cfg_scale, cfg_normalize)
        audio_combined = super().combine_cfg_noise(audio_pos, audio_neg, true_cfg_scale, cfg_normalize)
        if self._guidance_rescale and self._guidance_rescale > 0:
            video_combined = rescale_noise_cfg(video_combined, video_pos, guidance_rescale=self._guidance_rescale)
            audio_combined = rescale_noise_cfg(audio_combined, audio_pos, guidance_rescale=self._guidance_rescale)
        return (video_combined, audio_combined)

    def _synchronize_cfg_parallel_step_output(
        self,
        latents: tuple[torch.Tensor, torch.Tensor],
        do_true_cfg: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (do_true_cfg and get_classifier_free_guidance_world_size() > 1):
            return latents

        # Without this sync, CUDA async execution causes non-deterministic
        # numerical drift across denoising steps in CFG parallel mode,
        # producing different video outputs across runs.
        latents = tuple(tensor.contiguous() for tensor in latents)
        device = next((tensor.device for tensor in latents if tensor.is_cuda), None)
        if device is not None:
            torch.cuda.current_stream(device).synchronize()
        return latents

    @torch.no_grad()
    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        frame_rate: float | None = None,
        num_inference_steps: int | None = None,
        sigmas: list[float] | None = None,
        timesteps: list[int] | None = None,
        guidance_scale: float = 4.0,
        guidance_rescale: float = 0.0,
        noise_scale: float = 0.0,
        num_videos_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        decode_timestep: float | list[float] = 0.0,
        decode_noise_scale: float | list[float] | None = None,
        output_type: str = "np",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        max_sequence_length: int | None = None,
    ) -> DiffusionOutput:
        # Extract prompt/negative_prompt from request.
        # Input format: req.prompts is a list of str or dict with "prompt"/"negative_prompt" keys.
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in req.prompts] or prompt
        if all(isinstance(p, str) or p.get("negative_prompt") is None for p in req.prompts):
            negative_prompt = None
        elif req.prompts:
            negative_prompt = ["" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in req.prompts]

        height = req.sampling_params.height or height or 512
        width = req.sampling_params.width or width or 768
        num_frames = req.sampling_params.num_frames or num_frames or 121
        frame_rate = req.sampling_params.resolved_frame_rate or frame_rate or 24.0
        num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps or 40
        if timesteps is None:
            num_inference_steps = max(int(num_inference_steps), 2)
        elif len(timesteps) < 2:
            raise ValueError("`timesteps` must contain at least 2 values for FlowMatchEulerDiscreteScheduler.")
        num_videos_per_prompt = (
            req.sampling_params.num_outputs_per_prompt
            if req.sampling_params.num_outputs_per_prompt > 0
            else num_videos_per_prompt or 1
        )
        max_sequence_length = (
            req.sampling_params.max_sequence_length or max_sequence_length or self.tokenizer_max_length
        )

        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale
        if req.sampling_params.guidance_rescale is not None:
            guidance_rescale = req.sampling_params.guidance_rescale

        if generator is None:
            generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(req.sampling_params.seed)

        latents = req.sampling_params.latents if req.sampling_params.latents is not None else latents
        audio_latents = (
            req.sampling_params.audio_latents
            if req.sampling_params.audio_latents is not None
            else req.sampling_params.extra_args.get("audio_latents", audio_latents)
        )

        # Override with pre-computed embeddings if provided in request.
        req_prompt_embeds = [_get_prompt_field(p, "prompt_embeds") for p in req.prompts]
        if any(p is not None for p in req_prompt_embeds):
            prompt_embeds = torch.stack(req_prompt_embeds)  # type: ignore[arg-type]

        req_negative_prompt_embeds = [_get_prompt_field(p, "negative_prompt_embeds") for p in req.prompts]
        if any(p is not None for p in req_negative_prompt_embeds):
            negative_prompt_embeds = torch.stack(req_negative_prompt_embeds)  # type: ignore[arg-type]

        req_prompt_attention_masks = [
            _get_prompt_field(p, "prompt_attention_mask") or _get_prompt_field(p, "attention_mask") for p in req.prompts
        ]
        if any(m is not None for m in req_prompt_attention_masks):
            prompt_attention_mask = torch.stack(req_prompt_attention_masks)  # type: ignore[arg-type]

        req_negative_attention_masks = [
            _get_prompt_field(p, "negative_prompt_attention_mask") or _get_prompt_field(p, "negative_attention_mask")
            for p in req.prompts
        ]
        if any(m is not None for m in req_negative_attention_masks):
            negative_prompt_attention_mask = torch.stack(req_negative_attention_masks)  # type: ignore[arg-type]

        if req.sampling_params.decode_timestep is not None:
            decode_timestep = req.sampling_params.decode_timestep
        if req.sampling_params.decode_noise_scale is not None:
            decode_noise_scale = req.sampling_params.decode_noise_scale
        if req.sampling_params.output_type is not None:
            output_type = req.sampling_params.output_type

        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        self._current_timestep = None

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.device

        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        # Compute positive prompt connectors
        additive_attention_mask = (1 - prompt_attention_mask.to(prompt_embeds.dtype)) * -1000000.0
        connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask = self.connectors(
            prompt_embeds, additive_attention_mask, additive_mask=True
        )

        # Compute negative prompt connectors when CFG is enabled
        negative_connector_prompt_embeds = None
        negative_connector_audio_prompt_embeds = None
        negative_connector_attention_mask = None
        if self.do_classifier_free_guidance:
            negative_additive_attention_mask = (
                1 - negative_prompt_attention_mask.to(negative_prompt_embeds.dtype)
            ) * -1000000.0
            (
                negative_connector_prompt_embeds,
                negative_connector_audio_prompt_embeds,
                negative_connector_attention_mask,
            ) = self.connectors(
                negative_prompt_embeds,
                negative_additive_attention_mask,
                additive_mask=True,
            )

        latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        if latents is not None:
            if latents.ndim == 5:
                logger.info(
                    "Got latents of shape [batch_size, latent_dim, latent_frames, latent_height, latent_width], `latent_num_frames`, `latent_height`, `latent_width` will be inferred."  # noqa
                )
                _, _, latent_num_frames, latent_height, latent_width = latents.shape  # [B, C, F, H, W]
            elif latents.ndim == 3:
                logger.warning(
                    f"You have supplied packed `latents` of shape {latents.shape}, so the latent dims cannot be"
                    f" inferred. Make sure the supplied `height`, `width`, and `num_frames` are correct."
                )
            else:
                raise ValueError(
                    f"Provided `latents` tensor has shape {latents.shape}, but the expected shape is either [batch_size, seq_len, num_features] or [batch_size, latent_dim, latent_frames, latent_height, latent_width]."  # noqa
                )
        video_sequence_length = latent_num_frames * latent_height * latent_width

        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            noise_scale,
            torch.float32,
            device,
            generator,
            latents,
        )

        duration_s = num_frames / frame_rate
        audio_latents_per_second = (
            self.audio_sampling_rate / self.audio_hop_length / float(self.audio_vae_temporal_compression_ratio)
        )
        audio_num_frames = round(duration_s * audio_latents_per_second)
        if audio_latents is not None:
            if audio_latents.ndim == 4:
                logger.info(
                    "Got audio_latents of shape [batch_size, num_channels, audio_length, mel_bins], `audio_num_frames` will be inferred."  # noqa
                )
                _, _, audio_num_frames, _ = audio_latents.shape  # [B, C, L, M]
            elif audio_latents.ndim == 3:
                logger.warning(
                    f"You have supplied packed `audio_latents` of shape {audio_latents.shape}, so the latent dims"
                    f" cannot be inferred. Make sure the supplied `num_frames` and `frame_rate` are correct."
                )
            else:
                raise ValueError(
                    f"Provided `audio_latents` tensor has shape {audio_latents.shape}, but the expected shape is either [batch_size, seq_len, num_features] or [batch_size, num_channels, audio_length, mel_bins]."  # noqa
                )

        num_mel_bins = self.audio_vae.config.mel_bins if getattr(self, "audio_vae", None) is not None else 64
        latent_mel_bins = num_mel_bins // self.audio_vae_mel_compression_ratio

        num_channels_latents_audio = (
            self.audio_vae.config.latent_channels if getattr(self, "audio_vae", None) is not None else 8
        )
        audio_latents, original_audio_num_frames, padded_audio_num_frames = self.prepare_audio_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents=num_channels_latents_audio,
            audio_latent_length=audio_num_frames,
            num_mel_bins=num_mel_bins,
            noise_scale=noise_scale,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=audio_latents,
        )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        mu = calculate_shift(
            video_sequence_length,
            self.scheduler.config.get("base_image_seq_len", 1024),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.95),
            self.scheduler.config.get("max_shift", 2.05),
        )
        audio_scheduler = copy.deepcopy(self.scheduler)
        video_audio_scheduler = _VideoAudioScheduler(self.scheduler, audio_scheduler)
        _ = retrieve_timesteps(
            audio_scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas=sigmas,
            mu=mu,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas=sigmas,
            mu=mu,
        )
        self._num_timesteps = len(timesteps)

        video_coords = self.transformer.rope.prepare_video_coords(
            latents.shape[0], latent_num_frames, latent_height, latent_width, latents.device, fps=frame_rate
        )
        audio_coords = self.transformer.audio_rope.prepare_audio_coords(
            audio_latents.shape[0], padded_audio_num_frames, audio_latents.device
        )
        # No coord duplication needed: mixin handles CFG via separate forward calls,
        # not batch=2. Each forward gets batch=1 coords directly.

        with self.progress_bar(total=len(timesteps)) as pbar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                latent_model_input = latents.to(prompt_embeds.dtype)
                audio_latent_model_input = audio_latents.to(prompt_embeds.dtype)
                timestep = t.expand(latent_model_input.shape[0])
                do_true_cfg = self.do_classifier_free_guidance

                positive_kwargs = {
                    "hidden_states": latent_model_input,
                    "audio_hidden_states": audio_latent_model_input,
                    "encoder_hidden_states": connector_prompt_embeds,
                    "audio_encoder_hidden_states": connector_audio_prompt_embeds,
                    "timestep": timestep,
                    "encoder_attention_mask": connector_attention_mask,
                    "audio_encoder_attention_mask": connector_attention_mask,
                    "num_frames": latent_num_frames,
                    "height": latent_height,
                    "width": latent_width,
                    "fps": frame_rate,
                    "audio_num_frames": padded_audio_num_frames,
                    "video_coords": video_coords,
                    "audio_coords": audio_coords,
                    "attention_kwargs": attention_kwargs,
                    "return_dict": False,
                }
                negative_kwargs = (
                    {
                        **positive_kwargs,
                        "encoder_hidden_states": negative_connector_prompt_embeds,
                        "audio_encoder_hidden_states": negative_connector_audio_prompt_embeds,
                        "encoder_attention_mask": negative_connector_attention_mask,
                        "audio_encoder_attention_mask": negative_connector_attention_mask,
                    }
                    if do_true_cfg
                    else None
                )

                noise_pred_video, noise_pred_audio = self.predict_noise_maybe_with_cfg(
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=guidance_scale,
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    cfg_normalize=False,
                )

                latents, audio_latents = self.scheduler_step_maybe_with_cfg(
                    (noise_pred_video, noise_pred_audio),
                    (t, t),
                    (latents, audio_latents),
                    do_true_cfg=do_true_cfg,
                    per_request_scheduler=video_audio_scheduler,
                )
                latents, audio_latents = self._synchronize_cfg_parallel_step_output(
                    (latents, audio_latents),
                    do_true_cfg=do_true_cfg,
                )

                pbar.update()

        latents = self._unpack_latents(
            latents,
            latent_num_frames,
            latent_height,
            latent_width,
            self.transformer_spatial_patch_size,
            self.transformer_temporal_patch_size,
        )
        latents = self._denormalize_latents(
            latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
        )

        audio_latents = self._unpad_audio_latents(audio_latents, original_audio_num_frames)
        audio_latents = self._denormalize_audio_latents(
            audio_latents, self.audio_vae.latents_mean, self.audio_vae.latents_std
        )
        audio_latents = self._unpack_audio_latents(
            audio_latents,
            original_audio_num_frames,
            num_mel_bins=latent_mel_bins,
        )

        if output_type == "latent":
            video = latents
            audio = audio_latents
        else:
            latents = latents.to(prompt_embeds.dtype)

            if not self.vae.config.timestep_conditioning:
                timestep = None
            else:
                noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
                if not isinstance(decode_timestep, list):
                    decode_timestep = [decode_timestep] * batch_size
                if decode_noise_scale is None:
                    decode_noise_scale = decode_timestep
                elif not isinstance(decode_noise_scale, list):
                    decode_noise_scale = [decode_noise_scale] * batch_size

                timestep = torch.tensor(decode_timestep, device=device, dtype=latents.dtype)
                decode_noise_scale = torch.tensor(decode_noise_scale, device=device, dtype=latents.dtype)[
                    :, None, None, None, None
                ]
                latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

            latents = latents.to(self.vae.dtype)
            video = self.vae.decode(latents, timestep, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)

            audio_latents = audio_latents.to(self.audio_vae.dtype)
            generated_mel_spectrograms = self.audio_vae.decode(audio_latents, return_dict=False)[0]
            audio = self.vocoder(generated_mel_spectrograms)

        if not return_dict:
            return DiffusionOutput(output=(video, audio))

        return DiffusionOutput(output=(video, audio))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


class LTX2TwoStagesPipeline(nn.Module, SupportsComponentDiscovery):
    """LTX2TwoStagesPipeline is for two stages image to video generation"""

    _dit_modules: ClassVar[list[str]] = ["pipe.transformer"]
    _encoder_modules: ClassVar[list[str]] = ["pipe.text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["pipe.vae", "pipe.audio_vae"]

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()

        self.device = get_local_device()
        self.dtype = getattr(od_config, "dtype", torch.bfloat16)
        self.model_path = od_config.model
        self.distilled = False
        # User provided model path may contain '/' in the end and basename function
        # will not return the expected directory name, so we need to remove it by normpath
        if "distilled" in os.path.basename(os.path.normpath(self.model_path)):
            self.distilled = True
        else:
            raise NotImplementedError(f"{self.model_path} is not supported for {self.__class__.__name__}.")

        self.pipe = LTX2Pipeline(od_config=od_config, prefix=prefix)
        self.upsample_pipe = LTX2LatentUpsamplePipeline(
            vae=self.pipe.vae,
            od_config=od_config,
        )

        self.lora_manager = DiffusionLoRAManager(
            pipeline=self.pipe,
            device=self.device,
            dtype=self.dtype,
            max_cached_adapters=od_config.max_cpu_loras,
        )

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="pipe.transformer.",
                fall_back_to_pt=True,
            ),
        ]

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        frame_rate: float | None = None,
        num_inference_steps: int | None = None,
        timesteps: list[int] | None = None,
        guidance_scale: float = 4.0,
        guidance_rescale: float = 0.0,
        noise_scale: float = 0.0,
        num_videos_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        audio_latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt_attention_mask: torch.Tensor | None = None,
        decode_timestep: float | list[float] = 0.0,
        decode_noise_scale: float | list[float] | None = None,
        output_type: str = "np",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        max_sequence_length: int | None = None,
    ):
        video_latent, audio_latent = self.pipe(
            req=req,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            sigmas=DISTILLED_SIGMA_VALUES if self.distilled else None,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
            noise_scale=noise_scale,
            num_videos_per_prompt=num_videos_per_prompt,
            generator=generator,
            latents=latents,
            audio_latents=audio_latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            output_type="latent",
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
            max_sequence_length=max_sequence_length,
        ).output

        upscaled_video_latent = self.upsample_pipe(
            latents=video_latent,
            output_type="latent",
            return_dict=False,
        )[0]

        if not self.distilled:
            # Load Stage 2 distilled LoRA
            lora_path = f"{self.model_path}/ltx-2-19b-distilled-lora-384.safetensors"
            lora_request = LoRARequest(
                lora_name="stage_2_distilled",
                lora_int_id=1,
                lora_path=lora_path,
            )
            self.lora_manager.set_active_adapter(lora_request, lora_scale=1.0)

            # Change scheduler to use Stage 2 distilled sigmas as is
            new_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config,
                use_dynamic_shifting=False,
                shift_terminal=None,
            )
            self.pipe.scheduler = new_scheduler

        # We only want to change num_inference_steps here, so no need
        # to deep copy the whole request
        stage_2_req = copy.copy(req)
        stage_2_req.sampling_params = req.sampling_params.clone()
        stage_2_req.sampling_params.num_inference_steps = 3

        video, audio = self.pipe(
            req=stage_2_req,
            latents=upscaled_video_latent,
            audio_latents=audio_latent,
            prompt=prompt,
            negative_prompt=negative_prompt,
            noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
            sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
            guidance_scale=1.0,
            generator=generator,
            output_type="np",
            return_dict=False,
        ).output

        return DiffusionOutput(output=(video, audio))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)


class LTX2T2VDMD2Pipeline(DMD2PipelineMixin, LTX2Pipeline):
    """LTX-2 T2V pipeline for FastGen DMD2-distilled models."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.__init_dmd2__()
