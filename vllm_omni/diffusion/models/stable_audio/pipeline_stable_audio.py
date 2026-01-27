# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Stable Audio Open Pipeline for vLLM-Omni.

This module provides text-to-audio generation using the Stable Audio Open model
from Stability AI, integrated with the vLLM-Omni diffusion framework.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import torch
from diffusers import AutoencoderOobleck
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.pipelines.stable_audio.modeling_stable_audio import StableAudioProjectionModel
from diffusers.schedulers import CosineDPMSolverMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from transformers import T5EncoderModel, T5TokenizerFast
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportAudioOutput
from vllm_omni.diffusion.models.stable_audio.stable_audio_transformer import StableAudioDiTModel
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)


def get_stable_audio_post_process_func(
    od_config: OmniDiffusionConfig,
):
    """
    Create post-processing function for Stable Audio output.

    Converts raw audio tensor to numpy array for saving.
    """

    def post_process_func(
        audio: torch.Tensor,
        output_type: str = "np",
    ):
        if output_type == "latent":
            return audio
        if output_type == "pt":
            return audio
        # Convert to numpy
        audio_np = audio.cpu().float().numpy()
        return audio_np

    return post_process_func


class StableAudioPipeline(nn.Module, SupportAudioOutput):
    """
    Pipeline for text-to-audio generation using Stable Audio Open.

    This pipeline generates audio from text prompts using the Stable Audio Open model
    from Stability AI, integrated with vLLM-Omni's diffusion framework.

    Args:
        od_config: OmniDiffusion configuration object
        prefix: Weight prefix for loading (default: "")
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
        dtype = getattr(od_config, "dtype", torch.float16)

        model = od_config.model
        local_files_only = os.path.exists(model)

        # Set up weights sources for transformer
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
        ]

        # Load tokenizer
        self.tokenizer = T5TokenizerFast.from_pretrained(
            model,
            subfolder="tokenizer",
            local_files_only=local_files_only,
        )

        # Load text encoder
        self.text_encoder = T5EncoderModel.from_pretrained(
            model,
            subfolder="text_encoder",
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ).to(self.device)

        # Load VAE (AutoencoderOobleck for audio)
        self.vae = AutoencoderOobleck.from_pretrained(
            model,
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=local_files_only,
        ).to(self.device)

        # Load projection model (using diffusers implementation)
        self.projection_model = StableAudioProjectionModel.from_pretrained(
            model,
            subfolder="projection_model",
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ).to(self.device)

        # Initialize our custom transformer (weights loaded via load_weights)
        self.transformer = StableAudioDiTModel(od_config=od_config)

        # Load scheduler
        self.scheduler = CosineDPMSolverMultistepScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )

        # Compute rotary embedding dimension
        self.rotary_embed_dim = self.transformer.config.attention_head_dim // 2

        # Cache backend (set by worker if needed)
        self._cache_backend = None

        # Properties
        self._guidance_scale = None
        self._num_timesteps = None
        self._current_timestep = None

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

    def check_inputs(
        self,
        prompt: str | list[str] | None,
        audio_start_in_s: float,
        audio_end_in_s: float,
        negative_prompt: str | list[str] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
    ):
        """Validate input parameters."""
        if audio_end_in_s < audio_start_in_s:
            raise ValueError(
                f"`audio_end_in_s={audio_end_in_s}` must be higher than `audio_start_in_s={audio_start_in_s}`"
            )

        min_val = self.projection_model.config.min_value
        max_val = self.projection_model.config.max_value

        if audio_start_in_s < min_val or audio_start_in_s > max_val:
            raise ValueError(f"`audio_start_in_s` must be between {min_val} and {max_val}, got {audio_start_in_s}")

        if audio_end_in_s < min_val or audio_end_in_s > max_val:
            raise ValueError(f"`audio_end_in_s` must be between {min_val} and {max_val}, got {audio_end_in_s}")

        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`. Cannot leave both undefined.")

        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt` and `prompt_embeds`. Please provide only one.")

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device,
        do_classifier_free_guidance: bool,
        negative_prompt: str | list[str] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        negative_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode text prompt to embeddings."""
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # Tokenize
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask

            text_input_ids = text_input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Encode
            self.text_encoder.eval()
            prompt_embeds = self.text_encoder(
                text_input_ids,
                attention_mask=attention_mask,
            )[0]

        # Handle negative prompt for CFG
        if do_classifier_free_guidance and negative_prompt is not None:
            if isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt` has batch size {len(negative_prompt)}, but `prompt` "
                    f"has batch size {batch_size}. Please make sure they match."
                )
            else:
                uncond_tokens = negative_prompt

            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            uncond_input_ids = uncond_input.input_ids.to(device)
            negative_attention_mask = uncond_input.attention_mask.to(device)

            self.text_encoder.eval()
            negative_prompt_embeds = self.text_encoder(
                uncond_input_ids,
                attention_mask=negative_attention_mask,
            )[0]

            if negative_attention_mask is not None:
                negative_prompt_embeds = torch.where(
                    negative_attention_mask.to(torch.bool).unsqueeze(2),
                    negative_prompt_embeds,
                    0.0,
                )

        # Concatenate for CFG
        if do_classifier_free_guidance and negative_prompt_embeds is not None:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if attention_mask is not None and negative_attention_mask is None:
                negative_attention_mask = torch.ones_like(attention_mask)
            elif attention_mask is None and negative_attention_mask is not None:
                attention_mask = torch.ones_like(negative_attention_mask)

            if attention_mask is not None:
                attention_mask = torch.cat([negative_attention_mask, attention_mask])

        # Project embeddings
        prompt_embeds = self.projection_model(
            text_hidden_states=prompt_embeds,
        ).text_hidden_states

        if attention_mask is not None:
            prompt_embeds = prompt_embeds * attention_mask.unsqueeze(-1).to(prompt_embeds.dtype)

        return prompt_embeds

    def encode_duration(
        self,
        audio_start_in_s: float,
        audio_end_in_s: float,
        device: torch.device,
        do_classifier_free_guidance: bool,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode audio duration to conditioning tensors."""
        audio_start_in_s = [audio_start_in_s] if isinstance(audio_start_in_s, (int, float)) else audio_start_in_s
        audio_end_in_s = [audio_end_in_s] if isinstance(audio_end_in_s, (int, float)) else audio_end_in_s

        if len(audio_start_in_s) == 1:
            audio_start_in_s = audio_start_in_s * batch_size
        if len(audio_end_in_s) == 1:
            audio_end_in_s = audio_end_in_s * batch_size

        audio_start_in_s = torch.tensor([float(x) for x in audio_start_in_s]).to(device)
        audio_end_in_s = torch.tensor([float(x) for x in audio_end_in_s]).to(device)

        projection_output = self.projection_model(
            start_seconds=audio_start_in_s,
            end_seconds=audio_end_in_s,
        )
        seconds_start_hidden_states = projection_output.seconds_start_hidden_states
        seconds_end_hidden_states = projection_output.seconds_end_hidden_states

        if do_classifier_free_guidance:
            seconds_start_hidden_states = torch.cat([seconds_start_hidden_states, seconds_start_hidden_states], dim=0)
            seconds_end_hidden_states = torch.cat([seconds_end_hidden_states, seconds_end_hidden_states], dim=0)

        return seconds_start_hidden_states, seconds_end_hidden_states

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_vae: int,
        sample_size: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Prepare initial latent noise."""
        shape = (batch_size, num_channels_vae, sample_size)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # Scale by scheduler's noise sigma
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        audio_end_in_s: float | None = None,
        audio_start_in_s: float = 0.0,
        num_inference_steps: int = 100,
        guidance_scale: float = 7.0,
        num_waveforms_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        output_type: str = "np",
    ) -> DiffusionOutput:
        """
        Generate audio from text prompt.

        Args:
            req: OmniDiffusionRequest containing generation parameters
            prompt: Text prompt for audio generation
            negative_prompt: Negative prompt for CFG
            audio_end_in_s: Audio end time in seconds (max ~47s for stable-audio-open-1.0)
            audio_start_in_s: Audio start time in seconds
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            num_waveforms_per_prompt: Number of audio outputs per prompt
            generator: Random generator for reproducibility
            latents: Pre-generated latents
            prompt_embeds: Pre-computed prompt embeddings
            negative_prompt_embeds: Pre-computed negative prompt embeddings
            output_type: Output format ("np", "pt", or "latent")

        Returns:
            DiffusionOutput containing generated audio
        """
        # Extract from request
        # TODO: In online mode, sometimes it receives [{"negative_prompt": None}, {...}], so cannot use .get("...", "")
        # TODO: May be some data formatting operations on the API side. Hack for now.
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in req.prompts] or prompt
        if all(isinstance(p, str) or p.get("negative_prompt") is None for p in req.prompts):
            negative_prompt = None
        elif req.prompts:
            negative_prompt = ["" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in req.prompts]

        num_inference_steps = req.sampling_params.num_inference_steps or num_inference_steps
        if req.sampling_params.guidance_scale_provided:
            guidance_scale = req.sampling_params.guidance_scale

        if generator is None:
            generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(req.sampling_params.seed)

        # Get audio duration from request extra params or defaults
        audio_start_in_s = req.sampling_params.extra_args.get("audio_start_in_s", audio_start_in_s)
        audio_end_in_s = req.sampling_params.extra_args.get("audio_end_in_s", audio_end_in_s)

        # Calculate audio length
        downsample_ratio = self.vae.hop_length
        max_audio_length_in_s = self.transformer.config.sample_size * downsample_ratio / self.vae.config.sampling_rate

        if audio_end_in_s is None:
            audio_end_in_s = max_audio_length_in_s

        if audio_end_in_s - audio_start_in_s > max_audio_length_in_s:
            raise ValueError(
                f"Requested audio length ({audio_end_in_s - audio_start_in_s}s) exceeds "
                f"maximum ({max_audio_length_in_s}s)"
            )

        waveform_start = int(audio_start_in_s * self.vae.config.sampling_rate)
        waveform_end = int(audio_end_in_s * self.vae.config.sampling_rate)
        waveform_length = int(self.transformer.config.sample_size)

        # Validate inputs
        self.check_inputs(
            prompt,
            audio_start_in_s,
            audio_end_in_s,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # Determine batch size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.device
        do_classifier_free_guidance = guidance_scale > 1.0
        self._guidance_scale = guidance_scale

        # Encode prompt
        prompt_embeds = self.encode_prompt(
            prompt,
            device,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # Encode duration
        seconds_start_hidden_states, seconds_end_hidden_states = self.encode_duration(
            audio_start_in_s,
            audio_end_in_s,
            device,
            do_classifier_free_guidance and (negative_prompt is not None or negative_prompt_embeds is not None),
            batch_size,
        )

        # Create combined embeddings
        text_audio_duration_embeds = torch.cat(
            [prompt_embeds, seconds_start_hidden_states, seconds_end_hidden_states],
            dim=1,
        )
        audio_duration_embeds = torch.cat(
            [seconds_start_hidden_states, seconds_end_hidden_states],
            dim=2,
        )

        # Handle CFG without negative prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None and negative_prompt is None:
            negative_text_audio_duration_embeds = torch.zeros_like(text_audio_duration_embeds)
            text_audio_duration_embeds = torch.cat(
                [negative_text_audio_duration_embeds, text_audio_duration_embeds],
                dim=0,
            )
            audio_duration_embeds = torch.cat(
                [audio_duration_embeds, audio_duration_embeds],
                dim=0,
            )

        # Duplicate for multiple waveforms per prompt
        bs_embed, seq_len, hidden_size = text_audio_duration_embeds.shape
        text_audio_duration_embeds = text_audio_duration_embeds.repeat(1, num_waveforms_per_prompt, 1)
        text_audio_duration_embeds = text_audio_duration_embeds.view(
            bs_embed * num_waveforms_per_prompt, seq_len, hidden_size
        )

        audio_duration_embeds = audio_duration_embeds.repeat(1, num_waveforms_per_prompt, 1)
        audio_duration_embeds = audio_duration_embeds.view(
            bs_embed * num_waveforms_per_prompt, -1, audio_duration_embeds.shape[-1]
        )

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        # Prepare latents
        num_channels_vae = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_waveforms_per_prompt,
            num_channels_vae,
            waveform_length,
            text_audio_duration_embeds.dtype,
            device,
            generator,
            latents,
        )

        # Prepare rotary embeddings and move to device
        rotary_embedding = get_1d_rotary_pos_embed(
            self.rotary_embed_dim,
            latents.shape[2] + audio_duration_embeds.shape[1],
            use_real=True,
            repeat_interleave_real=False,
        )
        # Move rotary embeddings to device (returns tuple of cos, sin)
        rotary_embedding = (
            rotary_embedding[0].to(device=device, dtype=latents.dtype),
            rotary_embedding[1].to(device=device, dtype=latents.dtype),
        )

        # Denoising loop
        for t in timesteps:
            self._current_timestep = t

            # Expand latents for CFG
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Predict noise
            noise_pred = self.transformer(
                latent_model_input,
                t.unsqueeze(0),
                encoder_hidden_states=text_audio_duration_embeds,
                global_hidden_states=audio_duration_embeds,
                rotary_embedding=rotary_embedding,
                return_dict=False,
            )[0]

            # Perform CFG
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        self._current_timestep = None

        # Decode
        if output_type == "latent":
            audio = latents
        else:
            # Convert latents to VAE dtype (VAE may use float32)
            latents_for_vae = latents.to(dtype=self.vae.dtype)
            audio = self.vae.decode(latents_for_vae).sample

        # Trim to requested length
        audio = audio[:, :, waveform_start:waveform_end]

        return DiffusionOutput(output=audio)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights using AutoWeightsLoader for vLLM integration."""
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
