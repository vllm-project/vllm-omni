# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
GLM-TTS Pipeline for vLLM-Omni.

This module provides text-to-speech generation using the GLM-TTS model,
integrated with the vLLM-Omni diffusion framework.

GLM-TTS uses a two-stage architecture:
1. LLM (Llama-based): Converts text to speech tokens
2. Flow Matching: Converts speech tokens to mel-spectrograms
3. Vocoder: Converts mel-spectrograms to waveforms
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import torch
from diffusers.utils.torch_utils import randn_tensor
from torch import nn
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.glm_tts.glm_tts_dit import GLMTTSDiTModel
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)


def get_glm_tts_post_process_func(
    od_config: OmniDiffusionConfig,
):
    """
    Create post-processing function for GLM-TTS output.

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


class GLMTTSPipeline(nn.Module):
    """
    Pipeline for text-to-speech generation using GLM-TTS.

    This pipeline generates audio from text prompts using the GLM-TTS model,
    integrated with vLLM-Omni's diffusion framework.

    The pipeline consists of:
    1. LLM: Generates speech tokens from text (loaded externally or provided)
    2. Flow Model (DiT): Converts speech tokens to mel-spectrograms
    3. Vocoder: Converts mel-spectrograms to waveforms

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

        # Set up weights sources for the flow model (transformer)
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="flow",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
        ]

        # Initialize our custom transformer (weights loaded via load_weights)
        self.transformer = GLMTTSDiTModel(od_config=od_config)

        # Try to load vocoder if available
        self._load_vocoder(model, local_files_only, dtype)

        # Model parameters
        self.sample_rate = 22050  # GLM-TTS default sample rate
        self.mel_dim = 80
        self.hop_length = 256

        # Compute rotary embedding dimension
        self.rotary_embed_dim = self.transformer.config.attention_head_dim // 2

        # Cache backend (set by worker if needed)
        self._cache_backend = None

        # Properties for generation tracking
        self._guidance_scale = None
        self._num_timesteps = None
        self._current_timestep = None

    def _load_vocoder(
        self,
        model: str,
        local_files_only: bool,
        dtype: torch.dtype,
    ):
        """Load vocoder from model path if available."""
        self.vocoder = None
        self.vocoder_type = None

        # Try to load Vocos vocoder
        try:
            vocos_path = os.path.join(model, "vocos2d") if local_files_only else model
            if os.path.exists(vocos_path) or not local_files_only:
                try:
                    from vocos import Vocos

                    self.vocoder = Vocos.from_pretrained(
                        vocos_path if local_files_only else model,
                        subfolder="vocos2d" if not local_files_only else None,
                    )
                    self.vocoder = self.vocoder.to(self.device)
                    self.vocoder_type = "vocos"
                    logger.info("Loaded Vocos vocoder")
                except Exception as e:
                    logger.debug(f"Could not load Vocos vocoder: {e}")
        except Exception as e:
            logger.debug(f"Vocoder loading failed: {e}")

        if self.vocoder is None:
            logger.warning("No vocoder loaded. Output will be mel-spectrograms. Install vocos for waveform generation.")

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
        speech_tokens: torch.Tensor | None,
        audio_duration_s: float,
        from_stage_processor: bool = False,
    ):
        """Validate input parameters."""
        if audio_duration_s <= 0:
            raise ValueError(f"`audio_duration_s` must be positive, got {audio_duration_s}")

        # When receiving from stage processor, speech_tokens will be provided
        if not from_stage_processor and prompt is None and speech_tokens is None:
            raise ValueError("Provide either `prompt` or `speech_tokens` in extra params. Cannot leave both undefined.")

    def encode_text_to_tokens(
        self,
        prompt: str | list[str],
        batch_size: int,
        audio_duration_s: float,
    ) -> torch.Tensor:
        """
        Convert text to speech tokens.

        Note: This is a placeholder. In production, this should use the
        GLM-TTS LLM to generate speech tokens from text.

        Args:
            prompt: Input text to synthesize
            batch_size: Batch size
            audio_duration_s: Target audio duration

        Returns:
            Speech token indices [B, T_tokens]
        """
        # Estimate token length based on duration
        # GLM-TTS uses ~12.5 tokens per second
        tokens_per_second = 12.5
        token_length = int(audio_duration_s * tokens_per_second)

        logger.warning(
            "LLM not loaded. Using placeholder speech tokens. "
            "For proper synthesis, provide speech_tokens in request.extra."
        )

        return torch.randint(
            0,
            self.transformer.config.speech_token_vocab_size,
            (batch_size, token_length),
            device=self.device,
            dtype=torch.long,
        )

    def get_speaker_embedding(
        self,
        batch_size: int,
        speaker_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Get or generate speaker embedding.

        Args:
            batch_size: Batch size
            speaker_embedding: Pre-computed speaker embedding

        Returns:
            Speaker embedding [B, speaker_dim]
        """
        speaker_dim = self.transformer.config.speaker_embed_dim

        if speaker_embedding is not None:
            return speaker_embedding.to(self.device, dtype=self.transformer.dtype)

        # Return random speaker embedding as placeholder
        logger.warning(
            "No speaker embedding provided. Using random embedding. "
            "For voice cloning, provide speaker_embedding in request.extra."
        )
        return torch.randn(
            batch_size,
            speaker_dim,
            device=self.device,
            dtype=self.transformer.dtype,
        )

    def prepare_latents(
        self,
        batch_size: int,
        mel_length: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Prepare initial latent noise for flow matching."""
        shape = (batch_size, mel_length, self.mel_dim)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        return latents

    def _get_rotary_embedding(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute rotary embeddings for the given sequence length."""
        dim = self.rotary_embed_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
        t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)
        # Double the dimension by concatenating
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
        return cos, sin

    def decode_mel_to_audio(
        self,
        mel: torch.Tensor,
        output_type: str = "np",
    ) -> torch.Tensor:
        """
        Decode mel-spectrogram to audio waveform.

        Args:
            mel: Mel-spectrogram [B, T, mel_dim] or [B, mel_dim, T]
            output_type: Output type

        Returns:
            Audio waveform
        """
        if output_type == "latent":
            return mel

        # Ensure mel is in [B, mel_dim, T] format for vocoder
        if mel.shape[-1] == self.mel_dim:
            mel = mel.transpose(1, 2)

        if self.vocoder is not None:
            mel_for_vocoder = mel.to(dtype=torch.float32)
            if self.vocoder_type == "vocos":
                audio = self.vocoder.decode(mel_for_vocoder)
                if audio.dim() == 2:
                    audio = audio.unsqueeze(1)
            else:
                audio = mel_for_vocoder
        else:
            # Return mel if no vocoder
            audio = mel

        return audio

    @torch.no_grad()
    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        audio_duration_s: float = 10.0,
        num_inference_steps: int = 32,
        guidance_scale: float = 1.0,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
        output_type: str = "np",
    ) -> DiffusionOutput:
        """
        Generate audio from text prompt or speech tokens.

        Args:
            req: OmniDiffusionRequest containing generation parameters
            prompt: Text prompt for audio generation
            negative_prompt: Negative prompt (not used in flow matching)
            audio_duration_s: Target audio duration in seconds
            num_inference_steps: Number of flow matching steps
            guidance_scale: Classifier-free guidance scale
            generator: Random generator for reproducibility
            latents: Pre-generated latents
            output_type: Output format ("np", "pt", or "latent")

        Returns:
            DiffusionOutput containing generated audio
        """
        # Extract from request
        prompt = req.prompt if req.prompt is not None else prompt
        num_inference_steps = req.num_inference_steps or num_inference_steps
        if req.guidance_scale_provided:
            guidance_scale = req.guidance_scale

        if generator is None:
            generator = req.generator
        if generator is None and req.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(req.seed)

        # Check if input comes from stage processor (two-stage pipeline)
        # Stage processor provides speech_tokens in additional_information
        additional_info = req.extra.get("additional_information", {})
        from_stage_processor = bool(additional_info)

        # Get speech tokens - priority: additional_information > extra > generate
        speech_tokens = additional_info.get("speech_tokens", None)
        if speech_tokens is None:
            speech_tokens = req.extra.get("speech_tokens", None)

        # Get speaker embedding - priority: additional_information > extra > generate
        speaker_embedding = additional_info.get("speaker_embedding", None)
        if speaker_embedding is None:
            speaker_embedding = req.extra.get("speaker_embedding", None)

        audio_duration_s = req.extra.get("audio_duration_s", audio_duration_s)

        # Validate inputs
        self.check_inputs(prompt, speech_tokens, audio_duration_s, from_stage_processor)

        # Determine batch size
        if speech_tokens is not None and isinstance(speech_tokens, torch.Tensor):
            batch_size = speech_tokens.shape[0]
        elif prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = 1

        device = self.device
        self._guidance_scale = guidance_scale

        # Generate or use provided speech tokens
        if speech_tokens is None:
            speech_tokens = self.encode_text_to_tokens(prompt, batch_size, audio_duration_s)
        else:
            if from_stage_processor:
                logger.info("Using speech tokens from LLM stage (two-stage pipeline)")
            speech_tokens = speech_tokens.to(device)
            # Ensure speech_tokens has batch dimension [B, T]
            if speech_tokens.dim() == 1:
                speech_tokens = speech_tokens.unsqueeze(0)

        # Get speaker embedding
        speaker_embedding = self.get_speaker_embedding(batch_size, speaker_embedding)

        # Calculate mel-spectrogram length
        mel_length = int(audio_duration_s * self.sample_rate / self.hop_length)

        # Prepare timesteps (flow matching: 0 -> 1)
        timesteps = torch.linspace(0, 1, num_inference_steps + 1, device=device, dtype=self.transformer.dtype)
        self._num_timesteps = num_inference_steps

        # Prepare latents (initial noise)
        latents = self.prepare_latents(
            batch_size,
            mel_length,
            self.transformer.dtype,
            device,
            generator,
            latents,
        )

        # Compute rotary embeddings
        rotary_embedding = self._get_rotary_embedding(mel_length, device, latents.dtype)

        # Flow matching loop (Euler integration)
        for i in range(num_inference_steps):
            t = timesteps[i]
            dt = timesteps[i + 1] - timesteps[i]
            self._current_timestep = t

            # Expand timestep for batch
            t_batch = t.expand(batch_size)

            # Predict velocity
            velocity = self.transformer(
                latents,
                t_batch,
                speech_tokens=speech_tokens,
                speaker_embedding=speaker_embedding,
                rotary_embedding=rotary_embedding,
                return_dict=False,
            )[0]

            # Euler step
            latents = latents + velocity * dt

        self._current_timestep = None

        # Decode to audio
        audio = self.decode_mel_to_audio(latents, output_type)

        return DiffusionOutput(output=audio)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights using AutoWeightsLoader for vLLM integration."""
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
