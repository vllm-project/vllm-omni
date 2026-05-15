# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OmniVoice Decoder (Stage 1) - Audio token to waveform conversion.

Implements the HiggsAudioV2 decode path using transformers' DacModel decoder
and a custom RVQ quantizer, compatible with transformers 4.x.

Decode path:
  audio_codes [B, 8, T]
    -> RVQ codebook lookup + project_out -> sum -> [B, 1024, T]
    -> fc2 Linear(1024, 256) -> [B, 256, T]
    -> DAC acoustic decoder (conv transpose upsampling) -> [B, 1, T*960]
    -> 24kHz waveform (25fps x 960 samples/frame)

The RVQ / DAC kernel itself lives in
``vllm_omni.model_executor.models._shared.higgs_audio_decoder`` so the new
higgs_audio_v2 integration can reuse it. ``HiggsAudioVQLayer`` and
``HiggsAudioRVQ`` are re-exported here for backward compatibility with any
external caller that imports them from this module.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.model_executor.models._shared.higgs_audio_decoder import (
    HiggsAudioRVQ,
    HiggsAudioVQLayer,
    adjust_conv_transpose_output_padding,
    load_higgs_audio_codec,
)
from vllm_omni.model_executor.models.omnivoice.config import OmniVoiceConfig

__all__ = [
    "HiggsAudioVQLayer",
    "HiggsAudioRVQ",
    "OmniVoiceDecoder",
]

logger = init_logger(__name__)


class OmniVoiceDecoder(nn.Module):
    """OmniVoice Stage 1: Token-to-audio decoder.

    Uses DAC acoustic decoder from transformers + custom HiggsAudio RVQ
    quantizer to convert 8-codebook tokens into 24kHz waveform.
    """

    def __init__(self, config: OmniVoiceConfig):
        super().__init__()
        self.config = config
        self.sample_rate = config.sample_rate
        self._loaded = False

        # These are populated by load_weights
        self.quantizer: HiggsAudioRVQ | None = None
        self.fc2: nn.Linear | None = None
        self.acoustic_decoder: nn.Module | None = None

    @torch.inference_mode()
    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """Decode audio tokens to waveform.

        Args:
            audio_codes: [B, 8, T] - 8-codebook audio token IDs

        Returns:
            waveform: [B, 1, audio_samples] at 24kHz
        """
        if not self._loaded:
            raise RuntimeError("Decoder not loaded. Call load_weights() first.")

        device = audio_codes.device

        # Transpose: [B, 8, T] -> [8, B, T]
        codes = audio_codes.transpose(0, 1).long()

        # RVQ decode: sum codebook embeddings -> [B, 1024, T]
        quantized = self.quantizer.decode(codes)

        # Project: [B, 1024, T] -> fc2 -> [B, 256, T]
        quantized = self.fc2(quantized.transpose(1, 2)).transpose(1, 2)

        # Acoustic decoder: [B, 256, T] -> [B, 1, T*960]
        audio = self.acoustic_decoder(quantized)

        # Ensure [B, 1, samples]
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        return audio.to(device)

    def _adjust_output_padding(self, decoder: nn.Module):
        """Kept for backwards compatibility; delegates to the shared helper."""
        adjust_conv_transpose_output_padding(decoder)

    def load_weights(self, model_dir: str, device: torch.device) -> None:
        """Load decoder components from audio_tokenizer/model.safetensors."""
        audio_tokenizer_path = os.path.join(model_dir, "audio_tokenizer")
        quantizer, fc2, acoustic_decoder, _tokenizer_config = load_higgs_audio_codec(
            audio_tokenizer_path, device
        )
        self.quantizer = quantizer
        self.fc2 = fc2
        self.acoustic_decoder = acoustic_decoder
        self._loaded = True

        logger.info(
            "Loaded OmniVoice decoder: %d quantizers, fc2(%d->%d)",
            len(self.quantizer.quantizers),
            self.fc2.in_features,
            self.fc2.out_features,
        )
