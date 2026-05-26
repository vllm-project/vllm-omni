# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SAME (Semantic-Acoustic Music Encoder) autoencoder for Stable Audio 3.

Replaces stable-audio-open's AutoencoderOobleck. Per issue #3787:
  - Stereo, 44.1 kHz audio
  - 256-dim latents
  - Separate Small / Large variants
  - Decoder supports chunked decoding to cap VRAM during inference

Ported from: https://github.com/Stability-AI/stable-audio-3 (MIT)

PORT STATUS: skeleton only — implementation must be copied from upstream
and adapted. The structure below defines the vllm-omni-facing contract that
the rest of the pipeline depends on, so it can be implemented incrementally.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SAMEDecodeOutput:
    """Output of SAMEAutoencoder.decode (mirrors diffusers `DecoderOutput`)."""

    sample: torch.Tensor  # [B, 2, samples] stereo waveform at 44.1 kHz


# ---------------------------------------------------------------------------
# USER DECISION #1 — Chunked decode policy
# ---------------------------------------------------------------------------
# SAME's decoder caps VRAM by decoding latents in chunks with crossfade overlap.
# This trades latency for VRAM. Choose defaults that match the upstream repo
# but expose them so users can tune.
#
# Approach A — single tunable: chunk_seconds + overlap_seconds (user-friendly)
# Approach B — token-level: chunk_latent_tokens + overlap_latent_tokens (precise)
# Approach C — auto: detect VRAM and pick chunk size dynamically
#
# Read upstream's autoencoder.py for the reference values, then set them below.
# TODO(stable-audio-3): fill in based on Stability-AI/stable-audio-3 reference impl
DEFAULT_DECODE_CHUNK_SECONDS: float = 0.0  # 0 = no chunking; non-zero enables
DEFAULT_DECODE_OVERLAP_SECONDS: float = 0.0


class SAMEAutoencoder(nn.Module):
    """SAME decoder wrapper exposing `.decode(latents) -> SAMEDecodeOutput`.

    The encoder is not needed for text-to-audio inference (we only decode the
    DiT's denoised latents into waveform). It is included as a placeholder for
    future audio-to-audio / inpainting modes.

    Class attrs mirror diffusers' Oobleck VAE so the pipeline can use the same
    `self.vae.config.sampling_rate`, `self.vae.hop_length` access patterns:
      - `sampling_rate`: 44100
      - `latent_channels`: 256 (per issue #3787)
      - `hop_length`: latent→waveform stride (TODO: confirm from upstream)
    """

    # TODO(stable-audio-3): replace with upstream values
    sampling_rate: int = 44100
    latent_channels: int = 256
    hop_length: int = 0  # samples-per-latent-token; fill from upstream

    def __init__(
        self,
        *,
        variant: str = "medium",  # "small_music" | "small_sfx" | "medium"
        chunk_seconds: float = DEFAULT_DECODE_CHUNK_SECONDS,
        overlap_seconds: float = DEFAULT_DECODE_OVERLAP_SECONDS,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds
        self.dtype = dtype

        # Diffusers-compatible config accessor:
        # downstream code reads `self.vae.config.sampling_rate`, etc.
        class _Cfg:
            pass

        self.config = _Cfg()
        self.config.sampling_rate = self.sampling_rate
        self.config.latent_channels = self.latent_channels

        # TODO(stable-audio-3): instantiate decoder layers.
        # Suggested structure (mirrors upstream):
        #   self.decoder = SAMEDecoder(...)
        #   self.encoder = SAMEEncoder(...)  # only needed for editing/inpainting
        raise NotImplementedError(
            "SAMEAutoencoder is a scaffold. Port the decoder from "
            "https://github.com/Stability-AI/stable-audio-3 (MIT) and remove "
            "this raise. See USER DECISION #1 above for chunked-decode policy."
        )

    @property
    def dtype_(self) -> torch.dtype:  # convenience: mirrors `.dtype` on diffusers VAE
        return self.dtype

    def decode(self, latents: torch.Tensor) -> SAMEDecodeOutput:
        """Decode [B, 256, T_latent] → SAMEDecodeOutput.sample [B, 2, T_audio].

        When `chunk_seconds > 0`, splits along the time axis, decodes each
        chunk independently, and crossfades the overlap region. This caps
        peak VRAM for long-form audio (e.g. 380s @ 44.1kHz stereo).
        """
        # TODO(stable-audio-3): implement chunked decode loop or single-shot
        raise NotImplementedError

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode [B, 2, T_audio] → latents [B, 256, T_latent]. v2 feature."""
        raise NotImplementedError("encode() is reserved for audio-to-audio mode (v2)")
