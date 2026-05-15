# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage 1 (codec decoder) for higgs-audio v2.

Consumes audio codes emitted by Stage 0 (the talker) and renders 24 kHz mono
PCM via the shared HiggsAudioV2 codec kernel
(`vllm_omni.model_executor.models._shared.higgs_audio_decoder`).

Inputs:
    audio_codes: torch.LongTensor of shape [B, num_codebooks=8, T]
        Real code IDs in [0, 1023]. Any value >= 1024 (the audio_stream_bos_id
        or audio_stream_eos_id markers) is rejected with `ValueError` because
        stream specials must be consumed at Stage 0 and never reach the codec.

Outputs:
    waveform: torch.FloatTensor of shape [B, 1, T * 960]  (24 kHz, mono)

The class implements just enough surface to plug into the existing two-stage
TTS plumbing; the engine-side scaffolding (`pipeline.py`, the stage-input
processor, and the registry entry) wraps it.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.model_executor.models._shared.higgs_audio_decoder import (
    HiggsAudioRVQ,
    load_higgs_audio_codec,
)
from vllm_omni.model_executor.models.higgs_audio_v2.configuration_higgs_audio_v2 import (
    HiggsAudioV2Config,
)

__all__ = [
    "HiggsAudioV2Code2Wav",
    "HiggsAudioV2Code2WavForConditionalGeneration",
]

logger = init_logger(__name__)


class HiggsAudioV2Code2Wav(nn.Module):
    """Stage-1 codec decoder for higgs-audio v2 (audio codes -> 24 kHz PCM).

    This is structurally analogous to ``OmniVoiceDecoder``: it owns the RVQ
    quantizer, the fc2 projection, and the DAC acoustic decoder. All three
    are loaded via the shared kernel so behavior is bit-identical to the
    existing OmniVoice path.
    """

    def __init__(self, config: HiggsAudioV2Config):
        super().__init__()
        self.config = config
        self.sample_rate: int = int(config.sample_rate)
        self.num_codebooks: int = int(config.num_codebooks)
        self.num_real_codes: int = int(config.num_real_codes)

        # Populated by load_weights().
        self.quantizer: HiggsAudioRVQ | None = None
        self.fc2: nn.Linear | None = None
        self.acoustic_decoder: nn.Module | None = None
        self._loaded: bool = False

    # ------------------------------------------------------------------ forward
    @torch.inference_mode()
    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        if not self._loaded:
            raise RuntimeError("HiggsAudioV2Code2Wav not loaded. Call load_weights() first.")

        codes = self._validate_codes(audio_codes)
        device = codes.device

        # [B, 8, T] -> [8, B, T] for the RVQ kernel
        rvq_codes = codes.transpose(0, 1).long()

        # RVQ decode -> [B, hidden, T]
        quantized = self.quantizer.decode(rvq_codes)
        # fc2 projection -> [B, 256, T]
        quantized = self.fc2(quantized.transpose(1, 2)).transpose(1, 2)
        # DAC decoder -> [B, 1, T*960]
        audio = self.acoustic_decoder(quantized)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        return audio.to(device)

    # ------------------------------------------------------------------ load
    def load_weights(self, model_dir: str, device: torch.device | None = None) -> None:
        """Load codec weights for Stage 1.

        ``model_dir`` may be either the standalone tokenizer repo (containing
        ``config.json`` + ``model.safetensors`` at the root) or the 3B Stage-0
        checkpoint that bundles the tokenizer at ``<model_dir>/audio_tokenizer/``.
        The ``audio_tokenizer_subdir`` config field controls which layout
        applies; an empty subdir means the model_dir IS the tokenizer dir.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        subdir = self.config.audio_tokenizer_subdir or ""
        audio_tokenizer_path = os.path.join(model_dir, subdir) if subdir else model_dir
        quantizer, fc2, acoustic_decoder, _tokenizer_config = load_higgs_audio_codec(
            audio_tokenizer_path, device
        )
        if len(quantizer.quantizers) != self.num_codebooks:
            raise ValueError(
                f"checkpoint has {len(quantizer.quantizers)} quantizers but config.num_codebooks={self.num_codebooks}"
            )
        self.quantizer = quantizer
        self.fc2 = fc2
        self.acoustic_decoder = acoustic_decoder
        self._loaded = True
        logger.info(
            "Loaded HiggsAudioV2Code2Wav: %d quantizers, fc2(%d->%d), sample_rate=%d",
            len(self.quantizer.quantizers),
            self.fc2.in_features,
            self.fc2.out_features,
            self.sample_rate,
        )

    # ------------------------------------------------------------------ helpers
    def _validate_codes(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """Ensure shape and value range; reject stream specials with ValueError."""
        if not isinstance(audio_codes, torch.Tensor):
            raise TypeError(f"audio_codes must be a torch.Tensor, got {type(audio_codes)!r}")
        if audio_codes.ndim != 3:
            raise ValueError(
                f"audio_codes must have shape [B, num_codebooks={self.num_codebooks}, T]; "
                f"got shape {tuple(audio_codes.shape)}"
            )
        if int(audio_codes.shape[1]) != self.num_codebooks:
            raise ValueError(
                f"audio_codes second dim must equal num_codebooks={self.num_codebooks}; "
                f"got {int(audio_codes.shape[1])}"
            )
        if audio_codes.numel() > 0:
            max_val = int(audio_codes.max().item())
            min_val = int(audio_codes.min().item())
            if max_val >= self.num_real_codes or min_val < 0:
                raise ValueError(
                    "audio_codes contains stream-special or out-of-range IDs: "
                    f"min={min_val}, max={max_val}; real code range is "
                    f"[0, {self.num_real_codes - 1}]. Filter audio_stream_bos_id="
                    f"{self.config.audio_stream_bos_id} and audio_stream_eos_id="
                    f"{self.config.audio_stream_eos_id} (and anything above) at "
                    "Stage 0 before sending codes to the codec decoder."
                )
        return audio_codes


# Convenience alias for registry-based loading (matches the canonical
# "<ModelType>ForConditionalGeneration"-style architecture identifier).
HiggsAudioV2Code2WavForConditionalGeneration = HiggsAudioV2Code2Wav
