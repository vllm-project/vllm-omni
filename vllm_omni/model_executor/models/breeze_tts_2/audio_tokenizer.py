# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Reference-audio encoding helpers for Breeze-TTS-2.

The Breeze prompt uses a Qwen3-TTS tokenizer only for reference audio.  This
module deliberately does not contain text tokenization or waveform decoding;
the latter belongs to the pipeline's codec stage.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


class BreezeReferenceAudioTokenizer:
    """Small adapter that normalizes Qwen3-TTS encoder output to ``(T, Q)``.

    The underlying tokenizer is created once per serving worker and is reused
    for all requests.  Keeping this wrapper independent from the
    prompt builder makes it straightforward to inject a fake encoder in tests
    and to replace the upstream Qwen3 implementation later.
    """

    def __init__(self, tokenizer: Any, *, num_codebooks: int = 16, codebook_size: int = 2048) -> None:
        self.tokenizer = tokenizer
        self.num_codebooks = int(num_codebooks)
        self.codebook_size = int(codebook_size)
        if self.num_codebooks <= 0 or self.codebook_size <= 0:
            raise ValueError("num_codebooks and codebook_size must be positive")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        num_codebooks: int = 16,
        codebook_size: int = 2048,
        **kwargs: Any,
    ) -> BreezeReferenceAudioTokenizer:
        """Load the bundled ``audio_tokenizer`` with the existing Qwen wrapper."""
        from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_tokenizer import (
            Qwen3TTSTokenizer,
        )

        local_path = Path(model_path) / "audio_tokenizer"
        if not local_path.is_dir():
            raise FileNotFoundError(f"Breeze reference audio tokenizer was not found at {local_path}")
        tokenizer = Qwen3TTSTokenizer.from_pretrained(str(local_path), **kwargs)
        return cls(tokenizer, num_codebooks=num_codebooks, codebook_size=codebook_size)

    @torch.inference_mode()
    def encode(self, audio: Any, sample_rate: int | None = None) -> torch.Tensor:
        """Encode one waveform or audio source into ``(T, Q)`` codes."""
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        elif isinstance(audio, list) and audio and all(isinstance(x, (int, float)) for x in audio):
            audio = np.asarray(audio, dtype=np.float32)
        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            # Accept both common layouts: (samples, channels) and
            # (channels, samples).  Qwen3TTSTokenizer expects one 1-D clip.
            channel_first = audio.shape[0] <= 8 and audio.shape[-1] > audio.shape[0]
            audio = audio.mean(axis=0 if channel_first else -1)

        if isinstance(audio, np.ndarray) and sample_rate is None:
            raise ValueError("sample_rate is required when ref_audio is a numpy waveform")

        encoded = self.tokenizer.encode(audio, sr=sample_rate)
        return self._normalize_codes(encoded)

    def _normalize_codes(self, encoded: Any) -> torch.Tensor:
        codes = getattr(encoded, "audio_codes", None)
        if codes is None and isinstance(encoded, Mapping):
            codes = encoded.get("audio_codes")
        if codes is None and isinstance(encoded, (tuple, list)):
            codes = encoded[0] if encoded else None
        if codes is None:
            raise ValueError("reference audio tokenizer returned no audio_codes")

        if isinstance(codes, (tuple, list)) and codes:
            # Qwen3TTSTokenizer returns one tensor per input waveform.
            if len(codes) != 1:
                raise ValueError(f"only one reference waveform is supported, got {len(codes)}")
            codes = codes[0]
        codes = torch.as_tensor(codes)
        if codes.ndim == 3:
            if codes.shape[0] != 1:
                raise ValueError(f"only one reference waveform is supported, got {tuple(codes.shape)}")
            codes = codes[0]
        if codes.ndim != 2:
            raise ValueError(f"reference audio codes must be 2D, got {tuple(codes.shape)}")
        if codes.shape[0] == 0:
            raise ValueError("reference audio tokenizer returned no frames")
        if codes.shape[-1] != self.num_codebooks:
            if codes.shape[0] == self.num_codebooks:
                codes = codes.transpose(0, 1)
            else:
                raise ValueError(
                    f"reference audio codes must have {self.num_codebooks} codebooks, got {tuple(codes.shape)}"
                )
        codes = codes.to(device="cpu", dtype=torch.long).contiguous()
        if codes.numel() and (int(codes.min()) < 0 or int(codes.max()) >= self.codebook_size):
            raise ValueError(
                f"reference audio code outside [0, {self.codebook_size}): "
                f"min={int(codes.min())}, max={int(codes.max())}"
            )
        # int16 is enough for Mimi/Qwen3-TTS codes and halves IPC/cache memory.
        return codes.to(dtype=torch.int16)


__all__ = ["BreezeReferenceAudioTokenizer"]
