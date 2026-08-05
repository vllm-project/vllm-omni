# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prompt builder for higgs-audio v3 TTS.

Prompt formats:
  Zero-shot:
    <|tts|> <|text|> {text tokens} <|audio|>
  Voice clone (no ref text):
    <|tts|> <|ref_audio|> [-100]×N <|text|> {text tokens} <|audio|>
  Voice clone (with ref text):
    <|tts|> <|ref_text|> {ref text tokens} <|ref_audio|> [-100]×N <|text|> {text tokens} <|audio|>

-100 placeholders are replaced at prefill time with fused multi-codebook
embeddings of the delay-pattern-encoded reference audio codes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

__all__ = [
    "HiggsAudioV3TokenizerAdapter",
    "AUDIO_PLACEHOLDER_ID",
    "encode_reference_audio",
    "apply_delay_pattern",
]

AUDIO_PLACEHOLDER_ID = -100

_REQUIRED_SPECIALS: tuple[str, ...] = (
    "<|tts|>",
    "<|text|>",
    "<|audio|>",
)

# Codec stream special token IDs (inside the codebook vocab, not the LM vocab)
BOC_ID = 1024
EOC_ID = 1025


def apply_delay_pattern(codes_tn: torch.Tensor) -> torch.Tensor:
    """Apply MusicGen-style delay pattern to raw codes.

    Input: [T, N] raw codes (T frames, N codebooks).
    Output: [T + N - 1, N] delayed codes with BOC/EOC padding.

    Codebook c is delayed by c positions: rows 0..c-1 get BOC,
    rows c..c+T-1 get real codes, rows c+T..T+N-2 get EOC.
    """
    T, N = codes_tn.shape
    L = T + N - 1
    delayed = torch.full((L, N), EOC_ID, dtype=codes_tn.dtype)
    for c in range(N):
        delayed[:c, c] = BOC_ID
        delayed[c : c + T, c] = codes_tn[:, c]
    return delayed


def encode_reference_audio(
    wav: np.ndarray,
    sr: int,
) -> torch.Tensor:
    """Encode a reference audio clip to codec codes [T, num_codebooks].

    Uses the same HiggsAudioV2TokenizerModel as v2 (same codec).
    Returns raw codes before delay pattern application.
    """
    from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_tokenizer import (
        _encode_ref_audio_codes,
    )

    # _encode_ref_audio_codes returns [num_codebooks, T_raw]
    codes_qt = _encode_ref_audio_codes(wav, sr)
    # Transpose to [T_raw, num_codebooks]
    return codes_qt.transpose(0, 1).contiguous()


class HiggsAudioV3TokenizerAdapter:
    """Wraps the HF tokenizer and builds TTS prompts."""

    def __init__(self, tokenizer: Any) -> None:
        self._tok = tokenizer
        vocab = dict(tokenizer.get_added_vocab())
        missing = [t for t in _REQUIRED_SPECIALS if t not in vocab]
        if missing:
            raise ValueError(f"Tokenizer is missing Higgs TTS v3 specials: {missing}")
        self.tts_id: int = vocab["<|tts|>"]
        self.text_id: int = vocab["<|text|>"]
        self.audio_id: int = vocab["<|audio|>"]
        self.ref_audio_id: int | None = vocab.get("<|ref_audio|>")
        self.ref_text_id: int | None = vocab.get("<|ref_text|>")

    @property
    def tokenizer(self) -> Any:
        return self._tok

    def build_prompt(
        self,
        text: str,
        *,
        num_ref_tokens: int = 0,
        reference_text: str | None = None,
    ) -> list[int]:
        """Build a TTS prompt.

        Args:
            text: Target text to synthesize.
            num_ref_tokens: Number of delay-pattern reference code rows.
                0 means zero-shot (no voice clone).
            reference_text: Optional transcript of the reference audio.
        """
        if not text or not text.strip():
            raise ValueError("Text input must be non-empty for TTS")
        if num_ref_tokens < 0:
            raise ValueError(f"num_ref_tokens must be >= 0, got {num_ref_tokens}")

        ids: list[int] = [self.tts_id]

        # Voice clone: optional ref text + ref audio placeholders
        if reference_text and num_ref_tokens > 0 and self.ref_text_id is not None:
            ids.append(self.ref_text_id)
            ids.extend(self._tok.encode(reference_text, add_special_tokens=False))
        if num_ref_tokens > 0:
            if self.ref_audio_id is None:
                raise ValueError("Tokenizer missing <|ref_audio|> for voice clone")
            ids.append(self.ref_audio_id)
            ids.extend([AUDIO_PLACEHOLDER_ID] * num_ref_tokens)

        ids.append(self.text_id)
        ids.extend(self._tok.encode(text, add_special_tokens=False))
        ids.append(self.audio_id)
        return ids
