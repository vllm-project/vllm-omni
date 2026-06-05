# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prompt builder for higgs-audio v3 (plain TTS only).

Zero-shot prompt format:
    <|tts|> <|text|> {user text tokens} <|audio|>

Voice-clone and ref-text prompts are not supported in this phase.
"""

from __future__ import annotations

from typing import Any

__all__ = ["HiggsAudioV3TokenizerAdapter", "AUDIO_PLACEHOLDER_ID"]

AUDIO_PLACEHOLDER_ID = -100

_REQUIRED_SPECIALS: tuple[str, ...] = (
    "<|tts|>",
    "<|text|>",
    "<|audio|>",
)


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
        # Optional voice-clone specials (not used in this phase)
        self.ref_audio_id: int | None = vocab.get("<|ref_audio|>")
        self.ref_text_id: int | None = vocab.get("<|ref_text|>")

    @property
    def tokenizer(self) -> Any:
        return self._tok

    def build_prompt(self, text: str) -> list[int]:
        """Build a zero-shot TTS prompt: <|tts|> <|text|> tokens <|audio|>."""
        if not text or not text.strip():
            raise ValueError("Text input must be non-empty for TTS")
        ids: list[int] = [self.tts_id, self.text_id]
        ids.extend(self._tok.encode(text, add_special_tokens=False))
        ids.append(self.audio_id)
        return ids
