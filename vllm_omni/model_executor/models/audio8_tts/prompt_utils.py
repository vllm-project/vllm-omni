# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio8 TTS prompt construction.

Reproduces ``ArkttsProcessor._prompt_segments`` from the reference checkpoint,
so tokenisation is byte-identical to HF inference.
"""

from __future__ import annotations

from typing import Any

import regex as re

SYSTEM_PROMPT_TEXT_ONLY = "convert the provided text to speech"
SYSTEM_PROMPT_CLONE_PREFIX = "convert the provided text to speech reference to the following:\n\nText:\n"
SYSTEM_PROMPT_CLONE_SUFFIX = "\n\nSpeech:\n"

_SPEAKER_TAG_PATTERN = re.compile(r"<\|speaker:\d+\|>")
_LEGACY_SPEAKER_TAG_PATTERN = re.compile(r"<speaker:(\d+)>")
_CONTROL_TOKEN_PATTERN = re.compile(r"<\|[^>]+\|>")


def clean_text(text: str) -> str:
    """Collapse all whitespace runs to single spaces (reference ``_clean_text``)."""
    return " ".join(str(text).strip().split())


def normalize_text(text: str, *, add_default_speaker: bool = False) -> str:
    """Clean text, normalise speaker tags, reject other control tokens.

    ``<|speaker:N|>`` is allowed; any other ``<|...|>`` token raises so callers
    cannot inject ``<|semantic:...|>`` / ``<|im_end|>`` into a prompt.
    """
    normalized = clean_text(_LEGACY_SPEAKER_TAG_PATTERN.sub(r"<|speaker:\1|>", text))

    disallowed = [
        token for token in _CONTROL_TOKEN_PATTERN.findall(normalized) if not _SPEAKER_TAG_PATTERN.fullmatch(token)
    ]
    if disallowed:
        tokens = ", ".join(sorted(set(disallowed)))
        raise ValueError(f"Audio8 TTS input contains unsupported control token(s): {tokens}")

    if add_default_speaker and not _SPEAKER_TAG_PATTERN.search(normalized):
        normalized = f"<|speaker:0|>{normalized}"
    return normalized


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _encode_segments(tokenizer: Any, segments: list[str]) -> list[int]:
    ids: list[int] = []
    for segment in segments:
        ids.extend(_encode(tokenizer, segment))
    return ids


def build_text_only_prompt_ids(tokenizer: Any, text: str) -> tuple[list[int], str]:
    """Build the no-reference prompt.

    Returns:
        ``(prompt_token_ids, normalized_text)``.
    """
    normalized = normalize_text(text)
    if not normalized:
        raise ValueError("Audio8 TTS input text must not be empty")
    ids = _encode_segments(
        tokenizer,
        [
            "<|im_start|>system\n",
            SYSTEM_PROMPT_TEXT_ONLY,
            "<|im_end|>\n",
            "<|im_start|>user\n",
            normalized,
            "<|im_end|>\n",
            "<|im_start|>assistant\n<|voice|>",
        ],
    )
    return ids, normalized


def build_voice_clone_prompt_parts(
    tokenizer: Any,
    text: str,
    ref_text: str,
) -> tuple[list[int], list[int], str, str]:
    """Build the clone prompt around the reference-code slot.

    Caller splices ``[code + semantic_begin_id for code in ref_codes[:, 0]]``
    between the returned prefix and suffix.
    """
    normalized = normalize_text(text)
    if not normalized:
        raise ValueError("Audio8 TTS input text must not be empty")
    normalized_ref = normalize_text(ref_text, add_default_speaker=True)
    if not normalized_ref:
        raise ValueError("Audio8 TTS voice cloning requires a non-empty reference transcript")

    prefix = _encode_segments(
        tokenizer,
        [
            "<|im_start|>system\n",
            SYSTEM_PROMPT_CLONE_PREFIX,
            normalized_ref,
            SYSTEM_PROMPT_CLONE_SUFFIX,
        ],
    )
    suffix = _encode_segments(
        tokenizer,
        [
            "<|im_end|>\n",
            "<|im_start|>user\n",
            normalized,
            "<|im_end|>\n",
            "<|im_start|>assistant\n<|voice|>",
        ],
    )
    return prefix, suffix, normalized, normalized_ref


def build_voice_clone_prompt_ids(
    tokenizer: Any,
    text: str,
    ref_text: str,
    semantic_token_ids: list[int],
) -> tuple[list[int], int, str, str]:
    """Build the full clone prompt.

    ``ref_start_index`` is the position of the first reference frame, i.e.
    where the residual codebook embeddings must be added.
    """
    prefix, suffix, normalized, normalized_ref = build_voice_clone_prompt_parts(tokenizer, text, ref_text)
    return prefix + list(semantic_token_ids) + suffix, len(prefix), normalized, normalized_ref


def estimate_voice_clone_prompt_len(
    tokenizer: Any,
    normalized_text: str,
    normalized_ref_text: str,
    ref_frames: int,
) -> int:
    """Exact clone prompt length, without encoding the reference audio."""
    prefix, suffix, _, _ = build_voice_clone_prompt_parts(tokenizer, normalized_text, normalized_ref_text)
    return len(prefix) + max(0, int(ref_frames)) + len(suffix)


__all__ = [
    "SYSTEM_PROMPT_CLONE_PREFIX",
    "SYSTEM_PROMPT_CLONE_SUFFIX",
    "SYSTEM_PROMPT_TEXT_ONLY",
    "build_text_only_prompt_ids",
    "build_voice_clone_prompt_ids",
    "build_voice_clone_prompt_parts",
    "clean_text",
    "estimate_voice_clone_prompt_len",
    "normalize_text",
]
