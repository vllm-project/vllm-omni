"""Shared Fish Speech prompt construction helpers."""

from __future__ import annotations

import re
from typing import Any

FISH_TEXT_ONLY_SYSTEM_PROMPT = "convert the provided text to speech"
FISH_CLONE_SYSTEM_PROMPT_PREFIX = "convert the provided text to speech reference to the following:\n\nText:\n"
FISH_CLONE_SYSTEM_PROMPT_SUFFIX = "\n\nSpeech:\n"

_LEGACY_SPEAKER_TAG_PATTERN = re.compile(r"<speaker:(\d+)>")
_CANONICAL_SPEAKER_TAG_PATTERN = re.compile(r"<\|speaker:\d+\|>")
_CONTROL_TOKEN_PATTERN = re.compile(r"<\|[^>]+\|>")


def normalize_fish_speech_text(text: str, *, add_default_speaker: bool = False) -> str:
    """Normalize supported speaker tags and reject unsafe control tokens."""
    normalized = _LEGACY_SPEAKER_TAG_PATTERN.sub(r"<|speaker:\1|>", text)

    disallowed_tokens = [
        token
        for token in _CONTROL_TOKEN_PATTERN.findall(normalized)
        if not _CANONICAL_SPEAKER_TAG_PATTERN.fullmatch(token)
    ]
    if disallowed_tokens:
        disallowed_list = ", ".join(sorted(set(disallowed_tokens)))
        raise ValueError(f"Fish Speech input contains unsupported control token(s): {disallowed_list}")

    if add_default_speaker and not _CANONICAL_SPEAKER_TAG_PATTERN.search(normalized):
        normalized = f"<|speaker:0|>{normalized}"

    return normalized


def _encode_plain_text(tokenizer: Any, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _encode_control_token(tokenizer: Any, token: str) -> list[int]:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError(f"Fish Speech tokenizer is missing required control token: {token}")
    return [int(token_id)]


def _build_message_prefix(tokenizer: Any, role: str) -> list[int]:
    return _encode_control_token(tokenizer, "<|im_start|>") + _encode_plain_text(tokenizer, f"{role}\n")


def _build_text_only_prompt_ids_from_normalized(tokenizer: Any, normalized_text: str) -> list[int]:
    return (
        _build_message_prefix(tokenizer, "system")
        + _encode_plain_text(tokenizer, FISH_TEXT_ONLY_SYSTEM_PROMPT)
        + _encode_control_token(tokenizer, "<|im_end|>")
        + _encode_plain_text(tokenizer, "\n")
        + _build_message_prefix(tokenizer, "user")
        + _encode_plain_text(tokenizer, normalized_text)
        + _encode_control_token(tokenizer, "<|im_end|>")
        + _encode_plain_text(tokenizer, "\n")
        + _build_message_prefix(tokenizer, "assistant")
        + _encode_control_token(tokenizer, "<|voice|>")
    )


def build_fish_text_only_prompt_ids(tokenizer: Any, text: str) -> tuple[list[int], str]:
    normalized_text = normalize_fish_speech_text(text)
    return _build_text_only_prompt_ids_from_normalized(tokenizer, normalized_text), normalized_text


def _build_voice_clone_prompt_ids_from_normalized(
    tokenizer: Any,
    normalized_text: str,
    normalized_ref_text: str,
    semantic_token_ids: list[int],
) -> list[int]:
    return (
        _build_message_prefix(tokenizer, "system")
        + _encode_plain_text(
            tokenizer,
            FISH_CLONE_SYSTEM_PROMPT_PREFIX + normalized_ref_text + FISH_CLONE_SYSTEM_PROMPT_SUFFIX,
        )
        + _encode_control_token(tokenizer, "<|audio_start|>")
        + semantic_token_ids
        + _encode_control_token(tokenizer, "<|audio_end|>")
        + _encode_control_token(tokenizer, "<|im_end|>")
        + _encode_plain_text(tokenizer, "\n")
        + _build_message_prefix(tokenizer, "user")
        + _encode_plain_text(tokenizer, normalized_text)
        + _encode_control_token(tokenizer, "<|im_end|>")
        + _encode_plain_text(tokenizer, "\n")
        + _build_message_prefix(tokenizer, "assistant")
        + _encode_control_token(tokenizer, "<|voice|>")
    )


def normalize_fish_voice_clone_texts(text: str, ref_text: str) -> tuple[str, str]:
    normalized_text = normalize_fish_speech_text(text)
    normalized_ref_text = normalize_fish_speech_text(ref_text, add_default_speaker=True)
    return normalized_text, normalized_ref_text


def build_fish_voice_clone_prompt_ids(
    tokenizer: Any,
    text: str,
    ref_text: str,
    semantic_token_ids: list[int],
) -> tuple[list[int], str, str]:
    normalized_text, normalized_ref_text = normalize_fish_voice_clone_texts(text, ref_text)
    return (
        _build_voice_clone_prompt_ids_from_normalized(
            tokenizer,
            normalized_text,
            normalized_ref_text,
            semantic_token_ids,
        ),
        normalized_text,
        normalized_ref_text,
    )


def estimate_fish_voice_clone_prompt_len_from_normalized(
    tokenizer: Any,
    normalized_text: str,
    normalized_ref_text: str,
    semantic_len: int,
) -> int:
    prompt_ids = _build_voice_clone_prompt_ids_from_normalized(
        tokenizer,
        normalized_text,
        normalized_ref_text,
        [0] * semantic_len,
    )
    return len(prompt_ids)


def estimate_fish_voice_clone_prompt_len(tokenizer: Any, text: str, ref_text: str, semantic_len: int) -> int:
    normalized_text, normalized_ref_text = normalize_fish_voice_clone_texts(text, ref_text)
    return estimate_fish_voice_clone_prompt_len_from_normalized(
        tokenizer,
        normalized_text,
        normalized_ref_text,
        semantic_len,
    )
