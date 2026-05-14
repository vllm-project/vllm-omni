# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
"""Shared TTS utility functions for speaker and language extraction.

These utilities are model-agnostic and can be used by any TTS model stage
processor (qwen3_omni, qwen2_5_omni, qwen3_tts, etc.).
"""

from typing import Any

from vllm_omni.data_entry_keys import OmniInputStruct

# =============================================================================
# Read from ``OmniInputStruct`` — producers emit ``OmniInputStruct(...)``
# directly on ``additional_information``. Per-model substructs live under
# their model name (e.g. ``qwen3_tts.task_type``).
# =============================================================================


def input_field_from_request(request: Any, field: str) -> Any:
    """Read a top-level ``OmniInputStruct`` field from the request envelope."""
    add_info = getattr(request, "additional_information", None)
    if isinstance(add_info, OmniInputStruct):
        return getattr(add_info, field, None)
    return None


def _input_struct(prompt: dict[str, Any] | list[dict[str, Any]] | None, index: int) -> OmniInputStruct | None:
    if prompt is None:
        return None
    p = prompt[index] if isinstance(prompt, list) and index < len(prompt) else prompt
    if not isinstance(p, dict):
        return None
    add_info = p.get("additional_information")
    return add_info if isinstance(add_info, OmniInputStruct) else None


def input_speaker_from_request(request: Any) -> str | None:
    val = input_field_from_request(request, "speaker")
    if isinstance(val, list) and val:
        val = val[0]
    if isinstance(val, str) and val.strip():
        return val.lower().strip()
    return None


def input_language_from_request(request: Any) -> list[str] | None:
    val = input_field_from_request(request, "language")
    if isinstance(val, list) and val:
        return val
    if isinstance(val, str) and val.strip():
        return [val.strip()]
    return None


def input_speaker_from_prompt(prompt: dict[str, Any] | list[dict[str, Any]] | None, index: int = 0) -> list[str] | None:
    inp = _input_struct(prompt, index)
    if inp is None or inp.speaker is None:
        return None
    return inp.speaker if isinstance(inp.speaker, list) else [inp.speaker]


def input_language_from_prompt(
    prompt: dict[str, Any] | list[dict[str, Any]] | None, index: int = 0
) -> list[str] | None:
    inp = _input_struct(prompt, index)
    if inp is None or inp.language is None:
        return None
    return inp.language if isinstance(inp.language, list) else [inp.language]


# =============================================================================
# Speaker helpers
# =============================================================================


def extract_speaker_from_prompt(
    prompt: Any,
    index: int = 0,
) -> list[str] | None:
    """Extract speaker from a prompt's additional_information dict.

    Used in non-async stage processors where the prompt is an
    OmniTokensPrompt / TextPrompt dict (or a list of them).

    Args:
        prompt: A single prompt dict, or a list of prompt dicts.
        index: Which element to pick when prompt is a list.

    Returns:
        The speaker as a list (for serialization compatibility), or None.
    """
    if prompt is None:
        return None
    p = prompt[index] if isinstance(prompt, list) and index < len(prompt) else prompt
    if p is None:
        return None
    add_info = p.get("additional_information")
    if not isinstance(add_info, dict):
        return None
    speaker = add_info.get("speaker")
    if isinstance(speaker, list) and speaker:
        return speaker
    return None


# =============================================================================
# Language helpers
# =============================================================================


def extract_language_from_prompt(
    prompt: Any,
    index: int = 0,
) -> list[str] | None:
    """Extract language from a prompt's additional_information dict.
    Args:
        prompt: A single prompt dict, or a list of prompt dicts.
        index: Which element to pick when prompt is a list.

    Returns:
        The language as a list (for serialization compatibility), or None.
    """
    if prompt is None:
        return None
    p = prompt[index] if isinstance(prompt, list) and index < len(prompt) else prompt
    if p is None:
        return None
    add_info = p.get("additional_information")
    if not isinstance(add_info, dict):
        return None
    language = add_info.get("language")
    if isinstance(language, list) and language:
        return language
    return None
