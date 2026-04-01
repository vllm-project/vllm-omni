# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for chat endpoint speaker validation.

Tests the speaker validation logic that mirrors the production code in
OmniOpenAIServingChat._get_supported_speakers() and the speaker check
in _preprocess_chat. Uses a local reimplementation since serving_chat.py
has heavy transitive imports (vllm._C).
"""

from unittest.mock import MagicMock

import pytest


def _get_supported_speakers(handler) -> set[str]:
    """Mirror of OmniOpenAIServingChat._get_supported_speakers."""
    if handler._supported_speakers is not None:
        return handler._supported_speakers
    try:
        hf_config = handler.model_config.hf_config
        for config_attr in ["talker_config"]:
            config = getattr(hf_config, config_attr, None)
            if config is None:
                continue
            for spk_attr in ["speaker_id", "spk_id"]:
                speakers_dict = config.get(spk_attr) if isinstance(config, dict) else getattr(config, spk_attr, None)
                if speakers_dict and isinstance(speakers_dict, dict):
                    handler._supported_speakers = {s.lower() for s in speakers_dict}
                    return handler._supported_speakers
    except Exception:
        pass
    handler._supported_speakers = set()
    return handler._supported_speakers


def _validate_speaker(speaker: str, supported: set[str]) -> str | None:
    """Mirror of the speaker validation in _preprocess_chat.

    Returns error message if invalid, None if valid.
    """
    if not speaker or not speaker.strip():
        return None
    normalized = speaker.lower().strip()
    if supported and normalized not in supported:
        return f"Invalid speaker '{speaker}'. Supported: {', '.join(sorted(supported))}"
    return None


def _make_handler(speakers: dict | None):
    handler = MagicMock()
    handler._supported_speakers = None
    talker_config = MagicMock()
    talker_config.speaker_id = speakers
    talker_config.spk_id = None
    handler.model_config.hf_config.talker_config = talker_config
    return handler


@pytest.mark.core_model
@pytest.mark.cpu
class TestChatSpeakerValidation:
    def test_valid_speaker_accepted(self):
        handler = _make_handler({"Vivian": 0, "Ethan": 1})
        supported = _get_supported_speakers(handler)
        assert _validate_speaker("Vivian", supported) is None
        assert _validate_speaker("vivian", supported) is None

    def test_invalid_speaker_rejected(self):
        handler = _make_handler({"Vivian": 0, "Ethan": 1})
        supported = _get_supported_speakers(handler)
        error = _validate_speaker("uncle_fu", supported)
        assert error is not None
        assert "uncle_fu" in error
        assert "ethan" in error
        assert "vivian" in error

    def test_no_speakers_skips_validation(self):
        handler = _make_handler(None)
        supported = _get_supported_speakers(handler)
        assert supported == set()
        assert _validate_speaker("anything", supported) is None

    def test_cached_after_first_call(self):
        handler = _make_handler({"Vivian": 0})
        _get_supported_speakers(handler)
        _get_supported_speakers(handler)
        assert handler._supported_speakers == {"vivian"}
