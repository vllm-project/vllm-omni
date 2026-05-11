# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures for Raon model executor tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.tokenizers.raon_tokenizer import (
    AUDIO_END_TOKEN,
    AUDIO_INPUT_PAD_TOKEN,
    AUDIO_OUTPUT_PAD_TOKEN,
    AUDIO_START_TOKEN,
)


class _StubTokenizerMM:
    """Stub tokenizer for multimodal processor tests."""

    def __init__(self) -> None:
        self._token_ids = {
            AUDIO_START_TOKEN: 151669,
            AUDIO_END_TOKEN: 151670,
            AUDIO_OUTPUT_PAD_TOKEN: 151674,
            AUDIO_INPUT_PAD_TOKEN: 151676,
        }

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if text in self._token_ids:
            return [self._token_ids[text]]
        raise KeyError(text)


class _StubAudioItems:
    def get_audio_length(self, item_idx: int) -> int:
        assert item_idx == 0
        return 1921


class _StubMMItems:
    def get_count(self, modality: str, strict: bool = False) -> int:
        assert modality == "audio"
        return 1

    def get_items(self, modality: str, expected_type):
        assert modality == "audio"
        return _StubAudioItems()


class _StubInfo:
    def __init__(self) -> None:
        self._tokenizer = _StubTokenizerMM()
        self._hf_config = SimpleNamespace(
            audio_tokenizer_config=SimpleNamespace(
                sampling_rate=24000,
                _frame_rate=12.5,
            ),
            audio_input_token_id=151676,
        )

    def get_tokenizer(self) -> _StubTokenizerMM:
        return self._tokenizer

    def get_hf_config(self):
        return self._hf_config


@pytest.fixture()
def stub_mm_items() -> _StubMMItems:
    """Stub multimodal items container with a single audio item."""
    return _StubMMItems()


@pytest.fixture()
def stub_info() -> _StubInfo:
    """Stub processor info combining tokenizer and hf_config."""
    return _StubInfo()
