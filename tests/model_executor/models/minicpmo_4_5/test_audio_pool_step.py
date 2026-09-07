# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for MiniCPM-o 4.5 audio placeholder pooling."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMProcessingInfo,
    MiniCPMOConfig,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeProcessor:
    """Minimal processor surface used by ``get_audio_placeholder``."""

    def __init__(self) -> None:
        # Reproduce the cached vLLM processor's legacy initial state.
        self.pool_step = 2
        self.audio_processor = SimpleNamespace(hop_length=160)
        self.image_processor = SimpleNamespace(mean=[0.0], std=[1.0])

    def get_audio_placeholder(
        self,
        audio_lens: int,
        chunk_input: bool = True,
        chunk_length: int = 1,
    ) -> str:
        del chunk_input, chunk_length
        feature_lens = math.ceil(audio_lens / self.audio_processor.hop_length)
        cnn_feature_lens = (feature_lens - 1) // 2 + 1
        output_lens = (cnn_feature_lens - self.pool_step) // self.pool_step + 1
        return "<unk>" * output_lens


def _processing_info(pool_step: int, processor: _FakeProcessor) -> MiniCPMO45OmniLLMProcessingInfo:
    info = object.__new__(MiniCPMO45OmniLLMProcessingInfo)
    info.ctx = SimpleNamespace(
        tokenizer=object(),
        get_hf_config=lambda: SimpleNamespace(audio_pool_step=pool_step),
        get_hf_processor=lambda **kwargs: processor,
    )
    return info


@pytest.mark.parametrize(
    ("configured_pool_step", "expected_placeholders"),
    [(5, 50), (2, 125)],
)
def test_audio_placeholders_follow_checkpoint_pool_step(
    configured_pool_step: int,
    expected_placeholders: int,
) -> None:
    """The prompt path must use the same pool step as the audio encoder."""
    processor = _FakeProcessor()
    info = _processing_info(configured_pool_step, processor)

    placeholder = info.get_audio_placeholder(5 * 16_000)

    assert processor.pool_step == configured_pool_step
    assert placeholder.count("<unk>") == expected_placeholders


def test_minicpmo_4_5_pool_step_defaults_to_checkpoint_value() -> None:
    assert MiniCPMOConfig().audio_pool_step == 5
