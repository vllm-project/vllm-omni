# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the MiniCPM-o 4.5 ``audio_embeds`` input path.

Two defects made the OpenAI ``audio_embeds`` content part unusable with this model:

1. ``MiniCPMOAudioEmbeddingItems`` registered itself under ``modality="image"``
   (copy-pasted from ``MiniCPMVImageEmbeddingItems``), so every request failed with
   ``Modality 'image' not found. Available modalities: {'audio'}``.
2. ``_get_prompt_updates`` sized the placeholder from
   ``sum(map(len, single_audio_embeds))``. ``MiniCPMOAudioEmbeddingInputs`` declares
   ``TensorShape("bn", "s", "h")``, so one item is ``(s, h)`` and that expression
   iterates rows and sums the hidden size -- ``s * h`` instead of ``s``. A one-second
   slice of shape ``(10, 4096)`` asked for 40960 placeholder tokens instead of 10,
   pushing every real request past ``max_model_len``.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from vllm.multimodal.parse import MultiModalDataItems

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMMultiModalProcessor,
    MiniCPMO45OmniLLMProcessingInfo,
    MiniCPMOAudioEmbeddingItems,
    _minicpmo_field_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

POOL_STEP = 5
HIDDEN_SIZE = 4096
#: Audio embedding vectors produced per second: 100 mel -> 50 cnn -> (50-5)//5+1.
EMBEDS_PER_SECOND = 10


class _FakeProcessor:
    """Minimal processor surface used by ``get_audio_placeholder``."""

    def __init__(self) -> None:
        self.pool_step = POOL_STEP
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


class _IdentityTokenizer:
    """``_get_prompt_updates`` round-trips the image/video patterns through the
    tokenizer to detect encode/decode drift; an identity pair keeps that a no-op."""

    def encode(self, text: str, add_special_tokens: bool = True) -> str:
        del add_special_tokens
        return text

    def decode(self, tokens: str) -> str:
        return tokens


def _processing_info() -> MiniCPMO45OmniLLMProcessingInfo:
    processor = _FakeProcessor()
    tokenizer = _IdentityTokenizer()
    info = object.__new__(MiniCPMO45OmniLLMProcessingInfo)
    info.ctx = SimpleNamespace(
        tokenizer=tokenizer,
        get_tokenizer=lambda: tokenizer,
        get_hf_config=lambda: SimpleNamespace(audio_pool_step=POOL_STEP),
        get_hf_processor=lambda **kwargs: processor,
    )
    return info


def _audio_placeholder_for(single_audio_embeds: torch.Tensor) -> str:
    """Run the real prompt-update path and return the audio placeholder text.

    Goes through ``_get_prompt_updates`` rather than recomputing the length, so the
    ``sum(map(len, ...))`` regression is actually exercised.
    """
    info = _processing_info()
    proc = object.__new__(MiniCPMO45OmniLLMMultiModalProcessor)
    proc.info = info

    items = MiniCPMOAudioEmbeddingItems(
        {"audio_embeds": single_audio_embeds.unsqueeze(0)},
        fields_factory=_minicpmo_field_config,
    )
    mm_items = MultiModalDataItems({"audio": items})

    updates = proc._get_prompt_updates(mm_items, {}, {})
    audio_update = next(u for u in updates if u.modality == "audio")
    return audio_update.content(0).full


def test_audio_embedding_items_use_the_audio_modality() -> None:
    """Registering under ``image`` makes every ``audio_embeds`` request fail."""
    items = MiniCPMOAudioEmbeddingItems(
        {"audio_embeds": torch.zeros(1, EMBEDS_PER_SECOND, HIDDEN_SIZE)},
        fields_factory=_minicpmo_field_config,
    )

    assert items.modality == "audio"


@pytest.mark.parametrize("num_seconds", [1, 3, 30])
def test_placeholder_length_matches_embedding_count(num_seconds: int) -> None:
    """An ``(s, h)`` item must yield exactly ``s`` placeholder tokens."""
    num_embeds = num_seconds * EMBEDS_PER_SECOND

    placeholder = _audio_placeholder_for(torch.zeros(num_embeds, HIDDEN_SIZE))

    assert placeholder.count("<unk>") == num_embeds


def test_placeholder_does_not_scale_with_hidden_size() -> None:
    """The count must depend on ``s`` alone, never on ``h``.

    ``sum(map(len, ...))`` returned ``s * h``, so the placeholder grew with the model's
    hidden size. Two items with the same ``s`` but different ``h`` must agree.
    """
    narrow = _audio_placeholder_for(torch.zeros(EMBEDS_PER_SECOND, 8))
    wide = _audio_placeholder_for(torch.zeros(EMBEDS_PER_SECOND, HIDDEN_SIZE))

    assert narrow.count("<unk>") == wide.count("<unk>") == EMBEDS_PER_SECOND
