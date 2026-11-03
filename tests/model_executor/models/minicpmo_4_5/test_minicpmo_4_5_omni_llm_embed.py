# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_embed_multimodal_delegates_to_get_multimodal_embeddings(monkeypatch):
    """vLLM V1 encoder profiling calls embed_multimodal; without an override the
    SupportsMultiModal protocol stub returns None and startup fails downstream."""
    sentinel = (object(),)
    seen: dict[str, object] = {}

    def fake_get_multimodal_embeddings(self, **kwargs):
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        MiniCPMO45OmniLLMForConditionalGeneration,
        "get_multimodal_embeddings",
        fake_get_multimodal_embeddings,
    )
    model = object.__new__(MiniCPMO45OmniLLMForConditionalGeneration)

    out = model.embed_multimodal(pixel_values="pv", audio_features="af")

    assert out is sentinel
    assert seen == {"pixel_values": "pv", "audio_features": "af"}
