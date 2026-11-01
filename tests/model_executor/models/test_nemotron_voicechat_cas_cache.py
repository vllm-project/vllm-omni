# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Keep cached singleton embeddings separate from padding and session state."""

from collections import OrderedDict

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_talker import (
    NemotronVoiceChatTalkerForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _CharacterEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding.from_pretrained(torch.ones(1, 1))
        self.calls: list[tuple[int, ...]] = []

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert mask.shape == ids.shape and mask.dtype == torch.bool and bool(mask.all())
        self.calls.append(tuple(ids.reshape(-1).tolist()))
        # The cohort width affects the output, as real character padding can.
        weight = self.embed_tokens.weight[0]
        return (ids.to(weight.dtype) * weight + ids.shape[1]).unsqueeze(-1)


class _TTSModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_subword = _CharacterEncoder()


class _TTS(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tts_model = _TTSModel()


def _model() -> NemotronVoiceChatTalkerForConditionalGeneration:
    model = NemotronVoiceChatTalkerForConditionalGeneration.__new__(NemotronVoiceChatTalkerForConditionalGeneration)
    nn.Module.__init__(model)
    model.tts = _TTS().eval()
    model._cas_cache = OrderedDict()
    model.register_buffer("_cas_singleton_mask", torch.ones((1, 1), dtype=torch.bool), persistent=False)
    return model


def test_repeated_singleton_reuses_only_its_immutable_embedding() -> None:
    model = _model()
    first = model._encode_cas_rows(torch.tensor([7]))
    second = model._encode_cas_rows(torch.tensor([7]))
    assert torch.equal(first, second)
    assert model.tts.tts_model.embed_subword.calls == [(7,)]


def test_wider_cohorts_preserve_the_original_padding_geometry() -> None:
    model = _model()
    single = model._encode_cas_rows(torch.tensor([7]))
    cohort = model._encode_cas_rows(torch.tensor([7, 8]))
    model._encode_cas_rows(torch.tensor([7, 8]))
    assert not torch.equal(single[0], cohort[0])
    assert model.tts.tts_model.embed_subword.calls == [(7,), (7, 8), (7, 8)]


def test_singleton_cache_is_bounded_and_refreshes_recency() -> None:
    model = _model()
    for token in range(128):
        model._encode_cas_rows(torch.tensor([token]))
    model._encode_cas_rows(torch.tensor([0]))
    model._encode_cas_rows(torch.tensor([128]))
    assert len(model._cas_cache) == 128
    assert {key[0] for key in model._cas_cache} == set(range(129)) - {1}
    model._encode_cas_rows(torch.tensor([1]))
    assert model.tts.tts_model.embed_subword.calls[-1] == (1,)


def test_encoder_dtype_change_cannot_return_an_old_cached_row() -> None:
    model = _model()
    assert model._encode_cas_rows(torch.tensor([7])).dtype == torch.float32
    model.tts.to(dtype=torch.float64)
    assert model._encode_cas_rows(torch.tensor([7])).dtype == torch.float64
    assert model.tts.tts_model.embed_subword.calls == [(7,), (7,)]


def test_shared_embedding_is_copied_into_private_session_tables() -> None:
    model = _model()
    sessions = [
        {"timeline": torch.tensor([1]), "cas_table": torch.tensor([[2.0]])},
        {"timeline": torch.tensor([2]), "cas_table": torch.tensor([[3.0]])},
    ]
    for session in sessions:
        info = {"ids.all": torch.cat((session["timeline"], torch.tensor([7])))}
        model._refresh_native_timeline(session, info, torch.device("cpu"))
    assert model.tts.tts_model.embed_subword.calls == [(7,)]
    sessions[0]["cas_table"][-1] = 100
    assert sessions[1]["cas_table"][-1].item() == 8
    assert model._encode_cas_rows(torch.tensor([7])).item() == 8
