# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LongCat-Next audio encoding: _encode_audios regrouping per audio item."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.longcat_next.modeling_longcat_next import (
    LongcatNextForCausalLM,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

HIDDEN = 4


def _encode_by_row(features, encoder_lengths, bridge_lengths):
    return torch.arange(int(bridge_lengths.sum()), dtype=torch.long).unsqueeze(1)


def _model(encode=_encode_by_row):
    return SimpleNamespace(
        dtype=torch.float32,
        audio_offset_vals=torch.zeros(1, dtype=torch.long),
        audio_tokenizer=SimpleNamespace(encode=encode),
        _code_embeddings=lambda ids: ids.to(torch.float32).expand(-1, HIDDEN),
    )


def _rows(start: int, stop: int) -> torch.Tensor:
    return torch.arange(start, stop, dtype=torch.float32).unsqueeze(1).expand(stop - start, HIDDEN)


def test_multi_chunk_clips_regroup_to_one_embedding_per_clip():
    model = _model()

    out = LongcatNextForCausalLM._encode_audios(
        model,
        torch.zeros(3, 5),
        torch.tensor([6, 8, 10]),
        torch.tensor([3, 4, 5]),
        torch.tensor([2, 1]),
    )

    assert len(out) == 2
    assert out[0].shape == (7, HIDDEN)
    assert out[1].shape == (5, HIDDEN)
    assert torch.equal(out[0], _rows(0, 7))
    assert torch.equal(out[1], _rows(7, 12))


def test_no_chunk_counts_keeps_one_embedding_per_chunk():
    model = _model()

    out = LongcatNextForCausalLM._encode_audios(model, torch.zeros(2, 5), torch.tensor([4, 6]), torch.tensor([2, 3]))

    assert len(out) == 2
    assert out[0].shape == (2, HIDDEN)
    assert out[1].shape == (3, HIDDEN)
    assert torch.equal(out[0], _rows(0, 2))
    assert torch.equal(out[1], _rows(2, 5))


def test_skipped_clip_counts_zero_gets_empty_embedding():
    model = _model()

    out = LongcatNextForCausalLM._encode_audios(
        model,
        torch.zeros(3, 5),
        torch.tensor([4, 6, 8]),
        torch.tensor([2, 3, 4]),
        torch.tensor([1, 0, 2]),
    )

    assert len(out) == 3
    assert out[0].shape == (2, HIDDEN)
    assert out[1].shape == (0, HIDDEN)
    assert out[2].shape == (7, HIDDEN)
    assert torch.equal(out[2], _rows(2, 9))


def test_embed_multimodal_forwards_audio_chunk_counts():
    seen = {}

    def fake_encode_audios(features, encoder_lengths, bridge_lengths, chunk_counts=None):
        seen["chunk_counts"] = chunk_counts
        return []

    model = _model()
    model._encode_audios = fake_encode_audios
    counts = torch.tensor([2, 1])

    LongcatNextForCausalLM.embed_multimodal(
        model,
        audio_features=torch.zeros(3, 5),
        audio_encoder_lengths=torch.tensor([6, 8, 10]),
        audio_bridge_lengths=torch.tensor([3, 4, 5]),
        audio_chunk_counts=counts,
    )

    assert seen["chunk_counts"] is counts
