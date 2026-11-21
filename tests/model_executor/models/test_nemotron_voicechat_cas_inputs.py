# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Character packing preserves valid-token order, masks and exact padding."""

import pytest
import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput

from vllm_omni.model_executor.models.nemotron_voicechat.nemo_vendored import ear_tts_model

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Backbone(nn.Module):
    def forward(self, *, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> BaseModelOutput:
        assert attention_mask.shape == inputs_embeds.shape[:2]
        return BaseModelOutput(last_hidden_state=inputs_embeds)


@pytest.fixture
def encoder() -> ear_tts_model.CharAwareSubwordEncoder:
    model = ear_tts_model.CharAwareSubwordEncoder.__new__(ear_tts_model.CharAwareSubwordEncoder)
    nn.Module.__init__(model)
    model.subword_id_to_char_ids = {1: (2, 3), 2: (4,), 3: ()}
    model.char_padding_idx = 5
    model.embed_tokens = nn.Embedding.from_pretrained(torch.arange(12, dtype=torch.float32).reshape(6, 2))
    model.backbone = _Backbone()
    model.proj_embedding = nn.Identity()
    model.use_subword_flag_emb = False
    model.use_bos_eos_emb = False
    return model


def test_packing_preserves_masked_order_and_unknown_tokens(encoder) -> None:
    ids = torch.tensor([[1, 2, 99], [2, 1, 3]])
    mask = torch.tensor([[True, False, True], [True, True, True]])
    chars, lengths = encoder.prepare_inputs(ids, mask)
    assert torch.equal(chars, torch.tensor([[2, 3], [5, 5], [4, 5], [2, 3], [5, 5]]))
    assert torch.equal(lengths, torch.tensor([2, 0, 1, 2, 0]))
    assert chars.dtype == lengths.dtype == torch.long
    assert chars.device == lengths.device == ids.device


@pytest.mark.parametrize("ids", [torch.tensor([[1, 2]]), torch.empty((0, 0), dtype=torch.long)])
def test_empty_selection_retains_two_dimensional_character_tensor(encoder, ids) -> None:
    chars, lengths = encoder.prepare_inputs(ids, torch.zeros_like(ids, dtype=torch.bool))
    assert chars.shape == (0, 0)
    assert lengths.shape == (0,)


def test_tokens_without_characters_keep_zero_width(encoder) -> None:
    ids = torch.tensor([[3, 99]])
    chars, lengths = encoder.prepare_inputs(ids, torch.ones_like(ids, dtype=torch.bool))
    assert chars.shape == (2, 0)
    assert torch.equal(lengths, torch.zeros(2, dtype=torch.long))


def test_packing_uses_host_lengths_without_tensor_item(encoder, monkeypatch) -> None:
    def unexpected_item(*args, **kwargs):
        raise AssertionError("host-known character lengths must not be read back from a tensor")

    monkeypatch.setattr(torch.Tensor, "item", unexpected_item)
    chars, lengths = encoder.prepare_inputs(torch.tensor([[1, 2, 3]]), torch.ones((1, 3), dtype=torch.bool))
    assert chars.shape == (3, 2)
    assert lengths.shape == (3,)


def test_forward_reuses_packed_width_and_scatter_mask(encoder, monkeypatch) -> None:
    original = ear_tts_model.sequence_mask

    def mask_with_host_width(lengths, max_length=None):
        assert isinstance(max_length, int)
        return original(lengths, max_length)

    monkeypatch.setattr(ear_tts_model, "sequence_mask", mask_with_host_width)
    ids = torch.tensor([[1, 2], [2, 1]])
    mask = torch.tensor([[True, False], [True, True]])
    actual = encoder(ids, mask)
    assert torch.equal(actual, torch.tensor([[[5.0, 6.0], [0.0, 0.0]], [[8.0, 9.0], [5.0, 6.0]]]))
