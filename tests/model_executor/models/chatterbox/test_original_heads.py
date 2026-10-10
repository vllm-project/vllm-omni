# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Original prompt layout and strict checkpoint routing."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.chatterbox.original_heads import (
    OriginalHeads,
    original_prefill_embeds,
    original_speech_embeds,
    split_original_weights,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_original_prompt_layout_and_cfg():
    heads = OriginalHeads(
        SimpleNamespace(hidden_size=4, text_vocab_size=704, speech_vocab_size=8194, speaker_embed_size=2)
    )
    heads.eval()
    with torch.no_grad():
        for parameter in heads.parameters():
            parameter.zero_()
        heads.text_emb.weight.fill_(7)
        heads.speech_emb.weight.fill_(3)
        heads.text_pos_emb.emb.weight.copy_(torch.arange(2050)[:, None].tile(1, 4))
        heads.speech_pos_emb.emb.weight.copy_(torch.arange(4100)[:, None].tile(1, 4))
        heads.cond_enc.spkr_enc.weight.fill_(2)
        heads.cond_enc.emotion_adv_fc.weight.fill_(6)
    text, reference, speaker = torch.tensor([12, 13]), torch.tensor([1, 2, 3]), torch.tensor([[1.0, 2.0]])
    conditional = original_prefill_embeds(heads, text, reference, speaker)
    unconditional = original_prefill_embeds(heads, text, reference, speaker, unconditional=True)
    assert conditional.shape == (40, 4)
    torch.testing.assert_close(conditional[0], torch.full((4,), 6.0))
    torch.testing.assert_close(conditional[1:33], torch.zeros(32, 4))
    torch.testing.assert_close(conditional[33], torch.full((4,), 3.0))
    torch.testing.assert_close(conditional[34:38], (torch.arange(4) + 7.0)[:, None].tile(1, 4))
    torch.testing.assert_close(unconditional[34:38], torch.arange(4.0)[:, None].tile(1, 4))
    torch.testing.assert_close(conditional[-2:], torch.full((2, 4), 3.0))
    torch.testing.assert_close(conditional[:34], unconditional[:34])
    torch.testing.assert_close(conditional[-2:], unconditional[-2:])
    torch.testing.assert_close(
        original_speech_embeds(heads, torch.tensor([9, 10]), torch.tensor([1, 8])),
        torch.tensor([[4.0] * 4, [11.0] * 4]),
    )


def test_perceiver_matches_explicit_attention():
    torch.manual_seed(9)
    heads = OriginalHeads(
        SimpleNamespace(hidden_size=8, text_vocab_size=704, speech_vocab_size=8194, speaker_embed_size=2)
    )
    heads.eval()
    perceiver = heads.cond_enc.perceiver
    context = torch.randn(2, 7, 8)
    expected = perceiver.pre_attention_query.tile(2, 1, 1)
    attention = perceiver.attn
    for source in (context, None):
        source = expected if source is None else source
        q = attention.to_q(attention.norm(expected)).unflatten(-1, (4, 2)).transpose(1, 2)
        k = attention.to_k(attention.norm(source)).unflatten(-1, (4, 2)).transpose(1, 2)
        v = attention.to_v(attention.norm(source)).unflatten(-1, (4, 2)).transpose(1, 2)
        probabilities = torch.softmax(q @ k.transpose(-1, -2) / 2**0.5, dim=-1)
        expected = expected + attention.proj_out((probabilities @ v).transpose(1, 2).flatten(2))
    torch.testing.assert_close(perceiver(context), expected)


def test_original_weight_routes_are_complete_and_strict():
    heads = OriginalHeads(
        SimpleNamespace(hidden_size=4, text_vocab_size=704, speech_vocab_size=8194, speaker_embed_size=2)
    )
    state = heads.state_dict()
    marker = torch.zeros(1)
    backbone, routed = split_original_weights(
        [*state.items(), ("tfmr.layers.0.weight", marker), ("text_head.weight", marker)]
    )
    assert backbone == [("layers.0.weight", marker)]
    heads.load_state_dict(routed, strict=True)
    assert heads.speech_head.bias is None
    with pytest.raises(KeyError, match="unexpected key"):
        split_original_weights([("speech_head.bias", marker)])
