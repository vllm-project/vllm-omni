# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch

from vllm_omni.model_executor.models.chatterbox.chatterbox_t3 import (
    T3Heads,
    prefill_embeds,
    prefill_slice,
    speech_logits,
    split_t3_weights,
)
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(scope="module")
def config() -> ChatterboxConfig:
    cfg = ChatterboxConfig()
    cfg.hidden_size = 16  # the layout does not depend on width; this keeps the test small
    return cfg


@pytest.fixture(scope="module")
def heads(config: ChatterboxConfig) -> T3Heads:
    torch.manual_seed(0)
    return T3Heads(config)


@pytest.mark.parametrize("cond_len", [150, 375])
def test_prefill_layout_follows_the_conditioning_length_001(
    heads: T3Heads, config: ChatterboxConfig, cond_len: int
) -> None:
    """Speaker slot, the prompt tokens, the text, one start-of-speech slot."""
    text_ids = torch.tensor([5, 6, 7])
    cond_tokens = torch.randint(0, 6561, (cond_len,))
    speaker = torch.randn(1, 256)

    embeds = prefill_embeds(heads, text_ids, cond_tokens, speaker, config.start_speech_token)

    assert embeds.shape == (1 + cond_len + 3 + 1, config.hidden_size)
    assert torch.equal(embeds[0], heads.spkr_enc(speaker)[0])
    assert torch.equal(embeds[1 : 1 + cond_len], heads.speech_emb(cond_tokens))
    assert torch.equal(embeds[1 + cond_len : -1], heads.text_emb(text_ids))
    assert torch.equal(embeds[-1], heads.speech_emb.weight[config.start_speech_token])


def test_prefill_split_with_one_token_tail_matches_one_call_001(heads: T3Heads, config: ChatterboxConfig) -> None:
    """However the scheduler splits the prompt, each span gets its own rows."""
    text_ids = torch.tensor([5, 6, 7])
    cond_tokens = torch.randint(0, 6561, (375,))
    speaker = torch.randn(1, 256)
    whole = prefill_embeds(heads, text_ids, cond_tokens, speaker, config.start_speech_token)
    total = whole.shape[0]

    head = prefill_slice(heads, config, text_ids, cond_tokens, speaker, total, 0, total - 1)
    tail = prefill_slice(heads, config, text_ids, cond_tokens, speaker, total, total - 1, 1)

    assert torch.equal(torch.cat([head, tail]), whole)
    # The one-token tail is the prompt's last row, not the embedding a decode
    # step would give the same placeholder id.
    assert torch.equal(tail[0], heads.speech_emb.weight[config.start_speech_token])


def test_prefill_refuses_a_prompt_of_the_wrong_length_001(heads: T3Heads, config: ChatterboxConfig) -> None:
    """A placeholder span longer than the embeddings would zero-pad the prompt."""
    text_ids = torch.tensor([5, 6, 7])
    cond_tokens = torch.randint(0, 6561, (375,))
    with pytest.raises(RuntimeError, match="prompt of 381 tokens"):
        prefill_slice(heads, config, text_ids, cond_tokens, torch.randn(1, 256), 381, 0, 381)


def test_logits_are_padded_to_the_text_vocab_with_the_start_token_masked_001(
    heads: T3Heads, config: ChatterboxConfig
) -> None:
    """The sampler is sized by the tokenizer's vocabulary; only speech ids are live."""
    logits = speech_logits(heads, torch.randn(2, config.hidden_size), config)

    assert logits.shape == (2, config.vocab_size)
    assert torch.isfinite(logits[:, : config.start_speech_token]).all()
    assert torch.isinf(logits[:, config.start_speech_token]).all()
    assert torch.isfinite(logits[:, config.stop_speech_token]).all()
    assert torch.isinf(logits[:, config.speech_vocab_size :]).all()


def test_weight_routing_sends_the_backbone_to_vllm_and_the_rest_to_the_heads_001() -> None:
    weights = [
        ("tfmr.h.0.attn.c_attn.weight", torch.zeros(1)),
        ("tfmr.wte.weight", torch.zeros(1)),
        ("tfmr.wpe.weight", torch.zeros(1)),
        ("text_head.weight", torch.zeros(1)),
        ("text_emb.weight", torch.zeros(1)),
        ("speech_emb.weight", torch.zeros(1)),
        ("speech_head.weight", torch.zeros(1)),
        ("speech_head.bias", torch.zeros(1)),
        ("cond_enc.spkr_enc.weight", torch.zeros(1)),
        ("cond_enc.spkr_enc.bias", torch.zeros(1)),
    ]

    backbone, heads = split_t3_weights(iter(weights))

    assert [name for name, _ in backbone] == ["h.0.attn.c_attn.weight", "wte.weight", "wpe.weight"]
    assert set(heads) == {
        "text_emb.weight",
        "speech_emb.weight",
        "speech_head.weight",
        "speech_head.bias",
        "spkr_enc.weight",
        "spkr_enc.bias",
    }


def test_weight_routing_rejects_a_key_it_does_not_know_001() -> None:
    with pytest.raises(KeyError, match="cond_enc.perceiver.weight"):
        split_t3_weights(iter([("cond_enc.perceiver.weight", torch.zeros(1))]))
