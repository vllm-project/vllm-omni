# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.chatterbox.chatterbox_t3 import (
    ChatterboxT3ForConditionalGeneration,
    T3Heads,
    prefill_embeds,
    prefill_slice,
    speech_logits,
    split_t3_weights,
)
from vllm_omni.model_executor.models.chatterbox.conditioning import VoiceConditioning, build_prompt
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


@pytest.fixture(scope="module")
def talker(heads: T3Heads, config: ChatterboxConfig) -> ChatterboxT3ForConditionalGeneration:
    """The stage without its backbone: __init__ needs a vLLM config, the embedding hooks only these two."""
    model = object.__new__(ChatterboxT3ForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config, model.heads = config, heads
    return model


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

    head = prefill_slice(heads, config, text_ids, cond_tokens, speaker, 0, total - 1)
    tail = prefill_slice(heads, config, text_ids, cond_tokens, speaker, total - 1, 1)

    assert torch.equal(torch.cat([head, tail]), whole)
    # The one-token tail is the prompt's last row, not the embedding a decode
    # step would give the same placeholder id.
    assert torch.equal(tail[0], heads.speech_emb.weight[config.start_speech_token])


def test_prefill_span_past_the_prompt_returns_only_the_prompt_rows_001(
    heads: T3Heads, config: ChatterboxConfig
) -> None:
    """A preempted request is recomputed from zero with its generated tokens in the span."""
    text_ids = torch.tensor([5, 6, 7])
    cond_tokens = torch.randint(0, 6561, (375,))
    speaker = torch.randn(1, 256)
    whole = prefill_embeds(heads, text_ids, cond_tokens, speaker, config.start_speech_token)
    total = whole.shape[0]

    from_zero = prefill_slice(heads, config, text_ids, cond_tokens, speaker, 0, total + 2)
    inside = prefill_slice(heads, config, text_ids, cond_tokens, speaker, total - 3, 5)

    assert torch.equal(from_zero, whole)
    assert torch.equal(inside, whole[-3:])


def test_preprocess_embeds_the_generated_tokens_after_the_prompt_001(
    talker: ChatterboxT3ForConditionalGeneration, heads: T3Heads, config: ChatterboxConfig
) -> None:
    """The rows past the prompt are speech ids, embedded as a decode step would."""
    text_ids = [5, 6, 7]
    cond_tokens = torch.randint(0, 6561, (375,))
    speaker = torch.randn(1, 256)
    whole = prefill_embeds(heads, torch.tensor(text_ids), cond_tokens, speaker, config.start_speech_token)
    total = whole.shape[0]
    generated = torch.tensor([11, 12])
    input_ids = torch.cat([torch.full((total,), config.start_speech_token), generated])

    _, embeds, _ = talker.preprocess(
        input_ids,
        None,
        _omni_is_prefill=True,
        _omni_prompt_len=total,
        _omni_num_computed_tokens=0,
        ids={"prompt": text_ids, "speech_token": cond_tokens.tolist()},
        embed={"voice": speaker},
    )

    assert embeds.shape == (total + 2, config.hidden_size)
    assert torch.equal(embeds[:total], whole)
    assert torch.equal(embeds[total:], heads.speech_emb(generated))


@pytest.mark.parametrize("text_len", [1, 3, 671])
@pytest.mark.parametrize("cond_len", [125, 150, 375])
def test_the_placeholder_prompt_is_as_long_as_its_embeddings_001(
    heads: T3Heads, config: ChatterboxConfig, cond_len: int, text_len: int
) -> None:
    """``build_prompt`` counts the placeholders, ``prefill_embeds`` builds what replaces them.

    A longer placeholder span would be filled past the prompt with the
    start token's embedding, a shorter one would drop the start-of-speech
    slot; either ends generation wrongly with no error. Nothing checks it
    at run time, so the two are pinned to each other here, through the
    request payload ``preprocess`` is handed.
    """
    conditioning = VoiceConditioning(
        cond_tokens=torch.randint(0, 6561, (1, cond_len)),
        speaker_emb=torch.randn(1, 256),
        prompt_token=torch.randint(0, 6561, (1, 250)),
        prompt_feat=torch.randn(1, 500, 80),
        embedding=torch.randn(1, 192),
    )
    prompt = build_prompt(list(range(text_len)), conditioning, config)
    info = prompt["additional_information"]

    embeds = prefill_embeds(
        heads,
        torch.tensor(info["ids"]["prompt"]),
        torch.tensor(info["ids"]["speech_token"]),
        info["embed"]["voice"],
        config.start_speech_token,
    )

    assert len(prompt["prompt_token_ids"]) == embeds.shape[0]


def test_batched_decode_embedding_matches_per_request_preprocess_001(
    talker: ChatterboxT3ForConditionalGeneration, config: ChatterboxConfig
) -> None:
    """The runner hands a step's one-token decode rows to one call instead of one call each."""
    input_ids = torch.tensor([11, 6560, 0])
    req_infos = [
        {
            "_omni_is_prefill": False,
            "_omni_prompt_len": 381,
            "_omni_num_computed_tokens": 381 + row,
            "ids": {},
            "embed": {},
        }
        for row in range(3)
    ]

    ids, embeds, updates = talker.preprocess_decode_batch(input_ids=input_ids, req_infos=req_infos)

    each = [talker.preprocess(input_ids[row : row + 1], None, **info) for row, info in enumerate(req_infos)]
    assert ids is input_ids
    assert embeds.shape == (3, config.hidden_size)
    assert torch.equal(embeds, torch.cat([row_embeds for _, row_embeds, _ in each]))
    assert updates == [update for _, _, update in each] == [{}, {}, {}]


def test_logits_are_the_speech_heads_with_the_start_token_masked_001(heads: T3Heads, config: ChatterboxConfig) -> None:
    """The sampler's vocabulary is the speech head's; the text vocabulary only sizes the text embedding."""
    logits = speech_logits(heads, torch.randn(2, config.hidden_size), config)

    assert logits.shape == (2, config.vocab_size) == (2, heads.speech_head.out_features)
    assert heads.text_emb.num_embeddings == config.text_vocab_size
    assert torch.isfinite(logits[:, : config.start_speech_token]).all()
    assert (logits[:, config.start_speech_token] == float("-inf")).all()
    assert torch.isfinite(logits[:, config.stop_speech_token]).all()


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


def test_prefix_caching_is_refused_at_startup_001() -> None:
    """Every prompt is the same placeholder id, so a cache would match on length alone."""
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(enable_prefix_caching=True))
    with pytest.raises(ValueError, match="enable_prefix_caching=False"):
        ChatterboxT3ForConditionalGeneration(vllm_config=vllm_config)
