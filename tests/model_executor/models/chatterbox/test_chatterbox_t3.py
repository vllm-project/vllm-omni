# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage 0's prompt layout, hooks and checkpoint loading, on the stages the engine builds.

The forward pass and Model Runner V2's own steps need a GPU; they are in
``test_chatterbox_gpu.py``.
"""

import copy
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from tests.model_executor.models.chatterbox.helpers import (
    DEPLOY,
    build_stage,
    load_stage_weights,
    random_voice,
    stage_vllm_config,
    t3_checkpoint,
    write_model_dir,
    write_t3_checkpoint,
)
from vllm_omni.model_executor.models.chatterbox.chatterbox import ChatterboxForConditionalGeneration
from vllm_omni.model_executor.models.chatterbox.chatterbox_t3 import (
    HEAD_KEYS,
    ChatterboxT3ForConditionalGeneration,
    T3Heads,
    prefill_embeds,
    speech_logits,
    split_t3_weights,
)
from vllm_omni.model_executor.models.chatterbox.conditioning import build_prompt
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(scope="module")
def talker(
    built_stage: Callable[[str, int], ChatterboxForConditionalGeneration],
) -> ChatterboxT3ForConditionalGeneration:
    """Turbo's stage 0 as the engine builds it from the deploy file, in its bfloat16."""
    return built_stage("turbo", 0).model


@pytest.fixture(scope="module")
def heads(talker: ChatterboxT3ForConditionalGeneration) -> T3Heads:
    return talker.heads


@pytest.fixture(scope="module")
def config(talker: ChatterboxT3ForConditionalGeneration) -> ChatterboxConfig:
    return talker.config


@pytest.mark.parametrize("cond_len", [150, 375])
def test_prefill_layout_follows_the_conditioning_length(
    heads: T3Heads, config: ChatterboxConfig, cond_len: int
) -> None:
    """Speaker slot, the prompt tokens, the text, one start-of-speech slot."""
    text_ids = torch.tensor([5, 6, 7])
    cond_tokens = torch.randint(0, 6561, (cond_len,))
    speaker = torch.randn(1, 256, dtype=heads.spkr_enc.weight.dtype)

    embeds = prefill_embeds(heads, text_ids, cond_tokens, speaker, config.start_speech_token)

    assert embeds.shape == (1 + cond_len + 3 + 1, config.hidden_size)
    assert torch.equal(embeds[0], heads.spkr_enc(speaker)[0])
    assert torch.equal(embeds[1 : 1 + cond_len], heads.speech_emb(cond_tokens))
    assert torch.equal(embeds[1 + cond_len : -1], heads.text_emb(text_ids))
    assert torch.equal(embeds[-1], heads.speech_emb.weight[config.start_speech_token])


# The prompt below is 1 + 375 + 3 + 1 = 380 rows long.
@pytest.mark.parametrize(
    ("num_computed", "span"),
    [(0, 380), (0, 379), (379, 1), (0, 382), (377, 5), (10, 372)],
    ids=[
        "whole prompt",
        "all but the last row",
        "one-row tail",
        "recomputed past the prompt",
        "across the end",
        "middle",
    ],
)
def test_preprocess_embeds_each_scheduled_span_as_one_call_would(
    talker: ChatterboxT3ForConditionalGeneration,
    heads: T3Heads,
    config: ChatterboxConfig,
    num_computed: int,
    span: int,
) -> None:
    """However the scheduler splits or recomputes the prompt, a span gets exactly its own rows.

    Rows inside the prompt are the prompt's embedding; rows past it are the
    speech ids a preempted request had generated, embedded as a decode step
    would. The voice arrives in float32 and the stage runs in bfloat16.
    """
    torch.manual_seed(0)
    text_ids = [5, 6, 7]
    cond_tokens = torch.randint(0, 6561, (375,))
    speaker = torch.randn(1, 256)
    whole = prefill_embeds(
        heads, torch.tensor(text_ids), cond_tokens, speaker.to(heads.spkr_enc.weight.dtype), config.start_speech_token
    )
    total = whole.shape[0]
    assert total == 380
    generated = torch.tensor([11, 12])
    every_id = torch.cat([torch.full((total,), config.start_speech_token), generated])

    _, embeds, update = talker.preprocess(
        every_id[num_computed : num_computed + span],
        None,
        _omni_is_prefill=True,
        _omni_prompt_len=total,
        _omni_num_computed_tokens=num_computed,
        ids={"prompt": text_ids, "speech_token": cond_tokens.tolist()},
        embed={"voice": speaker},
    )

    assert embeds.shape == (span, config.hidden_size)
    assert torch.equal(embeds, torch.cat([whole, heads.speech_emb(generated)])[num_computed : num_computed + span])
    assert update == {}


@pytest.mark.parametrize("text_len", [1, 3, 671])
@pytest.mark.parametrize("cond_len", [125, 150, 375])
def test_the_placeholder_prompt_is_as_long_as_its_embeddings(
    heads: T3Heads, config: ChatterboxConfig, cond_len: int, text_len: int
) -> None:
    """``build_prompt`` counts the placeholders, ``prefill_embeds`` builds what replaces them.

    A longer placeholder span would be filled past the prompt with the
    start token's embedding, a shorter one would drop the start-of-speech
    slot; either ends generation wrongly with no error. Nothing checks it
    at run time, so the two are pinned to each other here, through the
    request payload ``preprocess`` is handed.
    """
    conditioning = random_voice(cond_len, 250)
    prompt = build_prompt(list(range(text_len)), conditioning, config)
    info = prompt["additional_information"]

    embeds = prefill_embeds(
        heads,
        torch.tensor(info["ids"]["prompt"]),
        torch.tensor(info["ids"]["speech_token"]),
        info["embed"]["voice"].to(heads.spkr_enc.weight.dtype),
        config.start_speech_token,
    )

    assert len(prompt["prompt_token_ids"]) == embeds.shape[0]


def test_batched_decode_embedding_matches_per_request_preprocess(
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
    # Both are the speech table's rows: a decode row is a sampled speech id.
    assert torch.equal(embeds, talker.heads.speech_emb.weight[input_ids])
    assert updates == [update for _, _, update in each] == [{}, {}, {}]


@pytest.mark.parametrize("padded_rows", [7, 16], ids=["as scheduled", "padded by the graph"])
def test_every_token_row_gets_a_payload_flag_and_nothing_else(
    talker: ChatterboxT3ForConditionalGeneration, config: ChatterboxConfig, padded_rows: int
) -> None:
    """The runner-side transport calls the chunk processor only for a request the step has a payload for.

    The runner cuts the payload per request by token rows, so it is one
    false flag per row, CUDA-graph padding included, on the CPU, and no
    client output.
    """
    hidden = torch.randn(padded_rows, config.hidden_size)

    output = talker.make_omni_output(hidden, model_intermediate_buffer=[{}, {}, {}])

    assert output.text_hidden_states is hidden
    assert list(output.multimodal_outputs) == ["meta"] and list(output.multimodal_outputs["meta"]) == [
        "codec_frame_valid"
    ]
    flags = output.multimodal_outputs["meta"]["codec_frame_valid"]
    assert flags.device.type == "cpu" and flags.tolist() == [False] * padded_rows


def test_logits_are_the_speech_heads_with_the_start_token_masked(heads: T3Heads, config: ChatterboxConfig) -> None:
    """The sampler's vocabulary is the speech head's; the text vocabulary only sizes the text embedding."""
    logits = speech_logits(heads, torch.randn(2, config.hidden_size, dtype=heads.speech_head.weight.dtype), config)

    assert logits.shape == (2, config.vocab_size) == (2, heads.speech_head.out_features)
    assert heads.text_emb.num_embeddings == config.text_vocab_size
    assert torch.isfinite(logits[:, : config.start_speech_token]).all()
    assert (logits[:, config.start_speech_token] == float("-inf")).all()
    assert torch.isfinite(logits[:, config.stop_speech_token]).all()


def test_the_key_map_fills_every_head_parameter_and_nothing_else(heads: T3Heads) -> None:
    assert sorted(HEAD_KEYS.values()) == sorted(heads.state_dict())


def test_weight_routing_sends_the_backbone_to_vllm_and_the_rest_to_the_heads(
    talker: ChatterboxT3ForConditionalGeneration,
) -> None:
    """Every head tensor reaches its own parameter, by value; the backbone keeps its own, in file order."""
    checkpoint = t3_checkpoint(talker, seed=1)

    backbone, routed = split_t3_weights(iter(checkpoint.items()))

    assert [name for name, _ in backbone] == [
        name.removeprefix("tfmr.") for name in checkpoint if name.startswith("tfmr.")
    ]
    assert backbone[0][0] == "wte.weight" and backbone[-1][0] == "ln_f.bias"
    assert all(tensor is checkpoint[f"tfmr.{name}"] for name, tensor in backbone)
    heads = copy.deepcopy(talker.heads)
    heads.load_state_dict(routed, strict=True)
    assert torch.equal(heads.speech_head.bias, checkpoint["speech_head.bias"])
    assert torch.equal(heads.spkr_enc.weight, checkpoint["cond_enc.spkr_enc.weight"])
    assert all(torch.equal(heads.state_dict()[parameter], checkpoint[name]) for name, parameter in HEAD_KEYS.items())


def test_weight_routing_rejects_a_key_it_does_not_know() -> None:
    with pytest.raises(KeyError, match="cond_enc.perceiver.weight"):
        split_t3_weights(iter([("cond_enc.perceiver.weight", torch.zeros(1))]))


@pytest.mark.parametrize("variant", ["turbo", "original"])
def test_prefix_caching_is_refused_at_startup(model_dir: str, variant: str) -> None:
    """Every prompt is the same placeholder id, so a cache would match on length alone."""
    config = stage_vllm_config(model_dir, DEPLOY[variant], 0, enable_prefix_caching=True)
    assert config.cache_config.enable_prefix_caching is True

    with pytest.raises(ValueError, match="enable_prefix_caching=False"):
        build_stage(config, "cpu")


@pytest.mark.usefixtures("process_groups")
@pytest.mark.parametrize("variant", ["turbo", "original"])
def test_load_weights_fills_the_heads_and_names_every_parameter(tmp_path: Path, variant: str) -> None:
    """vLLM's loader fails the load on a parameter the returned set does not name.

    The file is in the released layout: transformers' backbone under
    ``tfmr.``, the heads under the checkpoint's names, and the training-only
    ``text_head``. It loads through vLLM's loader into a stage built from the
    deploy file; the heads hold the file's values and so does the backbone,
    in vLLM's own layout.
    """
    model_dir = write_model_dir(tmp_path)
    config = stage_vllm_config(model_dir, DEPLOY[variant], 0)
    stage = build_stage(config, "cpu")
    checkpoint = write_t3_checkpoint(model_dir, stage.model, seed=2)

    load_stage_weights(stage, config)

    parameters = dict(stage.named_parameters())
    heads = stage.model.heads.state_dict()
    if variant == "turbo":
        assert all(torch.equal(heads[parameter], checkpoint[name]) for name, parameter in HEAD_KEYS.items())
        rows = checkpoint["tfmr.wte.weight"].shape[0]
        assert torch.equal(parameters["model.tfmr.wte.weight"][:rows], checkpoint["tfmr.wte.weight"])
        # GPT-2's Conv1D stores (in, out); vLLM's linear layers (out, in).
        assert torch.equal(parameters["model.tfmr.h.0.attn.c_attn.weight"], checkpoint["tfmr.h.0.attn.c_attn.weight"].T)
    else:
        assert all(torch.equal(tensor, checkpoint[name]) for name, tensor in heads.items())
        qkv = [checkpoint[f"tfmr.layers.0.self_attn.{part}_proj.weight"] for part in "qkv"]
        assert torch.equal(parameters["model.tfmr.layers.0.self_attn.qkv_proj.weight"], torch.cat(qkv))
    assert stage.load_weights(iter(checkpoint.items())) == set(parameters)
    assert all(name.startswith("model.") for name in parameters)


def test_preprocess_embeds_in_the_heads_dtype_whatever_the_voices(
    talker: ChatterboxT3ForConditionalGeneration, config: ChatterboxConfig
) -> None:
    """Stage 0 runs in bfloat16 while the request's voice arrives as float32."""
    torch.manual_seed(3)
    speaker = torch.randn(1, 256)
    cond_tokens = [1, 2, 3]
    total = 1 + len(cond_tokens) + 2 + 1

    _, embeds, _ = talker.preprocess(
        torch.full((total,), config.start_speech_token),
        None,
        _omni_is_prefill=True,
        _omni_prompt_len=total,
        _omni_num_computed_tokens=0,
        ids={"prompt": [5, 6], "speech_token": cond_tokens},
        embed={"voice": speaker},
    )

    assert talker.heads.text_emb.weight.dtype == embeds.dtype == torch.bfloat16
    assert embeds.shape == (total, config.hidden_size)
    assert torch.equal(embeds[0], talker.heads.spkr_enc(speaker.to(torch.bfloat16))[0])
