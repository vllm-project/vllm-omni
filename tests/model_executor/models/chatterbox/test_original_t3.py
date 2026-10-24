# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Original scheduled prompt reconstruction and guidance request metadata."""

from collections.abc import Callable, Iterator

import pytest
import torch
from vllm.sampling_params import SamplingParams

from tests.model_executor.models.chatterbox.helpers import DEPLOY, build_stage, random_voice, stage_vllm_config
from vllm_omni.model_executor.models.chatterbox.chatterbox import ChatterboxForConditionalGeneration
from vllm_omni.model_executor.models.chatterbox.chatterbox_original_t3 import ChatterboxOriginalT3
from vllm_omni.model_executor.models.chatterbox.conditioning import build_prompt
from vllm_omni.model_executor.models.chatterbox.original_heads import (
    ORIGINAL_SPEECH_POSITIONS,
    original_prefill_embeds,
    original_speech_embeds,
)
from vllm_omni.model_executor.stage_input_processors.chatterbox import (
    expand_original_cfg_prompts,
    prepare_original_prompt,
)
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def model(built_stage: Callable[[str, int], ChatterboxForConditionalGeneration]) -> Iterator[ChatterboxOriginalT3]:
    """Original's stage 0 as the engine builds it from the deploy file.

    The stage is shared by the module, so each test's guidance
    registrations are released afterwards the way the runner releases them.
    """
    talker = built_stage("original", 0).model
    yield talker
    talker.on_requests_finished(set(talker.cfg_pairs))


@pytest.fixture
def metadata():
    return {
        "_omni_prompt_len": 41,
        "ids": {"prompt": [12, 13, 14], "speech_token": [1, 2, 3, 4]},
        "embed": {"voice": torch.ones(1, 256)},
        "chatterbox": {"cfg_weight": 0.5, "exaggeration": 0.7},
    }


@pytest.mark.parametrize("offset,span", [(0, 41), (0, 19), (19, 21), (40, 1), (39, 5)])
def test_prefill_slices_and_recomputed_speech(model, metadata, offset, span):
    scheduled = torch.tensor([17] * span)
    _, actual, updates = model.preprocess(
        scheduled, None, _omni_is_prefill=True, _omni_num_computed_tokens=offset, **metadata
    )
    prompt = original_prefill_embeds(
        model.heads,
        torch.tensor(metadata["ids"]["prompt"]),
        torch.tensor(metadata["ids"]["speech_token"]),
        # The voice arrives in float32; the stage runs in the deploy file's bfloat16.
        metadata["embed"]["voice"].to(model.heads.text_emb.weight.dtype),
        exaggeration=0.7,
    )
    full = torch.cat((prompt, original_speech_embeds(model.heads, torch.tensor([17] * 3), torch.arange(1, 4))))
    torch.testing.assert_close(actual, full[offset : offset + span])
    assert updates == {}


def test_decode_batches_preserve_independent_positions(model):
    ids = torch.tensor([8, 9, 10])
    infos = [
        {"_omni_num_computed_tokens": 41, "_omni_prompt_len": 41},
        {"_omni_num_computed_tokens": 100, "_omni_prompt_len": 43},
        {"_omni_num_computed_tokens": 80, "_omni_prompt_len": 70},
    ]
    expected = original_speech_embeds(model.heads, ids, torch.tensor([1, 58, 11]))
    _, actual, updates = model.preprocess_decode_batch(input_ids=ids, req_infos=infos)
    torch.testing.assert_close(actual, expected)
    assert updates == [{}, {}, {}]
    _, actual_v2, codebooks, residuals, updates_v2 = model.preprocess_decode_batch_mrv2(
        input_ids=ids, input_embeds=torch.zeros_like(expected), req_infos=infos
    )
    torch.testing.assert_close(actual_v2, expected)
    assert codebooks.shape == residuals.shape == (3, 0)
    assert updates_v2 == updates


def test_model_runner_v2_registers_the_pair_its_sampling_parameters_carry(model, metadata):
    """MRv2 hands preprocess each request's sampling parameters and samples by row identity.

    The pair is the ``extra_args`` tuple admission tagged the request with,
    which the default runner hands its sampler directly; the default runner
    passes no sampling parameters, and nothing is registered for it.
    """
    for role, req_id in (("cond", "internal-a"), ("uncond", "internal-b")):
        params = SamplingParams(extra_args={"chatterbox_cfg": ("external-request", role, 0.5)})
        info = {**metadata, "cfg_group": {"role": role}, "sampling_params": params, "req_id": req_id}
        model.preprocess(torch.tensor([8]), None, _omni_is_prefill=False, _omni_num_computed_tokens=41, **info)
        assert model.cfg_pairs[req_id] == ("external-request", role, 0.5)
    model.on_requests_finished({"internal-a", "already-finished"})
    assert set(model.cfg_pairs) == {"internal-b"}
    model.on_requests_finished({"internal-b"})

    model.preprocess(torch.tensor([8]), None, _omni_is_prefill=False, _omni_num_computed_tokens=41, **metadata)
    assert model.cfg_pairs == {}


@pytest.mark.usefixtures("process_groups")
def test_a_context_beyond_the_speech_position_table_is_refused(model_dir: str) -> None:
    """A request's speech positions stay below its context; a longer context would index past the table."""
    context = ORIGINAL_SPEECH_POSITIONS + 1
    config = stage_vllm_config(
        model_dir, DEPLOY["original"], 0, max_model_len=context, max_num_batched_tokens=2 * context
    )
    with pytest.raises(ValueError, match=f"max_model_len up to {ORIGINAL_SPEECH_POSITIONS}"):
        build_stage(config, "cpu")


@pytest.mark.parametrize("text_len", [1, 3, 2048])
def test_the_placeholder_prompt_is_as_long_as_its_embeddings(model, text_len):
    """``build_prompt`` counts the placeholders, ``original_prefill_embeds`` builds what replaces them.

    Nothing checks the two agree at run time, so they are pinned to each
    other here, through the request payload ``preprocess`` is handed.
    """
    conditioning = random_voice(150, 250)
    prompt = build_prompt([5] * text_len, conditioning, model.config, exaggeration=0.5, cfg_weight=0.5)
    info = prompt["additional_information"]

    embeds = original_prefill_embeds(
        model.heads,
        torch.tensor(info["ids"]["prompt"]),
        torch.tensor(info["ids"]["speech_token"]),
        info["embed"]["voice"].to(model.heads.text_emb.weight.dtype),
        exaggeration=info["chatterbox"]["exaggeration"],
    )

    assert len(prompt["prompt_token_ids"]) == embeds.shape[0]


@pytest.mark.parametrize("weight", [0.0, 0.5])
def test_only_a_guided_original_prompt_is_expanded_with_an_unconditional_companion(weight):
    config = ChatterboxConfig("original")
    voice = random_voice(2, 2)
    prompt = build_prompt([12, 13, 14], voice, config, exaggeration=0.7, cfg_weight=weight)
    assert prompt["prompt_token_ids"] == [6561] * 41
    info = prompt["additional_information"]
    assert info["chatterbox"] == {"cfg_weight": weight, "exaggeration": 0.7}
    info["global_request_id"] = ["external-request"]
    params = SamplingParams(max_tokens=1000)
    prepare_original_prompt(prompt, [params])
    expanded = expand_original_cfg_prompts(prompt, params)
    if weight == 0:
        assert "cfg_group" not in info
        assert expanded == []
    else:
        assert info["cfg_group"]["role"] == "cond"
        assert len(expanded) == 1
        companion_params, _ = expanded[0].apply_overrides(params, [params])
        pair_key, role, cfg_weight = params.extra_args["chatterbox_cfg"]
        assert (pair_key, role, cfg_weight) == ("external-request", "cond", weight)
        assert companion_params.extra_args["chatterbox_cfg"] == (pair_key, "uncond", weight)
        assert expanded[0].request_id_suffix == "__cfg_uncond"
        companion = expanded[0].prompt
        assert companion["prompt_token_ids"] == prompt["prompt_token_ids"]
        assert companion["additional_information"]["cfg_group"]["role"] == "uncond"
        assert companion["additional_information"]["ids"] == info["ids"]
        assert info["cfg_group"]["role"] == "cond"
