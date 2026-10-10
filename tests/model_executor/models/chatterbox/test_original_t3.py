# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Original scheduled prompt reconstruction and guidance request metadata."""

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.chatterbox.chatterbox_original_t3 import ChatterboxOriginalT3
from vllm_omni.model_executor.models.chatterbox.conditioning import VoiceConditioning, build_prompt
from vllm_omni.model_executor.models.chatterbox.original_heads import (
    OriginalHeads,
    original_prefill_embeds,
    original_speech_embeds,
)
from vllm_omni.model_executor.stage_input_processors.chatterbox import expand_original_cfg_prompts
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def model():
    config = ChatterboxConfig("original")
    config.hidden_size = 8
    talker = object.__new__(ChatterboxOriginalT3)
    nn.Module.__init__(talker)
    talker.config = config
    talker.heads = OriginalHeads(config).eval()
    talker.cfg_pairs = {}
    return talker


@pytest.fixture
def metadata():
    return {
        "_omni_prompt_len": 41,
        "ids": {"prompt": [12, 13, 14], "speech_token": [1, 2, 3, 4]},
        "embed": {"voice": torch.ones(1, 256)},
        "req_id": "engine-request",
        "global_request_id": ["external-request"],
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
        metadata["embed"]["voice"],
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


def test_cfg_registration_uses_external_pair_id_and_cleans_finished_requests(model, metadata):
    for role, request_id, external_id in (
        ("cond", "internal-a", "external-request"),
        ("uncond", "internal-b", "external-request__cfg_uncond"),
    ):
        info = {**metadata, "req_id": request_id, "global_request_id": [external_id], "cfg_group": {"role": role}}
        model.preprocess(torch.tensor([8]), None, _omni_is_prefill=False, _omni_num_computed_tokens=41, **info)
        assert model.cfg_pairs[request_id] == ("external-request", role, 0.5)
    model.on_requests_finished({"internal-a", "already-finished"})
    assert set(model.cfg_pairs) == {"internal-b"}
    model.on_requests_finished({"internal-b"})
    assert model.cfg_pairs == {}


def test_prompt_length_mismatch_fails(model, metadata):
    with pytest.raises(ValueError, match="placeholder length"):
        model.preprocess(
            torch.tensor([6561]),
            None,
            _omni_is_prefill=True,
            _omni_num_computed_tokens=0,
            **{**metadata, "_omni_prompt_len": 42},
        )


@pytest.mark.parametrize("weight", [0.0, 0.5])
def test_original_prompt_and_companion_metadata(weight):
    config = ChatterboxConfig("original")
    voice = VoiceConditioning(
        cond_tokens=torch.tensor([[1, 2]]),
        speaker_emb=torch.ones(1, 256),
        prompt_token=torch.tensor([[3, 4]]),
        prompt_feat=torch.zeros(1, 4, 80),
        embedding=torch.ones(1, 192),
    )
    prompt = build_prompt([12, 13, 14], voice, config, exaggeration=0.7, cfg_weight=weight)
    assert prompt["prompt_token_ids"] == [6561] * 41
    info = prompt["additional_information"]
    assert info["chatterbox"] == {"cfg_weight": weight, "exaggeration": 0.7}
    expanded = expand_original_cfg_prompts(prompt, None)
    if weight == 0:
        assert "cfg_group" not in info
        assert expanded == []
    else:
        assert info["cfg_group"]["role"] == "cond"
        assert len(expanded) == 1
        assert expanded[0].request_id_suffix == "__cfg_uncond"
        companion = expanded[0].prompt
        assert companion["prompt_token_ids"] == prompt["prompt_token_ids"]
        assert companion["additional_information"]["cfg_group"]["role"] == "uncond"
        assert companion["additional_information"]["ids"] == info["ids"]
        assert info["cfg_group"]["role"] == "cond"
    assert config.speech_vocab_size == 8194
    assert config.text_vocab_size == 704
    assert config.enc_cond_seconds == 6
    assert config.n_cfm_timesteps == 10
    assert config.n_silence_tokens == 0
