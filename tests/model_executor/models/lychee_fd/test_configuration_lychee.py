# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import copy

import pytest

from vllm_omni.model_executor.models.lychee_fd.configuration_lychee import LycheeFDConfig
from vllm_omni.model_executor.models.lychee_fd.contract import LycheeConfigError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_released_defaults_validate() -> None:
    config = LycheeFDConfig()
    contract = config.validate_lychee_contract()

    assert contract.layout.total_layers == 40
    assert contract.layout.control_branch_index == 20
    assert contract.stoken_delay == 10
    assert config.audio_encoder_config.llm_dim == contract.layout.hidden_size
    assert config.stoken_no_repeat_ngram_size == 4
    assert config.stoken_max_tokens == 1000
    assert config.audio_patch_token_id == 151_690
    assert config.audio_pad_token_id == 158_360
    assert (config.stoken_top_k, config.stoken_top_p) == (0, 1.0)


def test_rejects_non_release_branch_depth() -> None:
    config = LycheeFDConfig().to_dict()
    config["control_layer_config"] = copy.deepcopy(config["control_layer_config"])
    config["control_layer_config"]["num_hidden_layers"] = 3
    config["control_layer_config"]["layer_types"] = config["control_layer_config"]["layer_types"][:3]

    with pytest.raises(LycheeConfigError, match="28/4/4/4"):
        LycheeFDConfig.from_dict(config)


def test_rejects_audio_adaptor_dimension_mismatch() -> None:
    config = LycheeFDConfig().to_dict()
    config["audio_encoder_config"] = copy.deepcopy(config["audio_encoder_config"])
    config["audio_encoder_config"]["llm_dim"] = 1024

    with pytest.raises(LycheeConfigError, match="llm_dim"):
        LycheeFDConfig.from_dict(config)


@pytest.mark.parametrize("field", ["audio_patch_token_id", "audio_pad_token_id"])
def test_rejects_audio_tokens_outside_vocab(field: str) -> None:
    with pytest.raises(LycheeConfigError, match=field):
        LycheeFDConfig(**{field: 158_363})


def test_round_trip_keeps_checkpoint_architecture_alias() -> None:
    config = LycheeFDConfig(architectures=["StepAudio2FullDuplex"])

    restored = LycheeFDConfig.from_dict(config.to_dict())

    assert restored.architectures == ["StepAudio2FullDuplex"]
    assert restored.model_type == "step_audio_2_full_duplex"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stoken_top_k", -1),
        ("stoken_top_p", 0),
        ("stoken_top_p", 1.1),
        ("stoken_no_repeat_ngram_size", -1),
        ("stoken_max_tokens", 0),
    ],
)
def test_rejects_invalid_speech_sampling_contract(field: str, value: int | float) -> None:
    with pytest.raises(LycheeConfigError, match=field):
        LycheeFDConfig(**{field: value})
