# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest

from vllm_omni.model_executor.models.lychee_fd.contract import (
    LycheeConfigError,
    LycheeDialogueState,
    LycheeFDContract,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _reference_config() -> dict[str, object]:
    branch = {"model_type": "qwen2", "hidden_size": 3584, "num_hidden_layers": 4, "vocab_size": 158363}
    return {
        "model_type": "step_audio_2_full_duplex",
        "text_config": {**branch, "num_hidden_layers": 28},
        "stoken_layer_config": dict(branch),
        "control_layer_config": dict(branch),
        "merge_layer_config": dict(branch),
        "control_branch_layer": 8,
        "start_speaking_token_id": 158352,
        "start_listening_token_id": 158353,
        "keep_listening_token_id": 158354,
        "keep_speaking_token_id": 158355,
        "start_bc_token_id": 158362,
        "keep_bc_token_id": 158362,
        "end_bc_token_id": 158353,
        "stoken_token_ids_min": 151694,
        "stoken_token_ids_max": 158352,
        "control_token_ids_min": 158352,
        "control_token_ids_max": 158356,
        "stoken_delay_num": 10,
    }


def test_reference_checkpoint_contract() -> None:
    contract = LycheeFDContract.from_config(_reference_config())

    assert contract.layout.main_layers == 28
    assert contract.layout.control_branch_index == 20
    assert contract.layout.total_layers == 40
    assert contract.layout.hidden_size == 3584
    assert contract.layout.vocab_size == 158363
    assert contract.input_sample_rate_hz == 16000
    assert contract.output_sample_rate_hz == 24000
    assert contract.inference_window_ms == 400
    assert contract.stoken_delay == 10


@pytest.mark.parametrize(
    ("initial", "token_key", "expected"),
    [
        (LycheeDialogueState.LISTENING, "keep_listening_token_id", LycheeDialogueState.LISTENING),
        (LycheeDialogueState.LISTENING, "start_speaking_token_id", LycheeDialogueState.SPEAKING),
        (LycheeDialogueState.SPEAKING, "keep_speaking_token_id", LycheeDialogueState.SPEAKING),
        (LycheeDialogueState.SPEAKING, "start_listening_token_id", LycheeDialogueState.LISTENING),
        (LycheeDialogueState.LISTENING, "start_bc_token_id", LycheeDialogueState.BACKCHANNEL),
        (LycheeDialogueState.BACKCHANNEL, "end_bc_token_id", LycheeDialogueState.LISTENING),
    ],
)
def test_control_token_transition(
    initial: LycheeDialogueState,
    token_key: str,
    expected: LycheeDialogueState,
) -> None:
    config = _reference_config()
    contract = LycheeFDContract.from_config(config)

    token_id = config[token_key]
    assert isinstance(token_id, int)
    assert contract.tokens.next_state(initial, token_id) is expected


def test_unknown_control_token_keeps_state() -> None:
    contract = LycheeFDContract.from_config(_reference_config())

    assert contract.tokens.next_state(LycheeDialogueState.BACKCHANNEL, 42) is LycheeDialogueState.BACKCHANNEL


def test_rejects_mismatched_branch_hidden_size() -> None:
    config = _reference_config()
    config["control_layer_config"] = {
        "model_type": "qwen2",
        "hidden_size": 1024,
        "num_hidden_layers": 4,
        "vocab_size": 158363,
    }

    with pytest.raises(LycheeConfigError, match="share hidden_size"):
        LycheeFDContract.from_config(config)


def test_rejects_invalid_control_branch_boundary() -> None:
    config = _reference_config()
    config["control_branch_layer"] = 29

    with pytest.raises(LycheeConfigError, match="control_branch_layer"):
        LycheeFDContract.from_config(config)


def test_rejects_out_of_vocabulary_special_token() -> None:
    config = _reference_config()
    config["start_speaking_token_id"] = 999999

    with pytest.raises(LycheeConfigError, match="outside vocab_size"):
        LycheeFDContract.from_config(config)
