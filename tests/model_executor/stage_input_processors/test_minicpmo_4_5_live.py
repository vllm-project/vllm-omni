# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""/v1/live/sessions Thinker -> Talker handoff: marker lookup, latent row ledger, per-unit Talker input."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import (
    MINICPMO45_LIVE_CODEC_TOKENS_PER_UNIT,
    _minicpmo45_live_marker,
    _minicpmo45_live_tts_input,
    _minicpmo45_live_unit_rows,
    llm2tts,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

LISTEN, CHUNK_EOS, CHUNK_TTS_EOS, TURN_EOS = 1, 3, 4, 5
MARKER = {"listen": LISTEN, "chunk_eos": CHUNK_EOS, "chunk_tts_eos": CHUNK_TTS_EOS, "turn_eos": TURN_EOS}
HIDDEN = 2


def _latent(rows: int) -> torch.Tensor:
    return torch.arange(rows * HIDDEN, dtype=torch.float32).reshape(rows, HIDDEN)


def _ledger(fed_ids: list[int], positions: list[int] | None = None) -> dict:
    positions = list(range(len(fed_ids))) if positions is None else positions
    return {
        "latent": _latent(len(fed_ids)),
        "latent_input_ids": torch.tensor(fed_ids).reshape(-1, 1),
        "latent_positions": torch.tensor(positions).reshape(-1, 1),
    }


def _output(sampled: list[int], mm_output: dict | None, *, completion_mm=None, hidden_states=None):
    completion = SimpleNamespace(
        cumulative_token_ids=sampled,
        token_ids=sampled[-1:],
        multimodal_output=completion_mm,
        hidden_states=hidden_states,
    )
    return SimpleNamespace(request_id="live-1", outputs=[completion], multimodal_output=mm_output)


def _live(result) -> dict:
    return result["model_intermediate_buffer"]["minicpmo_live"]


def test_marker_is_read_from_the_prompt():
    assert _minicpmo45_live_marker({"minicpmo_live": MARKER}) == MARKER


@pytest.mark.parametrize("prompt", [None, "text", {"prompt_token_ids": [1]}, {"minicpmo_live": [1, 2]}])
def test_marker_absent(prompt):
    assert _minicpmo45_live_marker(prompt) is None


def test_unit_rows_are_the_latest_contiguous_match():
    ledger = _ledger([9, 10, 11, 10, 11])
    rows = _minicpmo45_live_unit_rows(ledger, ledger["latent"], [10, 11], "r")
    assert torch.equal(rows, ledger["latent"][3:5])


def test_unit_rows_skip_position_gaps():
    ledger = _ledger([10, 11, 10, 11], positions=[0, 1, 5, 7])
    rows = _minicpmo45_live_unit_rows(ledger, ledger["latent"], [10, 11], "r")
    assert torch.equal(rows, ledger["latent"][0:2])


def test_unit_rows_without_a_ledger():
    with pytest.raises(ValueError, match="missing latent row ledger"):
        _minicpmo45_live_unit_rows({}, _latent(2), [10], "r")


def test_unit_rows_with_a_mismatched_ledger():
    ledger = _ledger([10, 11])
    with pytest.raises(ValueError, match="ledger mismatch"):
        _minicpmo45_live_unit_rows(ledger, _latent(3), [10], "r")


def test_unit_rows_without_a_match():
    ledger = _ledger([10, 11])
    with pytest.raises(ValueError, match="no latent rows"):
        _minicpmo45_live_unit_rows(ledger, ledger["latent"], [12], "r")


@pytest.mark.parametrize("sampled", [[], [10, 11], [10, LISTEN], [LISTEN]])
def test_no_talker_input_unless_a_speaking_unit_finished(sampled):
    assert _minicpmo45_live_tts_input(_output(sampled, _ledger([10, 11])), MARKER) is None


def test_first_unit_of_a_turn():
    ledger = _ledger([99, 10, 11, 12])
    result = _minicpmo45_live_tts_input(_output([10, 11, 12, CHUNK_EOS], ledger), MARKER)

    live = _live(result)
    assert live["token_ids"] == [11, 12]
    assert live["hidden_states"] == ledger["latent"][2:4].tolist()
    assert (live["turn_start"], live["turn_end"]) == (True, False)
    assert (live["codec_min"], live["codec_max"]) == (0, MINICPMO45_LIVE_CODEC_TOKENS_PER_UNIT)
    assert result["model_intermediate_buffer"]["meta"] == {"replace_streaming_prompt": True}
    assert result["prompt_token_ids"] == [0, 0, 0]


def test_unit_ending_the_turn():
    ledger = _ledger([10, 11, 12, CHUNK_EOS, 13, TURN_EOS])
    result = _minicpmo45_live_tts_input(_output([10, 11, 12, CHUNK_EOS, 13, TURN_EOS, CHUNK_TTS_EOS], ledger), MARKER)
    live = _live(result)
    assert live["token_ids"] == [TURN_EOS]
    assert (live["turn_start"], live["turn_end"], live["codec_min"]) == (False, True, 0)
    assert "meta" not in result["model_intermediate_buffer"]


def test_mid_turn_unit_needs_a_full_chunk():
    ledger = _ledger([10, 11, CHUNK_EOS, 12, 13])
    live = _live(_minicpmo45_live_tts_input(_output([10, 11, CHUNK_EOS, 12, 13, CHUNK_EOS], ledger), MARKER))
    assert live["token_ids"] == [13]
    assert (live["turn_start"], live["turn_end"]) == (False, False)
    assert live["codec_min"] == MINICPMO45_LIVE_CODEC_TOKENS_PER_UNIT


def test_listen_units_do_not_decide_the_turn_start():
    sampled = [10, TURN_EOS, CHUNK_EOS, LISTEN, LISTEN, 12, 13, CHUNK_EOS]
    ledger = _ledger([10, TURN_EOS, CHUNK_EOS, LISTEN, LISTEN, 12, 13])
    assert _live(_minicpmo45_live_tts_input(_output(sampled, ledger), MARKER))["turn_start"] is True

    sampled = [10, 11, CHUNK_EOS, LISTEN, 12, 13, CHUNK_EOS]
    ledger = _ledger([10, 11, CHUNK_EOS, LISTEN, 12, 13])
    assert _live(_minicpmo45_live_tts_input(_output(sampled, ledger), MARKER))["turn_start"] is False


def test_unit_with_only_a_stop_token_needs_no_latent():
    live = _live(_minicpmo45_live_tts_input(_output([CHUNK_EOS], None), MARKER))
    assert (live["token_ids"], live["hidden_states"], live["turn_end"]) == ([], [], False)


def test_single_spoken_token_conditions_nothing():
    result = _minicpmo45_live_tts_input(_output([10, CHUNK_EOS], _ledger([10])), MARKER)
    assert _live(result)["token_ids"] == []
    assert _live(result)["hidden_states"] == []
    assert result["prompt_token_ids"] == [0]


def test_missing_latent_is_an_error():
    with pytest.raises(ValueError, match="no thinker latent"):
        _minicpmo45_live_tts_input(_output([10, 11, CHUNK_EOS], None), MARKER)


def test_completion_multimodal_output_and_batched_latent():
    ledger = _ledger([10, 11])
    ledger["latent"] = ledger["latent"].unsqueeze(0)
    live = _live(_minicpmo45_live_tts_input(_output([10, 11, CHUNK_EOS], None, completion_mm=ledger), MARKER))
    assert live["hidden_states"] == ledger["latent"][0, 1:].tolist()


def test_hidden_states_fallback_for_the_latent():
    ledger = _ledger([10, 11])
    latent = ledger.pop("latent")
    live = _live(_minicpmo45_live_tts_input(_output([10, 11, CHUNK_EOS], ledger, hidden_states=latent), MARKER))
    assert live["hidden_states"] == latent[1:].tolist()


def test_token_ids_used_without_cumulative_ids():
    output = _output([], _ledger([10, 11]))
    output.outputs[0].cumulative_token_ids = None
    output.outputs[0].token_ids = [10, 11, CHUNK_EOS]
    assert _live(_minicpmo45_live_tts_input(output, MARKER))["token_ids"] == [11]


def test_llm2tts_routes_live_requests():
    speaking = _output([10, 11, CHUNK_EOS], _ledger([10, 11]))
    listening = _output([LISTEN], None)
    listening.request_id = "live-2"
    prompts = [{"minicpmo_live": MARKER}, {"minicpmo_live": MARKER}]

    inputs = llm2tts([speaking, listening], prompts)

    assert len(inputs) == 1
    assert _live(inputs[0])["token_ids"] == [11]
