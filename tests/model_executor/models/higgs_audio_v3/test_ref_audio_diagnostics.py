# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reference-substitution diagnostics without loading a model checkpoint."""

import pytest
import torch
from pytest_mock import MockerFixture

from vllm_omni.model_executor.models.higgs_audio_v3 import higgs_audio_v3_talker as mod

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def talker() -> mod.HiggsAudioV3TalkerForConditionalGeneration:
    # Exercise the real method and fused embedding without constructing Qwen3.
    instance = mod.HiggsAudioV3TalkerForConditionalGeneration.__new__(mod.HiggsAudioV3TalkerForConditionalGeneration)
    torch.nn.Module.__init__(instance)
    instance._last_step_query_start_loc = torch.tensor([0, 4])
    instance.multimodal_embedding = mod.HiggsFusedMultiTextEmbedding(num_codebooks=1, vocab_size=32, hidden_size=1)
    with torch.no_grad():
        instance.multimodal_embedding.weight.copy_(torch.arange(32).reshape(32, 1))
    return instance


@pytest.mark.parametrize(
    "ref_positions,mask,expected_positions,expected_codes",
    [
        ([], [True, True], 0, 2),
        ([1], [True, True], 1, 2),
        ([1, 2, 3], [True, False], 3, 1),
    ],
    ids=["empty-positions", "too-few-positions", "count-mismatch-after-masking"],
)
def test_explicit_mismatch_logs_counts_without_substitution(
    talker: mod.HiggsAudioV3TalkerForConditionalGeneration,
    mocker: MockerFixture,
    ref_positions: list[int],
    mask: list[bool],
    expected_positions: int,
    expected_codes: int,
) -> None:
    debug = mocker.spy(mod.logger, "debug")
    hidden = torch.zeros(4, 1)
    info = {
        "audio_input_ids": torch.tensor([[5], [7]]),
        "audio_input_ids_mask": torch.tensor(mask),
        "audio_placeholder_positions": torch.tensor(ref_positions, dtype=torch.long),
    }
    result = talker._apply_ref_audio_substitution(hidden, torch.ones(4, dtype=torch.long), torch.arange(4), [info])

    assert result is hidden
    assert torch.count_nonzero(hidden) == 0
    debug.assert_called_once()
    message, request_index, position_count, code_count = debug.call_args.args
    assert "reference audio" in message
    assert "placeholder" in message
    assert (request_index, position_count, code_count) == (0, expected_positions, expected_codes)


@pytest.mark.parametrize("input_ids", [[1, -100, 2, 3], [1, 2, 3, 4]], ids=["one-slot", "no-slots"])
def test_legacy_short_span_logs_counts_without_substitution(
    talker: mod.HiggsAudioV3TalkerForConditionalGeneration,
    mocker: MockerFixture,
    input_ids: list[int],
) -> None:
    debug = mocker.spy(mod.logger, "debug")
    hidden = torch.zeros(4, 1)
    result = talker._apply_ref_audio_substitution(
        hidden, torch.tensor(input_ids), torch.arange(4), [{"audio_input_ids": torch.tensor([[5], [7]])}]
    )

    assert result is hidden
    debug.assert_called_once()
    message, request_index, position_count, code_count = debug.call_args.args
    assert "legacy" in message
    assert (request_index, position_count, code_count) == (0, input_ids.count(-100), 2)


@pytest.mark.parametrize(
    "positions,expected",
    [([0, 1, 2, 3], [0.0, 5.0, 7.0, 0.0]), ([2], [7.0]), ([0, 3], [0.0, 0.0])],
    ids=["full-prefill", "chunk-intersects-reference", "chunk-outside-reference"],
)
def test_valid_explicit_positions_do_not_log_mismatch(
    talker: mod.HiggsAudioV3TalkerForConditionalGeneration,
    mocker: MockerFixture,
    positions: list[int],
    expected: list[float],
) -> None:
    debug = mocker.spy(mod.logger, "debug")
    talker._last_step_query_start_loc = torch.tensor([0, len(positions)])
    hidden = torch.zeros(len(positions), 1)
    result = talker._apply_ref_audio_substitution(
        hidden,
        torch.ones(len(positions), dtype=torch.long),
        torch.tensor(positions),
        [{"audio_input_ids": torch.tensor([[5], [7]]), "audio_placeholder_positions": torch.tensor([1, 2])}],
    )

    assert result[:, 0].tolist() == expected
    debug.assert_not_called()
    assert torch.count_nonzero(hidden) == 0


def test_matching_mask_filters_codes_and_positions_without_diagnostic(
    talker: mod.HiggsAudioV3TalkerForConditionalGeneration, mocker: MockerFixture
) -> None:
    debug = mocker.spy(mod.logger, "debug")
    result = talker._apply_ref_audio_substitution(
        torch.zeros(4, 1),
        torch.ones(4, dtype=torch.long),
        torch.arange(4),
        [
            {
                "audio_input_ids": torch.tensor([[5], [7]]),
                "audio_input_ids_mask": torch.tensor([True, False]),
                "audio_placeholder_positions": torch.tensor([1, 2]),
            }
        ],
    )
    assert result[:, 0].tolist() == [0.0, 5.0, 0.0, 0.0]
    debug.assert_not_called()


@pytest.mark.parametrize("masked", [False, True], ids=["empty-codes", "all-masked"])
def test_empty_reference_does_not_log_mismatch(
    talker: mod.HiggsAudioV3TalkerForConditionalGeneration, mocker: MockerFixture, masked: bool
) -> None:
    debug = mocker.spy(mod.logger, "debug")
    info = {
        "audio_input_ids": torch.tensor([[5], [7]]) if masked else torch.empty((0, 1), dtype=torch.long),
        "audio_input_ids_mask": torch.tensor([False, False] if masked else [], dtype=torch.bool),
        "audio_placeholder_positions": torch.tensor([1, 2] if masked else [], dtype=torch.long),
    }
    hidden = torch.zeros(4, 1)
    result = talker._apply_ref_audio_substitution(hidden, torch.ones(4, dtype=torch.long), torch.arange(4), [info])
    assert result is hidden
    debug.assert_not_called()


def test_legacy_decode_does_not_log_mismatch(
    talker: mod.HiggsAudioV3TalkerForConditionalGeneration, mocker: MockerFixture
) -> None:
    debug = mocker.spy(mod.logger, "debug")
    talker._last_step_query_start_loc = torch.tensor([0, 1])
    hidden = torch.zeros(1, 1)
    result = talker._apply_ref_audio_substitution(
        hidden, torch.tensor([-100]), torch.tensor([2]), [{"audio_input_ids": torch.tensor([[5], [7]])}]
    )
    assert result is hidden
    debug.assert_not_called()


@pytest.mark.parametrize("input_ids", [[1, -100, -100, 2], [-100, -100, -100, 2]])
def test_legacy_sufficient_slots_remain_supported(
    talker: mod.HiggsAudioV3TalkerForConditionalGeneration, mocker: MockerFixture, input_ids: list[int]
) -> None:
    debug = mocker.spy(mod.logger, "debug")
    hidden = torch.zeros(4, 1)
    result = talker._apply_ref_audio_substitution(
        hidden, torch.tensor(input_ids), torch.arange(4), [{"audio_input_ids": torch.tensor([[5], [7]])}]
    )
    targets = [i for i, token in enumerate(input_ids) if token == -100][:2]
    assert result[targets, 0].tolist() == [5.0, 7.0]
    assert torch.count_nonzero(result) == 2
    assert torch.count_nonzero(hidden) == 0
    debug.assert_not_called()
