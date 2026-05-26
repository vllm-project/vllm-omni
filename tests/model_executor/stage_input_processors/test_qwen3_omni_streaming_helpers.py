# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Qwen3-Omni streaming thinker→talker / talker→codec helpers (PR #2581)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import vllm_omni.model_executor.stage_input_processors.qwen3_omni as q3

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _streaming_context() -> SimpleNamespace:
    return SimpleNamespace(bridge_states={})


def test_get_streaming_talker_tokens_first_segment(_streaming_context: SimpleNamespace) -> None:
    inc_p, inc_o, merged, thinker_in = q3._get_streaming_talker_tokens(
        "r1",
        [1, 2],
        [10, 11],
        streaming_context=_streaming_context,
    )
    assert inc_p == [1, 2]
    assert inc_o == [10, 11]
    assert merged == [1, 2, 10, 11]
    assert thinker_in == [1, 2]


def test_get_streaming_talker_tokens_second_segment_accumulates(_streaming_context: SimpleNamespace) -> None:
    q3._get_streaming_talker_tokens("r2", [1, 2], [10, 11], streaming_context=_streaming_context)
    inc_p, inc_o, merged, thinker_in = q3._get_streaming_talker_tokens(
        "r2",
        [1, 2, 3, 4],
        [10, 11, 12, 13],
        streaming_context=_streaming_context,
    )
    assert inc_p == [3, 4]
    assert inc_o == [12, 13]
    assert merged == [1, 2, 10, 3, 4, 12, 13]
    assert thinker_in == [1, 2, 10, 3, 4]


def test_get_streaming_talker_tokens_new_prompt_len_snapshot_truncates(
    _streaming_context: SimpleNamespace,
) -> None:
    inc_p, inc_o, merged, thinker_in = q3._get_streaming_talker_tokens(
        "r3",
        [1, 2, 3, 4, 5, 6],
        [10],
        new_prompt_len_snapshot=2,
        streaming_context=_streaming_context,
    )
    assert inc_p == [1, 2, 3, 4]
    assert inc_o == [10]
    assert merged == [1, 2, 3, 4, 10]
    assert thinker_in == [1, 2, 3, 4]


def test_get_streaming_talker_tokens_clear_state(_streaming_context: SimpleNamespace) -> None:
    q3._get_streaming_talker_tokens("r4", [1], [2], streaming_context=_streaming_context, clear_state=True)
    state = q3._get_qwen3_streaming_state("r4", _streaming_context).thinker2talker
    assert state.last_prompt_len == 0
    assert state.last_output_len == 0
    assert state.merged_sequences == []


def test_get_streaming_codec_delta_len_increments_and_finishes(_streaming_context: SimpleNamespace) -> None:
    d1 = q3._get_streaming_codec_delta_len(5, "c1", SimpleNamespace(finished=False), _streaming_context)
    assert d1 == 5
    d2 = q3._get_streaming_codec_delta_len(8, "c1", SimpleNamespace(finished=False), _streaming_context)
    assert d2 == 2
    # After d2, stored cursor is cur_seq_len + 1 == 9; next delta uses new cur_seq_len - 9.
    d3 = q3._get_streaming_codec_delta_len(10, "c1", SimpleNamespace(finished=True), _streaming_context)
    assert d3 == 1
    state = q3._get_qwen3_streaming_state("c1", _streaming_context)
    assert state.talker2code2wav_last_seq_len == 0


def test_talker2code2wav_full_payload_filters_by_output_token_ids() -> None:
    request = SimpleNamespace(
        request_id="codec",
        output_token_ids=[4197, 1, 2, 4198, -1, 2048],
    )
    rows = torch.tensor(
        [
            [100, 101, 102],
            [10, 11, 12],
            [20, 21, 22],
            [30, 31, 32],
            [40, 41, 42],
            [50, 51, 52],
        ],
        dtype=torch.long,
    )

    payload = q3.talker2code2wav_full_payload(None, {"codes.audio": rows}, request)

    assert payload is not None
    assert payload["codes"]["audio"] == [10, 20, 11, 21, 12, 22]
    assert payload["code_predictor_codes"] == payload["codes"]["audio"]


def test_talker2code2wav_full_payload_drops_count_matched_terminal_row() -> None:
    request = SimpleNamespace(
        request_id="codec_terminal_row",
        output_token_ids=[0, 4198],
    )
    rows = torch.tensor(
        [
            [10, 11, 12],
        ],
        dtype=torch.long,
    )

    payload = q3.talker2code2wav_full_payload(None, {"codes.audio": rows}, request)

    assert payload is None


def test_talker2code2wav_full_payload_drops_rows_aligned_to_non_codec_ids() -> None:
    request = SimpleNamespace(
        request_id="codec_invalid_ids",
        output_token_ids=[4197, 0, 4198, 4196, -1, 2048],
    )
    rows = torch.tensor(
        [
            [91, 92, 93],
            [0, 0, 0],
            [81, 82, 83],
            [71, 72, 73],
            [61, 62, 63],
            [51, 52, 53],
        ],
        dtype=torch.long,
    )

    payload = q3.talker2code2wav_full_payload(None, {"codes.audio": rows}, request)

    assert payload is not None
    assert payload["codes"]["audio"] == [0, 0, 0]
    assert payload["code_predictor_codes"] == payload["codes"]["audio"]


def test_talker2code2wav_full_payload_keeps_all_zero_codec_rows() -> None:
    request = SimpleNamespace(
        request_id="codec_zero",
        output_token_ids=[0, 1],
    )
    rows = torch.tensor(
        [
            [0, 0, 0],
            [7, 8, 9],
        ],
        dtype=torch.long,
    )

    payload = q3.talker2code2wav_full_payload(None, {"codes.audio": rows}, request)

    assert payload is not None
    assert payload["codes"]["audio"] == [0, 7, 0, 8, 0, 9]
    assert payload["code_predictor_codes"] == payload["codes"]["audio"]


def test_thinker2talker_full_payload_packs_complete_tensors() -> None:
    request = SimpleNamespace(
        request_id="thinker",
        prompt_token_ids=[151644, 872],
        output_token_ids=[3],
        all_token_ids=[151644, 872, 3],
    )
    pooling_output = {
        "hidden_states.layer_0": torch.ones(3, 2),
        "hidden_states.layer_24": torch.full((3, 2), 2.0),
        "embed.tts_bos": torch.zeros(1, 2),
    }

    payload = q3.thinker2talker_full_payload(None, pooling_output, request)

    assert payload is not None
    assert payload["ids"]["all"] == [151644, 872, 3]
    assert payload["embed"]["prefill"].device.type == "cpu"
    assert payload["hidden_states"]["output"].device.type == "cpu"
    assert payload["next_stage_prompt_len"] > 0
