# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Qwen3-Omni streaming thinker→talker / talker→codec helpers (PR #2581)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

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
