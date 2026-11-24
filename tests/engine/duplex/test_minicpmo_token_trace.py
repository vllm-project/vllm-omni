# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MiniCPM-o 4.5 ids in the per-unit token trace (``extra_body.trace_tokens``)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.engine.duplex.test_session_runner import _fake_encode_audio
from vllm_omni.engine.duplex.contracts import DuplexFence, duplex_resource_request_id
from vllm_omni.model_executor.models.minicpmo_4_5.duplex import token_trace
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import MiniCPMO45DuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

UNIT, UNIT_END, AUDIO, LISTEN, SPEAK, CHUNK_EOS = 151683, 151684, 151698, 151705, 151706, 151718


def _plugin() -> MiniCPMO45DuplexPlugin:
    return MiniCPMO45DuplexPlugin(_fake_encode_audio)


def _thinker_output(
    prompt_len: int, generated: list[int], real_rows: list[int], request_id: str = "r"
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=[UNIT] * prompt_len,
        outputs=[SimpleNamespace(token_ids=generated, multimodal_output={"duplex_prompt_token_ids": real_rows})],
    )


def test_the_thinker_trace_reads_the_worker_ids_and_splits_off_the_session_context() -> None:
    plugin = _plugin()
    cursor: dict[str, object] = {}
    runtime_config = {"duplex_first_append_context_tokens": 3}
    first_rows = [1, 2, 3, UNIT, AUDIO, AUDIO]
    first = plugin.unit_token_trace(
        stage_id=0,
        output=_thinker_output(6, [LISTEN], first_rows),
        context=None,
        cursor=cursor,
        runtime_config=runtime_config,
    )
    assert first == {"stage": "thinker", "input": [UNIT, AUDIO, AUDIO], "output": [LISTEN], "prompt_input": [1, 2, 3]}

    # The second segment's row is prompt-length (pads in front of the new ids), accumulated after the first.
    second_rows = first_rows + [0] * 6 + [LISTEN, UNIT_END, UNIT, AUDIO]
    second = plugin.unit_token_trace(
        stage_id=0,
        output=_thinker_output(10, [LISTEN, SPEAK, 9, CHUNK_EOS], second_rows),
        context=None,
        cursor=cursor,
        runtime_config=runtime_config,
    )
    assert second == {"stage": "thinker", "input": [LISTEN, UNIT_END, UNIT, AUDIO], "output": [SPEAK, 9, CHUNK_EOS]}


@pytest.mark.parametrize("real_rows", [[1], []], ids=["one id for a two-token delta", "no worker ids"])
def test_thinker_placeholders_are_labelled_when_the_workers_ids_are_missing_or_short(real_rows: list[int]) -> None:
    """A prompt delta the worker did not report must not be rendered as tokens the model consumed."""
    plugin = _plugin()
    cursor: dict[str, object] = {}
    trace = plugin.unit_token_trace(
        stage_id=0,
        output=_thinker_output(2, [LISTEN], real_rows),
        context=None,
        cursor=cursor,
        runtime_config={},
    )
    assert trace == {
        "stage": "thinker",
        "input": [[UNIT, f"placeholder:{UNIT}"], [UNIT, f"placeholder:{UNIT}"]],
        "output": [LISTEN],
    }
    # The next segment reports its own ids again.
    following = plugin.unit_token_trace(
        stage_id=0,
        output=_thinker_output(4, [LISTEN, SPEAK, 9, CHUNK_EOS], real_rows + [0] * 2 + [UNIT, AUDIO]),
        context=None,
        cursor=cursor,
        runtime_config={},
    )
    assert following == {"stage": "thinker", "input": [UNIT, AUDIO], "output": [SPEAK, 9, CHUNK_EOS]}


def test_a_window_rebuild_and_a_new_epoch_are_reported_whole_without_a_second_prompt() -> None:
    plugin = _plugin()
    cursor: dict[str, object] = {}
    runtime_config = {"duplex_first_append_context_tokens": 3}
    rows = [1, 2, 3, UNIT, AUDIO, AUDIO]
    plugin.unit_token_trace(
        stage_id=0,
        output=_thinker_output(6, [LISTEN], rows),
        context=None,
        cursor=cursor,
        runtime_config=runtime_config,
    )
    # The sliding window rebuilt a shorter prompt within the same request.
    rebuilt = plugin.unit_token_trace(
        stage_id=0,
        output=_thinker_output(4, [LISTEN, LISTEN], rows + [9, 9, UNIT, AUDIO]),
        context=None,
        cursor=cursor,
        runtime_config=runtime_config,
    )
    assert rebuilt == {"stage": "thinker", "input": [9, 9, UNIT, AUDIO], "output": [LISTEN]}
    # A new epoch opens a new request whose prompt is longer than the old cursor.
    epoch = plugin.unit_token_trace(
        stage_id=0,
        output=_thinker_output(8, [SPEAK], [7] * 8, request_id="r-epoch-1"),
        context=None,
        cursor=cursor,
        runtime_config=runtime_config,
    )
    assert epoch == {"stage": "thinker", "input": [7] * 8, "output": [SPEAK]}


def _request_id(session_id: str, epoch: int = 0) -> str:
    return duplex_resource_request_id(DuplexFence(session_id, epoch=epoch), "stage0")


def test_the_talker_trace_uses_the_recorded_handoff_only_for_traced_sessions() -> None:
    plugin = _plugin()
    traced, other = _request_id("s-trace"), _request_id("s-other")
    talker = SimpleNamespace(
        request_id=traced, prompt_token_ids=[0, 0, 0], outputs=[SimpleNamespace(token_ids=[41, 42, 6561])]
    )

    token_trace.record_handoff(traced, [9625, 374])  # not traced yet: dropped
    plugin.token_trace_started(session_id="s-trace")
    try:
        token_trace.record_handoff(other, [1])  # another, untraced session: never recorded
        token_trace.record_handoff(traced, [9625, 374])
        trace = plugin.unit_token_trace(stage_id=1, output=talker, context=None, cursor={}, runtime_config={})
        assert trace == {
            "stage": "talker",
            "input": [9625, 374, [151687, "<audio_bos>"]],
            "output": [[41, "s3:41"], [42, "s3:42"], [6561, "s3:6561"]],
        }
        fallback = plugin.unit_token_trace(stage_id=1, output=talker, context=None, cursor={}, runtime_config={})
        assert fallback is not None and fallback["input"] == [[0, "placeholder:0"]] * 3
        assert token_trace.pop_handoff(other) is None
    finally:
        plugin.token_trace_stopped(session_id="s-trace")
    assert token_trace.pop_handoff(traced) is None


def test_a_new_epoch_drops_the_cancelled_epochs_handoffs_and_stop_drops_everything() -> None:
    old, new = _request_id("s-epoch", 0), _request_id("s-epoch", 1)
    token_trace.start("s-epoch")
    try:
        token_trace.record_handoff(old, [1, 2])  # a barge-in aborts the Talker before it takes this
        token_trace.record_handoff(new, [3])
        assert token_trace.pop_handoff(old) is None
        assert token_trace.pop_handoff(new) == [3]
        token_trace.record_handoff(new, [4])
    finally:
        token_trace.stop("s-epoch")
    assert token_trace.pop_handoff(new) is None
