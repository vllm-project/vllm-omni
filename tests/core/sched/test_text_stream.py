# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the scheduler-side text streams (``append_text``)."""

import pytest

from vllm_omni.core.sched.output import TextStreamDelivery
from vllm_omni.core.sched.text_stream import (
    TextStreams,
    TextStreamSpec,
    declares_text_stream,
    end_read,
    text_ready,
    text_row,
    text_stream_timeout_s,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _streams(*, prompt_ids: int = 1, min_ids: int = 0, timeout_s: float = 5.0, clock: _Clock | None = None):
    return TextStreams(
        lambda _info, _model_config: TextStreamSpec(prompt_ids, min_ids, id_limit=100),
        timeout_s,
        clock=clock or _Clock(),
    )


def test_a_request_declares_its_stream_with_the_list_wrapped_flag():
    assert declares_text_stream({"text_stream": [True]})
    assert declares_text_stream({"text_stream": True})
    assert not declares_text_stream({"text_stream": [False]})
    assert not declares_text_stream({"text_stream": []})
    assert not declares_text_stream({})
    assert not declares_text_stream(None)


def test_a_step_runs_once_every_text_row_it_reads_has_arrived():
    # The prompt reads ids 0..2 (a lead of 2); frame f reads id 2 + f.
    assert not text_ready(3, 0, 2, ended=False)
    assert text_ready(3, 0, 3, ended=False)
    assert not text_ready(3, 1, 3, ended=False)
    assert text_ready(3, 1, 4, ended=False)
    # After the END only END and PAD rows remain to read.
    assert text_ready(3, 7, 0, ended=True)


@pytest.mark.parametrize("prompt_ids", [1, 3])
@pytest.mark.parametrize("num_ids", [0, 1, 4, 7])
def test_the_hold_rule_and_the_eos_gate_read_one_row_index(prompt_ids, num_ids):
    """The scheduler's hold and the Talker's EOS gate are one contract (one index, text_row)."""
    for frame in range(16):
        row = text_row(prompt_ids, frame)
        assert not end_read(prompt_ids, frame, num_ids, ended=False)
        assert text_ready(prompt_ids, frame, num_ids, ended=False) == (row < num_ids)
        if end_read(prompt_ids, frame, num_ids, ended=True):
            # EOS is only sampled on a step the scheduler lets run, after the END row is read.
            assert row >= num_ids and text_ready(prompt_ids, frame, num_ids, ended=True)
    first_end = next(frame for frame in range(64) if end_read(prompt_ids, frame, num_ids, ended=True))
    assert first_end == max(0, num_ids - prompt_ids + 1)


def test_register_refuses_without_a_policy_and_on_a_policy_error():
    assert "text_stream_policy" in TextStreams(None, 5.0).register("r", {})

    def refuse(_info, _model_config):
        raise ValueError("only CustomVoice")

    streams = TextStreams(refuse, 5.0)
    assert streams.register("r", {}) == "only CustomVoice"
    assert "r" not in streams


def test_appends_add_ids_until_the_final_one():
    streams = _streams()
    assert streams.register("r", {}) is None
    streams.append("r", [5, 6], final=False)
    streams.append("r", [7], final=True)
    with pytest.raises(ValueError, match="already ended"):
        streams.append("r", [8], final=False)
    with pytest.raises(ValueError, match=">= 0"):
        _register_and_append(streams, "bad", [-1])
    with pytest.raises(ValueError, match="below 100"):
        _register_and_append(streams, "too-big", [100])
    with pytest.raises(ValueError, match="no open text stream"):
        streams.append("unknown", [1], final=False)


def _register_and_append(streams, request_id, ids):
    streams.register(request_id, {})
    streams.append(request_id, ids, final=False)


def test_a_delivery_hands_each_id_and_the_end_over_once():
    streams = _streams(prompt_ids=2)
    streams.register("r", {})
    streams.append("r", [5, 6], final=False)
    assert streams.take_delivery("r") == TextStreamDelivery(start=0, token_ids=[5, 6], ended=False, prompt_ids=2)
    assert streams.take_delivery("r") is None
    streams.append("r", [7], final=False)
    assert streams.take_delivery("r") == TextStreamDelivery(start=2, token_ids=[7], ended=False, prompt_ids=2)
    streams.append("r", [], final=True)
    assert streams.take_delivery("r") == TextStreamDelivery(start=3, token_ids=[], ended=True, prompt_ids=2)
    assert streams.take_delivery("r") is None
    assert streams.take_delivery("unknown") is None


def test_the_first_delivery_announces_an_empty_stream():
    streams = _streams(prompt_ids=0)
    streams.register("r", {})
    assert streams.take_delivery("r") == TextStreamDelivery(start=0, token_ids=[], ended=False, prompt_ids=0)
    assert streams.take_delivery("r") is None


def test_the_deadline_runs_only_while_the_request_waits_for_text():
    clock = _Clock()
    streams = _streams(prompt_ids=2, timeout_s=5.0, clock=clock)
    streams.register("r", {})
    assert not streams.ready("r", 0)
    clock.now += 4.0
    assert not streams.ready("r", 0)
    assert streams.expired() == []
    # An append restarts the clock even when the request still cannot run.
    streams.append("r", [1], final=False)
    clock.now += 4.0
    assert not streams.ready("r", 0)
    clock.now += 4.0
    assert streams.expired() == []
    clock.now += 1.5
    assert streams.expired() == ["r"]
    # A request that can run is not waiting.
    streams.append("r", [2], final=False)
    assert streams.ready("r", 0)
    clock.now += 60.0
    assert streams.expired() == []


def test_a_zero_timeout_disables_the_deadline():
    clock = _Clock()
    streams = _streams(prompt_ids=1, timeout_s=0.0, clock=clock)
    streams.register("r", {})
    assert not streams.ready("r", 0)
    clock.now += 1e6
    assert streams.expired() == []


def test_a_stream_that_ends_short_of_the_prompt_minimum_is_reported():
    streams = _streams(prompt_ids=5, min_ids=3)
    streams.register("r", {})
    streams.append("r", [1, 2], final=False)
    assert streams.ended_short("r") is None
    streams.append("r", [], final=True)
    assert "needs at least 3" in streams.ended_short("r")


def test_release_drops_finished_streams():
    streams = _streams()
    streams.register("r", {})
    streams.release(["r", "never-registered"])
    assert "r" not in streams and not streams


@pytest.mark.parametrize(("raw", "expected"), [(None, 5.0), ("0", 0.0), ("2.5", 2.5)])
def test_the_timeout_comes_from_the_environment(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("VLLM_OMNI_TEXT_STREAM_TIMEOUT_S", raising=False)
    else:
        monkeypatch.setenv("VLLM_OMNI_TEXT_STREAM_TIMEOUT_S", raw)
    assert text_stream_timeout_s() == expected


@pytest.mark.parametrize("raw", ["-1", "nan", "inf", "soon"])
def test_a_bad_timeout_is_refused(monkeypatch, raw):
    monkeypatch.setenv("VLLM_OMNI_TEXT_STREAM_TIMEOUT_S", raw)
    with pytest.raises(ValueError, match="VLLM_OMNI_TEXT_STREAM_TIMEOUT_S"):
        text_stream_timeout_s()
