# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Session options that control how the runner advances model time between client inputs."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.engine.duplex.test_session_runner import (
    LISTEN_TOKEN_ID,
    SESSION_ID,
    append_audio,
    close_harness,
    commands,
    listen_output,
    open_harness,
    tts_output,
    types,
)
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex.config import INPUT_CLOCK_IDLE_TIMEOUT_S, DuplexSessionConfig
from vllm_omni.engine.duplex.messages import CloseDuplexSessionMessage, ResumeDuplexSessionMessage
from vllm_omni.engine.duplex.plugin import DuplexUnitDecision
from vllm_omni.engine.duplex.session.input_clock import DEFAULT_UNIT_MAX_AGE_S, DEFAULT_UNIT_TIMEOUT_S, StageProgress
from vllm_omni.engine.duplex.session.manager import DuplexSessionManager
from vllm_omni.engine.duplex.session.runner import DuplexSessionRunner
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.input import MiniCPMO45PcmAppendBuffer
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import MiniCPMO45DuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


async def test_silence_continuation_can_be_turned_off_per_session() -> None:
    """``extra_body.silence_continuation: false``: the model only hears audio the client sent."""
    h = await open_harness(extra_body={"silence_continuation": False})
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        await h.deliver_and_settle(tts_output(request_id, samples=24000, text="hello"))
        await h.deliver_and_settle(
            tts_output(request_id, samples=48000, text="hello", tts_is_last_chunk=True, finished=True)
        )

        assert len(h.port.submissions) == 1, "no silence unit may be invented"
        assert h.runner.model_state.continuation_units == 0
        assert h.session.active_response_id is not None
    finally:
        await close_harness(h)


# --------------------------------------------------------------------------- #
# Input-clocked sessions (extra_body.clock == "input")                       #
# --------------------------------------------------------------------------- #

INPUT_CLOCK = {"clock": "input"}


@pytest.fixture
def input_clock_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """No model opts into the input clock in this change: let the harness model stand in for one that has."""
    monkeypatch.setattr(MiniCPMO45DuplexPlugin, "supports_input_clock", True)


def _acks(events: list) -> list:
    return [event for event in events if event.type == "input_audio_buffer.processed"]


def _units(ack) -> list[tuple[str, object]]:
    return [(unit["decision"], unit.get("reason")) for unit in ack.units]


def _speak_segment_end(request_id: str) -> SimpleNamespace:
    """A finished Stage-0 segment with text and no listen token: forwarded to the TTS stage."""
    return SimpleNamespace(
        request_id=request_id,
        finished=False,
        outputs=[SimpleNamespace(text="hi", token_ids=[11, 12], multimodal_output={})],
        multimodal_output={"meta.listen_token_id": LISTEN_TOKEN_ID},
    )


def _deliver_listen(h, request_id: str) -> Awaitable[list]:
    return h.deliver_and_settle(
        listen_output(request_id),
        stage_id=0,
        segment_finished=True,
        segment_token_ids=[11, 12, LISTEN_TOKEN_ID],
        segment_output_metadata={"meta.listen_token_id": LISTEN_TOKEN_ID},
    )


def _deliver_listen_at(h, request_id: str, epoch: int) -> Awaitable[list]:
    return h.deliver_and_settle(
        listen_output(request_id),
        stage_id=0,
        segment_finished=True,
        segment_token_ids=[11, 12, LISTEN_TOKEN_ID],
        segment_output_metadata={"meta.listen_token_id": LISTEN_TOKEN_ID},
        epoch=epoch,
    )


def _deliver_audio_end(h, request_id: str) -> Awaitable[list]:
    return h.deliver_and_settle(
        tts_output(request_id, samples=24000, text="hello", tts_is_last_chunk=True), segment_finished=True
    )


async def _speaking_unit(h) -> str:
    """Append one unit and let the model start speaking it (Stage 0 forwarded, first TTS chunk out)."""
    await h.run(append_audio())
    request_id = h.stage0_request_id()
    assert h.deliver(_speak_segment_end(request_id), stage_id=0, segment_finished=True) is False
    events = await h.deliver_and_settle(tts_output(request_id, samples=24000, text="he"))
    assert "response.output_audio.delta" in types(events) and _acks(events) == []
    return request_id


def _fake_time(h) -> list[float]:
    """Drive the session's input clock from a fake monotonic clock (``now[0]``)."""
    now = [time.monotonic()]
    h.runner._input_clock._clock = lambda: now[0]
    return now


async def _check_timeouts(h) -> list:
    """What the once-a-second timer does, without waiting for it."""
    h.runner._on_input_clock_timer()
    return await h.settle()


async def _wait_until(predicate, *, timeout_s: float = 2.0) -> None:
    """Poll the loop until ``predicate()`` holds (instead of sleeping a fixed time)."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while not predicate():
        assert loop.time() < deadline, "condition not reached"
        await asyncio.sleep(0.002)


# ---- acknowledgement order ---------------------------------------------------


@pytest.mark.usefixtures("input_clock_model")
async def test_an_append_that_closes_no_unit_is_acknowledged_at_once() -> None:
    h = await open_harness(extra_body=INPUT_CLOCK)
    try:
        events = await h.run(append_audio(samples=3200))
        (ack,) = _acks(events)
        assert (ack.audio_end_ms, ack.unit_end_ms, ack.units) == (200, 0, ())
        assert (ack.trigger, ack.input_index) == ("input_audio_buffer.append", 1)
        assert h.port.submissions == []
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_an_input_rejected_with_an_error_is_still_acknowledged() -> None:
    h = await open_harness(extra_body=INPUT_CLOCK)
    try:
        events = await h.run(commands.Commit())
        assert types(events) == ["error", "input_audio_buffer.processed"]
        assert _acks(events)[0].units == ()
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_an_append_refused_at_admission_is_acknowledged_in_input_order() -> None:
    """Input backpressure refuses the second append before the runner sees it: it still gets its turn."""
    h = await open_harness(
        extra_body=INPUT_CLOCK,
        runtime_config=DuplexSessionRuntimeConfig(max_pending_input_bytes_per_session=16000 * 4),
    )
    try:
        h.submit(append_audio())  # holds the whole input budget until the runner dequeues it
        h.submit(append_audio(event_id="evt-refused"))
        events = await h.settle()
        errors = [event for event in events if event.type == "error"]
        assert [(e.code, e.related_event_id) for e in errors] == [("input_backpressure", "evt-refused")]
        assert _acks(events) == [], "the refused input waits behind the first one's unit"

        events = await _deliver_listen(h, h.stage0_request_id())

        acks = _acks(events)
        assert [a.input_index for a in acks] == [1, 2]
        assert [_units(a) for a in acks] == [[("listen", None)], []]
        assert (acks[1].trigger, acks[1].audio_end_ms) == ("input_audio_buffer.append", 1000)
        assert [(a.decision, a.reason) for a in acks] == [(None, None), ("rejected", "input_backpressure")]
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_an_append_with_a_modality_the_model_refuses_is_acknowledged() -> None:
    h = await open_harness(extra_body=INPUT_CLOCK)
    try:
        events = await h.run(replace(append_audio(event_id="evt-empty"), audio=b""))

        assert types(events) == ["error", "input_audio_buffer.processed"]
        assert (events[0].code, events[0].related_event_id) == ("invalid_input_modality", "evt-empty")
        assert (events[1].input_index, events[1].units) == (1, ())
        assert (events[1].decision, events[1].reason) == ("rejected", "invalid_input_modality")
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_an_append_the_runner_refuses_before_it_reaches_the_model_is_acknowledged_as_rejected() -> None:
    """Refused while its audio is prepared (here: undecodable pcm16): same acknowledgement as at admission."""
    h = await open_harness(extra_body=INPUT_CLOCK)
    try:
        odd = commands.AppendAudio(audio=b"\x00" * 3, format="pcm16", sample_rate_hz=16000, event_id="evt-odd")
        events = await h.run(odd)

        assert types(events) == ["error", "input_audio_buffer.processed"]
        assert (events[0].code, events[0].related_event_id) == ("bad_audio", "evt-odd")
        assert (events[1].input_index, events[1].decision, events[1].reason) == (1, "rejected", "bad_audio")
        assert not h.port.submissions

        events = await h.run(append_audio(samples=3200))  # the next input is acknowledged normally
        assert [(a.input_index, a.decision) for a in _acks(events)] == [(2, None)]
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_an_append_the_runner_refuses_after_decoding_does_not_count_its_audio() -> None:
    """pcm16 passes admission at 2 B/sample but the runner reserves the decoded 4 B/sample: refused, not heard."""
    h = await open_harness(
        extra_body=INPUT_CLOCK, runtime_config=DuplexSessionRuntimeConfig(max_pending_input_bytes_per_session=40000)
    )
    try:
        big = commands.AppendAudio(audio=b"\x10\x00" * 16000, format="pcm16", sample_rate_hz=16000, event_id="evt-big")
        events = await h.run(big)

        assert [(e.code, e.related_event_id) for e in events if e.type == "error"] == [
            ("input_backpressure", "evt-big")
        ]
        (ack,) = _acks(events)
        assert (ack.decision, ack.reason, ack.audio_end_ms) == ("rejected", "input_backpressure", 0)
        assert h.port.submissions == []

        (ack,) = _acks(await h.run(append_audio(samples=3200)))  # a resend of what fits counts once
        assert (ack.input_index, ack.audio_end_ms, ack.decision) == (2, 200, None)
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_a_handler_failure_is_reported_before_the_acknowledgement(monkeypatch: pytest.MonkeyPatch) -> None:
    h = await open_harness(extra_body=INPUT_CLOCK)
    try:

        async def _boom(self, command):
            raise RuntimeError("boom")

        monkeypatch.setattr(DuplexSessionRunner, "_on_command", _boom)
        events = await h.run(commands.CreateResponse())

        assert types(events) == ["error", "input_audio_buffer.processed"]
        assert events[0].code == "internal_error"
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_a_refused_response_create_error_names_its_event() -> None:
    h = await open_harness(auto_response=False, extra_body=INPUT_CLOCK)
    try:
        events = await h.run(commands.CreateResponse(event_id="evt-create"))

        errors = [e for e in events if e.type == "error"]
        assert [(e.code, e.related_event_id) for e in errors] == [("response_create_without_input", "evt-create")]
        assert _acks(events)[0].input_index == 1
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_an_input_refused_because_the_session_is_closing_is_not_acknowledged() -> None:
    h = await open_harness(extra_body=INPUT_CLOCK)
    try:
        h.runner._begin_close("client_close")

        events = await h.run(append_audio())

        assert [e.code for e in events if e.type == "error"] == ["session_closed"]
        assert _acks(events) == []
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_a_commit_refused_at_admission_is_acknowledged() -> None:
    h = await open_harness(
        extra_body=INPUT_CLOCK, runtime_config=DuplexSessionRuntimeConfig(max_pending_turns_per_session=1)
    )
    try:
        h.submit(commands.Commit())
        h.submit(commands.Commit(event_id="evt-refused"))
        events = await h.settle()

        assert [e.code for e in events if e.type == "error"] == ["input_backpressure", "input_audio_buffer_empty"]
        # The first commit is rejected while handled (nothing to commit), the
        # second refused at admission (its slot is still held by the first).
        assert [(a.input_index, a.trigger, a.decision, a.reason) for a in _acks(events)] == [
            (1, "input_audio_buffer.commit", None, None),
            (2, "input_audio_buffer.commit", "rejected", "input_backpressure"),
        ]
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_inputs_still_queued_at_teardown_are_covered_by_its_acknowledgement() -> None:
    h = await open_harness(extra_body=INPUT_CLOCK)
    try:
        h.port.submit_gate = asyncio.Event()
        h.submit(append_audio())
        await asyncio.wait_for(h.port.submit_started.wait(), timeout=2.0)
        h.submit(commands.Commit())  # waits for the append in flight
        h.submit(append_audio())  # queued behind the commit
        await _wait_until(lambda: h.runner._mailbox.qsize() == 1)  # the worker is inside the commit

        await h.manager.handle(
            CloseDuplexSessionMessage(control_id="c-close", session_id=SESSION_ID, reason="client_close")
        )
        events = await h.settle(timeout_s=1.0)

        (ack,) = _acks(events)
        assert (ack.first_input_index, ack.input_index) == (1, 3)
        assert _units(ack) == [("aborted", "client_close")]
        assert types(events).index("input_audio_buffer.processed") < types(events).index("session.closed")
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_a_listen_unit_is_acknowledged_after_its_decision_events() -> None:
    h = await open_harness(extra_body=INPUT_CLOCK)
    try:
        events = await h.run(append_audio())
        assert _acks(events) == [], "the unit is in flight"

        events = await _deliver_listen(h, h.stage0_request_id())

        assert types(events) == ["response.listen", "input_audio_buffer.processed"]
        assert (events[-1].audio_end_ms, events[-1].unit_end_ms) == (1000, 1000)
        assert events[-1].units == ({"end_ms": 1000, "decision": "listen"},)
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_a_speaking_unit_is_acknowledged_after_its_final_stage_output() -> None:
    h = await open_harness(extra_body=INPUT_CLOCK)
    try:
        request_id = await _speaking_unit(h)

        events = await h.deliver_and_settle(
            tts_output(request_id, samples=48000, text="hello", tts_is_last_chunk=True),
            segment_finished=True,
        )

        assert types(events)[-1] == "input_audio_buffer.processed"
        assert "response.output_audio.delta" in types(events)
        assert events[-1].units == ({"end_ms": 1000, "decision": "speak"},)
        # No silence unit: the model only hears what the client sent.
        assert len(h.port.submissions) == 1
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_a_commit_is_acknowledged_once_the_turn_it_submitted_is_decided() -> None:
    h = await open_harness(extra_body=INPUT_CLOCK)
    try:
        await h.run(append_audio(samples=3200))
        events = await h.run(commands.Commit())
        assert _acks(events) == [], "the committed residual is a unit in flight"
        assert len(h.port.submissions) == 1

        events = await _deliver_listen(h, h.port.submissions[-1].context.request_id)

        (ack,) = _acks(events)
        assert (ack.trigger, ack.input_index) == ("input_audio_buffer.commit", 2)
        assert ack.units == ({"end_ms": 200, "decision": "listen"},)
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_a_segment_end_the_plugin_says_hands_nothing_on_ends_its_unit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Stage-0 segment end with no decision is classified too: a speak with nothing for the next stage ends."""

    def _classify(self, *, stage_id, decision, output=None, context=None, runtime_config):
        assert "instructions" in runtime_config, "the session's runtime config"
        if decision is None and stage_id == 0:
            return DuplexUnitDecision(label="speak_empty")
        return None

    monkeypatch.setattr(MiniCPMO45DuplexPlugin, "unit_decision", _classify)
    h = await open_harness(extra_body=INPUT_CLOCK)
    try:
        await h.run(append_audio())
        events = await h.deliver_and_settle(
            _speak_segment_end(h.stage0_request_id()), stage_id=0, segment_finished=True
        )

        (ack,) = _acks(events)
        assert _units(ack) == [("speak_empty", None)], "not left waiting for a TTS segment that never comes"
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_an_append_longer_than_a_unit_submits_every_whole_unit_it_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _prepare_backlog(self, *, operation_id, chunk_period_ms):
        if self._sample_rate_hz is None or len(self._buffer) // 4 < self._sample_rate_hz * chunk_period_ms // 1000:
            return None
        payload = {"type": "audio", "audio": "", "format": "pcm_f32le", "sample_rate_hz": self._sample_rate_hz}
        return self.prepare_append(payload, operation_id=operation_id, chunk_period_ms=chunk_period_ms)

    monkeypatch.setattr(MiniCPMO45PcmAppendBuffer, "prepare_backlog", _prepare_backlog, raising=False)
    h = await open_harness(extra_body=INPUT_CLOCK)
    try:
        events = await h.run(append_audio(samples=16000 * 3 + 3200))  # 3.2 s: three whole 1 s units

        assert len(h.port.submissions) == 3 and _acks(events) == []
        for _ in range(2):
            assert _acks(await _deliver_listen(h, h.stage0_request_id())) == []
        (ack,) = _acks(await _deliver_listen(h, h.stage0_request_id()))
        assert (ack.input_index, ack.audio_end_ms, ack.unit_end_ms) == (1, 3200, 3000)
        assert _units(ack) == [("listen", None)] * 3
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_a_resumed_session_reports_how_many_inputs_reached_it() -> None:
    h = await open_harness(extra_body=INPUT_CLOCK)
    try:
        for _ in range(3):
            await h.run(append_audio(samples=3200))
        await h.manager.handle(
            ResumeDuplexSessionMessage(
                control_id="c-resume", session_id=SESSION_ID, expected_lease_generation=h.session.lease_generation
            )
        )
        result = await asyncio.wait_for(h.results.get(), timeout=2.0)

        assert result.ok and result.public_session["input_index"] == 3
    finally:
        await close_harness(h)


# ---- deferred committed turns ------------------------------------------------


async def _defer_a_commit_behind_the_active_response(h) -> str:
    """Turn mode: a committed turn speaks; a second commit arrives while it is still speaking."""
    acks = _acks(await h.run(append_audio()))
    assert [a.input_index for a in acks] == [1], "a buffered append is acknowledged at once"
    await h.run(commands.Commit(create_response=True))
    request_id = h.port.submissions[-1].context.request_id
    assert h.deliver(_speak_segment_end(request_id), stage_id=0, segment_finished=True) is False
    await h.deliver_and_settle(tts_output(request_id, samples=24000, text="he"))
    assert _acks(await h.run(append_audio())) == [], "behind the speaking unit"
    events = await h.run(commands.Commit(create_response=True))
    assert events[-1].type == "conversation.item.done" and _acks(events) == []
    assert len(h.port.submissions) == 1, "the second turn waits for the active response"
    return request_id


def _finish_turn(h, request_id: str) -> Awaitable[list]:
    return h.deliver_and_settle(
        tts_output(request_id, samples=24000, text="hello", tts_is_last_chunk=True, finished=True),
        segment_finished=True,
    )


@pytest.mark.usefixtures("input_clock_model")
async def test_a_deferred_commit_is_acknowledged_after_the_turn_it_was_deferred_for() -> None:
    h = await open_harness(auto_response=False, extra_body=INPUT_CLOCK)
    try:
        request_id = await _defer_a_commit_behind_the_active_response(h)

        events = await _finish_turn(h, request_id)

        acks = _acks(events)
        assert [(a.input_index, a.trigger) for a in acks] == [
            (2, "input_audio_buffer.commit"),
            (3, "input_audio_buffer.append"),
        ]
        assert types(events).index("response.done") < types(events).index("input_audio_buffer.processed")
        assert len(h.port.submissions) == 2, "the deferred turn was submitted once the response ended"

        events = await _deliver_listen(h, h.port.submissions[-1].context.request_id)

        (ack,) = _acks(events)
        assert (ack.input_index, ack.trigger) == (4, "input_audio_buffer.commit")
        assert _units(ack) == [("listen", None)]
    finally:
        await close_harness(h)


@pytest.mark.parametrize("clocked", [True, False], ids=["input-clocked", "wall-clock"])
@pytest.mark.usefixtures("input_clock_model")
async def test_an_input_clocked_session_never_overlaps_turns(monkeypatch: pytest.MonkeyPatch, clocked: bool) -> None:
    """A model that opens the next turn while the previous one still speaks: not in an input-clocked session.

    The clock credits outputs to the speaking units in order, so overlapping turns would credit one turn's
    output (or its silent decision) to the other. A commit during the response takes the deferred-turn path.
    """
    monkeypatch.setattr(MiniCPMO45DuplexPlugin, "release_concurrent_turn_requests", lambda self, *args, **kw: True)
    h = await open_harness(auto_response=False, extra_body=INPUT_CLOCK if clocked else None)
    try:
        h.session.capabilities = replace(h.session.capabilities, supports_concurrent_turn_requests=True)
        if not clocked:
            await h.run(append_audio())
            await h.run(commands.Commit(create_response=True))
            request_id = h.port.submissions[-1].context.request_id
            h.deliver(_speak_segment_end(request_id), stage_id=0, segment_finished=True)
            await h.settle()
            assert h.runner.run.concurrent_turn_requests_released, "the gate opens as before without the clock"
            return

        request_id = await _defer_a_commit_behind_the_active_response(h)
        assert not h.runner.run.concurrent_turn_requests_released

        events = await _finish_turn(h, request_id)

        assert [a.input_index for a in _acks(events)] == [2, 3]
        assert len(h.port.submissions) == 2, "the deferred turn was submitted once the response ended"
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_a_deferred_commit_cleared_before_submission_is_settled_as_cancelled() -> None:
    h = await open_harness(auto_response=False, extra_body=INPUT_CLOCK)
    try:
        request_id = await _defer_a_commit_behind_the_active_response(h)
        events = await h.run(commands.ClearInput())
        assert "input_audio_buffer.cleared" in types(events)
        assert _acks(events) == [], "the acknowledgements before it still wait for the speaking unit"

        acks = _acks(await _finish_turn(h, request_id))

        assert [a.input_index for a in acks] == [2, 3, 4]
        assert [_units(a) for a in acks] == [[("speak", None)], [], [("cancelled", "input_cleared")]]
        assert len(h.port.submissions) == 1, "the cleared turn is never submitted"
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_a_deferred_commit_dropped_with_a_short_interjection_is_settled_as_cancelled() -> None:
    h = await open_harness(auto_response=False, extra_body=INPUT_CLOCK)
    try:
        request_id = await _defer_a_commit_behind_the_active_response(h)
        h.session.reset_overlap_speech()  # the next interjection is too short to be a turn
        await h.run(append_audio(samples=3200))
        events = await h.run(commands.Commit(create_response=True))
        assert any(getattr(e, "type", "") == "input_audio_buffer.committed" for e in events)
        assert h.runner.model_state.committed_audio_payload is None, "the deferred turn's audio went with it"

        acks = _acks(await _finish_turn(h, request_id))

        assert [a.input_index for a in acks] == [2, 3, 4, 5, 6], "nothing waits for the dropped turn"
        assert _units(acks[0]) == [("speak", None)]
        assert _units(acks[2]) == [("cancelled", "short_overlap_discarded")], "with the commit that deferred it"
        assert len(h.port.submissions) == 1
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_a_cancel_that_keeps_a_deferred_turns_audio_still_settles_its_commit() -> None:
    """``output_audio_buffer.clear`` cancels the response but keeps the committed audio.

    No response is left whose end would submit the deferred turn, so its
    commit is settled with the cancel; the kept audio is submitted by the next
    ``response.create``, which is acknowledged once that turn is decided.
    """
    h = await open_harness(auto_response=False, extra_body=INPUT_CLOCK)
    try:
        await _defer_a_commit_behind_the_active_response(h)

        events = await h.run(commands.ClearOutputAudio())

        acks = _acks(events)
        assert [a.input_index for a in acks] == [2, 3, 4], "the deferred commit is not left to the timeouts"
        assert [_units(a) for a in acks] == [[("cancelled", "output_audio_buffer_clear")], [], _units(acks[0])]
        assert h.runner.model_state.committed_audio_payload is not None, "the audio was kept, as without the clock"
        assert len(h.port.submissions) == 1

        events = await h.run(commands.CreateResponse())
        assert len(h.port.submissions) == 2 and _acks(events) == [], "it waits for the turn it submitted"
        (ack,) = _acks(await _deliver_listen(h, h.port.submissions[-1].context.request_id))
        assert (ack.input_index, ack.trigger, _units(ack)) == (5, "response.create", [("listen", None)])
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_a_deferred_turn_submitted_after_its_slot_timed_out_is_not_credited_to_the_next_input() -> None:
    h = await open_harness(auto_response=False, extra_body=INPUT_CLOCK)
    try:
        now = _fake_time(h)
        request_id = await _defer_a_commit_behind_the_active_response(h)
        now[0] += DEFAULT_UNIT_TIMEOUT_S
        assert [a.input_index for a in _acks(await _check_timeouts(h))] == [2, 3]  # the speaking unit
        now[0] += DEFAULT_UNIT_TIMEOUT_S
        (ack,) = _acks(await _check_timeouts(h))  # the deferred turn's reserved slot
        assert ack.input_index == 4 and _units(ack) == [("timed_out", "no_progress")]

        await _finish_turn(h, request_id)  # the response ends after all: the deferred turn is submitted
        assert len(h.port.submissions) == 2

        (ack,) = _acks(await h.run(append_audio(samples=3200)))
        assert ack.input_index == 5, "a buffer-only append does not wait for the late turn"
        late = await _deliver_listen(h, h.port.submissions[-1].context.request_id)
        assert "response.listen" in types(late) and _acks(late) == [], "the late turn's decision stays its own"
    finally:
        await close_harness(h)


# ---- cancellation ------------------------------------------------------------


@pytest.mark.parametrize(
    ("command", "reason"),
    [
        (commands.CancelResponse(), "client_cancelled"),
        (commands.BargeIn(), "barge_in"),
        (commands.ClearOutputAudio(), "output_audio_buffer_clear"),
        (commands.CancelInput(), "barge_in"),
    ],
    ids=["response.cancel", "barge_in", "output_audio_buffer.clear", "input.cancel"],
)
@pytest.mark.usefixtures("input_clock_model")
async def test_a_cancel_settles_the_owed_acknowledgements_after_its_own_events(command, reason: str) -> None:
    h = await open_harness(extra_body=INPUT_CLOCK)
    try:
        request_id = await _speaking_unit(h)
        cancelled_epoch = h.session.epoch

        events = await h.run(command)

        assert "response.done" in types(events)
        assert types(events)[-1] == "input_audio_buffer.processed"
        (ack,) = _acks(events)
        assert ack.input_index == 1 and _units(ack) == [("cancelled", reason)]
        assert h.session.epoch > cancelled_epoch
        # Late output of the cancelled epoch is dropped, and so is its accounting.
        late = await h.deliver_and_settle(
            tts_output(request_id, samples=24000, text="hello", tts_is_last_chunk=True),
            segment_finished=True,
            epoch=cancelled_epoch,
        )
        assert late == []
        # The next input is clocked normally.
        events = await h.run(append_audio())
        assert _acks(events) == []
        (ack,) = _acks(await _deliver_listen(h, h.stage0_request_id()))
        assert ack.input_index == 2 and ack.units == ({"end_ms": 2000, "decision": "listen"},)
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_a_barge_in_settles_appends_still_in_flight_as_cancelled() -> None:
    """One append parked in its submission, the next queued behind it: both are the barge-in's."""
    h = await open_harness(extra_body=INPUT_CLOCK)
    try:
        h.port.submit_gate = asyncio.Event()
        h.submit(append_audio())
        await asyncio.wait_for(h.port.submit_started.wait(), timeout=2.0)
        h.submit(append_audio())
        # The worker takes the second append and queues it behind the first.
        await _wait_until(lambda: len(h.runner.tasks.append_tasks) == 2)
        assert h.port.submissions == []

        events = await h.run(commands.BargeIn())

        acks = _acks(h.events)
        assert [a.input_index for a in acks] == [1, 2]
        assert [_units(a) for a in acks] == [[("cancelled", "barge_in")], [("cancelled", "barge_in")]]
        assert types(events)[-1] == "input_audio_buffer.processed"
        assert h.port.submissions == []
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_an_overlap_barge_in_drops_the_output_of_an_append_it_called_off_after_stage0_had_it() -> None:
    """A committed turn reached Stage 0, then the server barged in before the append returned.

    With no request, response, stream or playback left to cancel, the barge-in
    calls off only the append; it must still close the epoch, or the
    called-off submission's segment end would decide the next unit.
    """
    h = await open_harness(extra_body=INPUT_CLOCK)
    try:
        h.session.config.overlap_policy = "barge_in_on_speech"
        await h.run(append_audio(samples=3200))  # input 1: buffered
        submit = h.port.submit
        reached, release = asyncio.Event(), asyncio.Event()

        async def _reach_stage0_then_park(submission):
            result = await submit(submission)  # Stage 0 has it ...
            reached.set()
            await release.wait()  # ... but the append has not returned
            return result

        h.port.submit = _reach_stage0_then_park  # type: ignore[method-assign]
        h.submit(commands.Commit())  # input 2: the committed turn
        await asyncio.wait_for(reached.wait(), timeout=2.0)
        h.port.submit = submit  # type: ignore[method-assign]
        # Nothing else to cancel: an earlier listen-only append of the shared
        # resident Stage-0 request returned after this commit bound it, and its
        # compare-before-clear ``clear_request`` removed the binding.
        h.session.clear_request(h.session.active_request_id)
        assert h.session.active_response_id is None and h.runner.run.stream_request_id is None
        epoch = h.session.epoch
        called_off = h.port.submissions[-1].context.request_id

        await h.run(append_audio())  # input 3, overlapping speech: the server barges in

        assert h.session.epoch > epoch, "the called-off submission's epoch is closed"
        acks = _acks(h.events)
        assert [a.input_index for a in acks] == [1, 2] and _units(acks[1]) == [("cancelled", "barge_in")]
        late = await _deliver_listen_at(h, called_off, epoch)
        assert _acks(late) == [], "the called-off unit's segment end must not decide the next unit"

        events = await _deliver_listen(h, h.port.submissions[-1].context.request_id)

        assert types(events) == ["response.listen", "input_audio_buffer.processed"]
        assert events[-1].input_index == 3 and _units(events[-1]) == [("listen", None)]
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_on_a_non_resumable_model_the_acknowledgement_waits_for_the_response_continuation() -> None:
    """A finished final output of a turn-mode response spawns ``maybe_continue_response``.

    On a non-resumable model that continuation is what ends a response whose
    output carries no end-of-turn marker (it emits ``response.done``). The
    harness plugin ends the turn on the output itself, so the continuation is
    marked with a probe event; the acknowledgement must come after it.
    """
    h = await open_harness(auto_response=False, extra_body=INPUT_CLOCK)
    try:
        h.session.capabilities = replace(h.session.capabilities, supports_core_resumable_request=False)
        model = h.runner.model
        continue_response = model.maybe_continue_response

        async def _probed_continuation(**kwargs):
            h.runner._emit_error("continuation_probe", "the response continuation ran")
            await continue_response(**kwargs)

        model.maybe_continue_response = _probed_continuation  # type: ignore[method-assign]
        await h.run(append_audio())
        await h.run(commands.Commit(create_response=True))
        request_id = h.port.submissions[-1].context.request_id
        assert h.deliver(_speak_segment_end(request_id), stage_id=0, segment_finished=True) is False
        await h.deliver_and_settle(tts_output(request_id, samples=24000, text="he"))

        events = await _finish_turn(h, request_id)

        order = types(events)
        assert "response.done" in order and "error" in order and _acks(events)
        assert order.index("response.done") < order.index("input_audio_buffer.processed")
        assert order.index("error") < order.index("input_audio_buffer.processed")
    finally:
        await close_harness(h)


# ---- failure and teardown ----------------------------------------------------


@pytest.mark.usefixtures("input_clock_model")
async def test_closing_with_acknowledgements_owed_flushes_them_in_one_event_before_session_closed() -> None:
    h = await open_harness(extra_body=INPUT_CLOCK)
    try:
        await _speaking_unit(h)
        await h.run(append_audio(samples=3200))  # buffered, but behind the speaking unit

        events = await h.run(commands.CloseSession())

        assert types(events) == ["input_audio_buffer.processed", "session.closed"]
        assert (events[0].first_input_index, events[0].input_index) == (1, 2)
        assert _units(events[0]) == [("aborted", "client_close")]
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_a_runtime_close_flushes_the_owed_acknowledgements_before_session_closed() -> None:
    h = await open_harness(extra_body=INPUT_CLOCK)
    try:
        await _speaking_unit(h)
        await h.run(append_audio(samples=3200))

        await h.runner._close_from_runtime("runtime_stage_failed")
        events = await h.settle()

        assert types(events) == ["input_audio_buffer.processed", "session.closed"]
        assert (events[0].first_input_index, events[0].input_index) == (1, 2)
        assert _units(events[0]) == [("aborted", "runtime_stage_failed")]
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_an_idle_expiry_flushes_the_owed_acknowledgement_before_session_expired() -> None:
    lease_clock = {"now": 100.0}
    h = await open_harness(
        extra_body=INPUT_CLOCK,
        runtime_config=DuplexSessionRuntimeConfig(idle_ttl_s=1.0),
        clock=lambda: lease_clock["now"],
    )
    try:
        h.session.config.idle_timeout_s = 5.0
        await _speaking_unit(h)
        lease_clock["now"] += 10.0

        assert await h.manager.reap_expired() == 1
        events = await h.settle()

        assert types(events) == ["input_audio_buffer.processed", "session.expired"]
        assert events[0].input_index == 1 and _units(events[0]) == [("aborted", "idle_ttl_expired")]
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_an_append_that_fails_after_acceptance_settles_its_unit_before_the_close() -> None:
    h = await open_harness(extra_body=INPUT_CLOCK)
    try:

        async def _fail_after_acceptance(*args, **kwargs):
            raise RuntimeError("stage output projection failed")

        h.runner.model._send_model_output_events = _fail_after_acceptance  # type: ignore[method-assign]

        events = await h.run(append_audio())

        assert len(h.port.submissions) == 1, "the model accepted the append"
        acks = _acks(events)
        assert [a.input_index for a in acks] == [1]
        assert _units(acks[0]) == [("aborted", "runtime_append_task_failed")]
        assert types(events).index("input_audio_buffer.processed") < types(events).index("session.closed")
        assert types(events)[-1] == "session.closed"
    finally:
        await close_harness(h)


# ---- timeouts ----------------------------------------------------------------


@pytest.mark.usefixtures("input_clock_model")
async def test_a_unit_the_model_never_finishes_is_settled_by_the_liveness_valve() -> None:
    h = await open_harness(extra_body=INPUT_CLOCK)
    try:
        now = _fake_time(h)
        await h.run(append_audio())
        now[0] += DEFAULT_UNIT_TIMEOUT_S - 1
        assert _acks(await _check_timeouts(h)) == []

        now[0] += 1
        (ack,) = _acks(await _check_timeouts(h))
        assert ack.units == ({"end_ms": 1000, "decision": "timed_out", "reason": "no_progress"},)
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_a_unit_that_keeps_producing_but_never_completes_is_settled_at_its_maximum_age() -> None:
    h = await open_harness(extra_body=INPUT_CLOCK)
    try:
        now = _fake_time(h)
        request_id = await _speaking_unit(h)
        started = now[0]
        for elapsed in (DEFAULT_UNIT_TIMEOUT_S - 1, 2 * DEFAULT_UNIT_TIMEOUT_S - 2, DEFAULT_UNIT_MAX_AGE_S - 1):
            now[0] = started + elapsed  # streaming, never finishing the unit
            await h.deliver_and_settle(tts_output(request_id, samples=2400, text="la"))
            assert _acks(await _check_timeouts(h)) == []

        now[0] = started + DEFAULT_UNIT_MAX_AGE_S
        (ack,) = _acks(await _check_timeouts(h))
        assert ack.units == ({"end_ms": 1000, "decision": "timed_out", "reason": "max_age"},)
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_a_late_segment_end_of_a_timed_out_unit_is_not_credited_to_the_next_unit() -> None:
    h = await open_harness(extra_body=INPUT_CLOCK)
    try:
        now = _fake_time(h)
        await h.run(append_audio())
        now[0] += DEFAULT_UNIT_TIMEOUT_S
        (ack,) = _acks(await _check_timeouts(h))
        assert _units(ack) == [("timed_out", "no_progress")]
        assert _acks(await h.run(append_audio())) == []

        assert _acks(await _deliver_listen(h, h.stage0_request_id())) == [], "unit 1's own segment end, late"
        (ack,) = _acks(await _deliver_listen(h, h.stage0_request_id()))

        assert ack.input_index == 2 and _units(ack) == [("listen", None)]
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_late_audio_of_a_timed_out_speaking_unit_is_not_credited_to_the_next_unit() -> None:
    h = await open_harness(extra_body=INPUT_CLOCK)
    try:
        now = _fake_time(h)
        request_id = await _speaking_unit(h)
        now[0] += DEFAULT_UNIT_MAX_AGE_S
        h.runner._input_clock.note_progress()
        (ack,) = _acks(await _check_timeouts(h))
        assert _units(ack) == [("timed_out", "max_age")]
        await h.run(append_audio())
        assert h.deliver(_speak_segment_end(request_id), stage_id=0, segment_finished=True) is False
        await h.settle()

        assert _acks(await _deliver_audio_end(h, request_id)) == [], "the end of unit 1's audio, late"
        (ack,) = _acks(await _deliver_audio_end(h, request_id))

        assert ack.input_index == 2 and _units(ack) == [("speak", None)]
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_a_unit_timed_out_in_submission_keeps_its_segment_end_from_the_next_unit() -> None:
    """Unit 1 times out before Stage 0 accepted it; unit 2 queues behind it; unit 1's segment end
    overtakes its acceptance callback: it must not decide (and acknowledge) unit 2."""
    h = await open_harness(extra_body=INPUT_CLOCK)
    try:
        now = _fake_time(h)
        h.port.submit_gate = asyncio.Event()
        h.submit(append_audio())
        await asyncio.wait_for(h.port.submit_started.wait(), timeout=2.0)
        now[0] += DEFAULT_UNIT_TIMEOUT_S
        h.runner._on_input_clock_timer()
        (ack,) = _acks(await h.settle(timeout_s=0.3))
        assert _units(ack) == [("timed_out", "no_progress")]
        h.submit(append_audio())  # queued behind unit 1's submission
        await h.settle(timeout_s=0.3)

        events = await _deliver_listen_at(h, h.stage0_request_id(), epoch=0)  # unit 1's, ahead of its callback
        assert _acks(events) == [], "unit 2 has not even been submitted"

        h.port.submit_gate.set()
        await h.settle()
        assert len(h.port.submissions) == 2
        (ack,) = _acks(await _deliver_listen(h, h.stage0_request_id()))
        assert ack.input_index == 2 and _units(ack) == [("listen", None)]
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_a_failing_completion_hook_does_not_send_an_error_per_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(self, **kwargs):
        raise RuntimeError("plugin bug")

    monkeypatch.setattr(MiniCPMO45DuplexPlugin, "unit_output_complete", _boom)
    h = await open_harness(extra_body=INPUT_CLOCK)
    try:
        now = _fake_time(h)
        request_id = await _speaking_unit(h)
        for _ in range(5):
            await h.deliver_and_settle(tts_output(request_id, samples=2400, text="la"))
        assert [e for e in h.events if e.type == "error"] == []

        now[0] += DEFAULT_UNIT_TIMEOUT_S
        (ack,) = _acks(await _check_timeouts(h))
        assert _units(ack) == [("timed_out", "no_progress")], "left to the timeouts"
    finally:
        await close_harness(h)


# ---- sessions without the clock ----------------------------------------------


async def _clock_free_scenario(h) -> tuple[list[str], list[str]]:
    events = list(await h.run(append_audio()))
    request_id = h.stage0_request_id()
    events += await _deliver_listen(h, request_id)
    events += await h.run(append_audio())
    assert h.deliver(_speak_segment_end(request_id), stage_id=0, segment_finished=True) is False
    events += await h.deliver_and_settle(tts_output(request_id, samples=24000, text="he"))
    events += await h.deliver_and_settle(
        tts_output(request_id, samples=48000, text="hello", tts_is_last_chunk=True, finished=True),
        segment_finished=True,
    )
    events += await h.run(append_audio(samples=3200))
    events += await h.run(commands.Commit())
    events += await h.run(commands.CancelResponse())
    events += await h.run(append_audio(samples=3200))
    # Refused at admission, then by the runner.
    events += await h.run(replace(append_audio(), audio=b""))
    events += await h.run(commands.AppendAudio(audio=b"\x00" * 3, format="pcm16", sample_rate_hz=16000))
    events += await h.run(commands.ClearInput())
    events += await h.run(commands.CloseSession())
    return types(events), [s.context.request_id for s in h.port.submissions]


async def test_without_the_clock_nothing_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A session that does not ask for the clock takes none of its paths and gets the same events."""
    h = await open_harness()
    queued: list = []
    put_nowait = h.runner._mailbox.put_nowait

    def _spy(item):
        queued.append(item)
        put_nowait(item)

    h.runner._mailbox.put_nowait = _spy  # type: ignore[method-assign]
    try:
        assert h.runner._input_clock is None and not h.runner.input_clocked
        with_feature = await _clock_free_scenario(h)
        assert h.runner._input_clock_appends == {} and h.runner._input_clock_timer is None
    finally:
        await close_harness(h)

    assert "input_audio_buffer.processed" not in with_feature[0]
    assert not any(isinstance(item, StageProgress) or type(item).__name__ == "_ClockedInput" for item in queued)
    assert not any(str(getattr(item, "kind", "")).startswith("input_clock") for item in queued)

    # The same scenario with every clock call site of the runner stubbed out.
    for name in (
        "input_refused",
        "_settle_cancelled_units",
        "_release_dropped_deferred_turn",
        "_close_input_clock",
        "_track_input_clock_append",
        "_arm_input_clock_timer",
    ):
        monkeypatch.setattr(DuplexSessionRunner, name, lambda self, *args: None)
    monkeypatch.setattr(DuplexSessionRunner, "_input_clock_progress", lambda self, *args: None)
    baseline = await open_harness()
    try:
        assert await _clock_free_scenario(baseline) == with_feature
    finally:
        await close_harness(baseline)


# ---- capability, mode lock and idle window -----------------------------------


async def test_a_model_that_has_not_opted_in_refuses_an_input_clocked_session(monkeypatch: pytest.MonkeyPatch) -> None:
    assert MiniCPMO45DuplexPlugin.supports_input_clock is False
    managers: list[DuplexSessionManager] = []
    init = DuplexSessionManager.__init__

    def _record(self: DuplexSessionManager, *args: object, **kwargs: object) -> None:
        init(self, *args, **kwargs)  # type: ignore[arg-type]
        managers.append(self)

    monkeypatch.setattr(DuplexSessionManager, "__init__", _record)
    error_codes: list[str] = []
    control_error = DuplexSessionManager._control_error

    def _record_error(error: BaseException) -> tuple[str, str, bool]:
        result = control_error(error)
        error_codes.append(result[0])
        return result

    monkeypatch.setattr(DuplexSessionManager, "_control_error", staticmethod(_record_error))
    try:
        with pytest.raises(AssertionError):  # the harness insists on a successful open
            await open_harness(extra_body=INPUT_CLOCK)
        assert set(error_codes) == {"input_clock_unsupported"}
        assert SESSION_ID not in managers[0].runners
    finally:
        for manager in managers:
            await manager.shutdown()

    h = await open_harness()
    try:
        events = await h.run(commands.UpdateSession(patch={"extra_body": INPUT_CLOCK}, event_id="evt-clock"))
        assert [(e.type, e.code, e.related_event_id) for e in events] == [
            ("error", "input_clock_unsupported", "evt-clock")
        ]
        assert not h.runner.input_clocked
    finally:
        await close_harness(h)


@pytest.mark.parametrize(
    ("opened", "patch"),
    [
        (INPUT_CLOCK, {"clock": None}),
        ({}, INPUT_CLOCK),
        (INPUT_CLOCK, {"input_clock_unit_timeout_s": 3}),
        (INPUT_CLOCK, {"input_clock_unit_max_s": 120}),
    ],
    ids=["on-to-off", "off-to-on", "unit-timeout", "unit-max-age"],
)
@pytest.mark.usefixtures("input_clock_model")
async def test_the_input_clock_is_fixed_at_session_creation(opened: dict, patch: dict) -> None:
    h = await open_harness(extra_body=opened)
    try:
        events = await h.run(commands.UpdateSession(patch={"extra_body": patch}, event_id="evt-clock"))
        errors = [event for event in events if event.type == "error"]
        assert [(e.code, e.related_event_id) for e in errors] == [("input_clock_update_unsupported", "evt-clock")]
        assert "session.updated" not in types(events)
        assert h.runner.input_clocked is bool(opened)
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_an_invalid_input_clock_timeout_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    managers: list[DuplexSessionManager] = []
    init = DuplexSessionManager.__init__

    def _record(self: DuplexSessionManager, *args: object, **kwargs: object) -> None:
        init(self, *args, **kwargs)  # type: ignore[arg-type]
        managers.append(self)

    monkeypatch.setattr(DuplexSessionManager, "__init__", _record)
    error_codes: list[str] = []
    control_error = DuplexSessionManager._control_error

    def _record_error(error: BaseException) -> tuple[str, str, bool]:
        result = control_error(error)
        error_codes.append(result[0])
        return result

    monkeypatch.setattr(DuplexSessionManager, "_control_error", staticmethod(_record_error))
    try:
        with pytest.raises(AssertionError):  # the harness insists on a successful open
            await open_harness(extra_body={**INPUT_CLOCK, "input_clock_unit_timeout_s": 0})
        assert set(error_codes) == {"invalid_duplex_runtime_config"}
    finally:
        for manager in managers:
            await manager.shutdown()

    h = await open_harness(extra_body=INPUT_CLOCK)
    try:
        patch = {"extra_body": {**INPUT_CLOCK, "input_clock_unit_max_s": "60"}}
        events = await h.run(commands.UpdateSession(patch=patch, event_id="evt-bad"))
        errors = [e for e in events if e.type == "error"]
        assert [(e.code, e.related_event_id) for e in errors] == [("invalid_duplex_runtime_config", "evt-bad")]
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_a_session_update_that_keeps_the_clock_is_accepted() -> None:
    h = await open_harness(extra_body=INPUT_CLOCK)
    try:
        events = await h.run(
            commands.UpdateSession(patch={"extra_body": {**INPUT_CLOCK, "overlap_policy": "barge_in"}})
        )
        assert "session.updated" in types(events) and "error" not in types(events)
        assert _acks(await h.run(append_audio(samples=3200)))
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_an_input_clocked_session_expires_only_after_its_own_idle_window() -> None:
    """The lease TTL (1 s here) does not apply; the session's idle window (default 10 min) does."""
    assert INPUT_CLOCK_IDLE_TIMEOUT_S == 600.0
    clock = {"now": 100.0}
    h = await open_harness(
        extra_body=INPUT_CLOCK,
        runtime_config=DuplexSessionRuntimeConfig(idle_ttl_s=1.0),
        clock=lambda: clock["now"],
    )
    try:
        h.session.config.idle_timeout_s = 5.0
        await h.run(append_audio(samples=3200))
        clock["now"] += 4.0
        assert await h.manager.reap_expired() == 0, "a client pausing inside its window keeps the session"
        assert SESSION_ID in h.manager.runners
        clock["now"] += 2.0
        assert await h.manager.reap_expired() == 1
        expired = [event for event in await h.settle() if event.type == "session.expired"]
        assert [event.reason for event in expired] == ["idle_ttl_expired"]
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_a_long_client_idle_window_is_capped_on_the_engine_lease() -> None:
    clock = {"now": 100.0}
    h = await open_harness(
        extra_body=INPUT_CLOCK,
        runtime_config=DuplexSessionRuntimeConfig(idle_ttl_s=300.0),
        clock=lambda: clock["now"],
    )
    try:
        h.session.config.idle_timeout_s = 86_400.0
        await h.run(append_audio(samples=3200))
        clock["now"] += INPUT_CLOCK_IDLE_TIMEOUT_S - 1
        assert await h.manager.reap_expired() == 0
        clock["now"] += 2.0
        assert await h.manager.reap_expired() == 1, "max(idle_ttl_s, 600 s) caps the client's window"
    finally:
        await close_harness(h)


@pytest.mark.usefixtures("input_clock_model")
async def test_a_disconnected_input_clocked_session_is_still_reaped_after_the_disconnect_grace() -> None:
    """Only the idle expiry of a connected session is skipped: a client that went away releases its slot."""
    clock = {"now": 100.0}
    h = await open_harness(
        extra_body=INPUT_CLOCK,
        runtime_config=DuplexSessionRuntimeConfig(idle_ttl_s=1.0, disconnect_grace_s=2.0),
        clock=lambda: clock["now"],
    )
    try:
        await h.run(append_audio(samples=3200))
        h.session.detach_lease()  # what the serving layer does when a resumable session's socket drops
        clock["now"] += 1.5
        assert await h.manager.reap_expired() == 0, "neither the idle window nor the grace has run out"
        clock["now"] += 1.0
        assert await h.manager.reap_expired() == 1
        assert SESSION_ID not in h.manager.runners
        expired = [event for event in await h.settle() if event.type == "session.expired"]
        assert [event.reason for event in expired] == ["disconnect_grace_expired"]
    finally:
        await close_harness(h)


def test_a_realtime_session_with_the_input_clock_gets_a_ten_minute_idle_window() -> None:
    explicit = DuplexSessionConfig.from_realtime({"idle_timeout_s": 30, "extra_body": {"clock": "input"}})
    assert explicit.idle_timeout_s == 30.0
    config = DuplexSessionConfig.from_realtime({"extra_body": {"clock": "input"}})
    assert config.idle_timeout_s == INPUT_CLOCK_IDLE_TIMEOUT_S
    assert DuplexSessionConfig.from_realtime({}).idle_timeout_s == 300.0
    assert DuplexSessionConfig.from_event({"session": {"extra_body": {"clock": "input"}}}).idle_timeout_s == 600.0
