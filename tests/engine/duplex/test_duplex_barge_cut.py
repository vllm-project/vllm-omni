# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""``barge_cut_on_model_yield``: cut the reply's tail when the model yields to a new user utterance."""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace

import numpy as np
import pytest

from tests.engine.duplex.test_session_runner import (
    LISTEN_TOKEN_ID,
    append_audio,
    close_harness,
    find,
    listen_output,
    open_harness,
    tts_output,
    types,
)
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.config import DuplexSessionState
from vllm_omni.engine.duplex.session.pacing import SessionPacing

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _pace(**runtime: object) -> SessionPacing:
    config: dict[str, object] = {"barge_cut_on_model_yield": True}
    config.update(runtime)
    pace = SessionPacing(DuplexSessionRuntimeConfig(**config), SimpleNamespace(active_response_id="resp_1"))
    for index in range(10):
        pace.on_audio_emit(0.1 * index, 1.0)  # ~10 s of reply audio queued at the client
    return pace


def _feed(pace: SessionPacing, t: float, *, quiet_s: float, speech_s: float, step: float = 0.2) -> float:
    for _ in range(round(quiet_s / step)):
        pace.on_client_append(t, is_speech=False, duration_s=step)
        t += step
    for _ in range(round(speech_s / step)):
        pace.on_client_append(t, is_speech=True, duration_s=step)
        t += step
    return t


def test_new_utterance_over_playing_audio_arms_after_the_speech_floor() -> None:
    pace = _pace()
    t = _feed(pace, 1.2, quiet_s=0.8, speech_s=0.2)
    assert pace.armed is False  # a 0.2 s backchannel stays below the 0.4 s floor
    t = _feed(pace, t, quiet_s=0.0, speech_s=0.2)
    assert pace.armed is True
    assert pace.cut_ready(t) is True
    pace.session.active_response_id = None  # the reply ended meanwhile
    assert pace.cut_ready(t) is False


def test_a_short_pause_inside_the_interruption_does_not_reset_it() -> None:
    pace = _pace()
    t = _feed(pace, 1.2, quiet_s=0.8, speech_s=0.2)
    t = _feed(pace, t, quiet_s=0.2, speech_s=0.2)  # one low-energy 200 ms chunk mid-sentence
    assert pace.armed is True


def test_a_long_pause_ends_the_utterance() -> None:
    pace = _pace()
    t = _feed(pace, 1.2, quiet_s=0.8, speech_s=0.2)
    # 0.4 s pause: the run ends; the next 0.2 s has too little quiet before it to start a new one.
    _feed(pace, t, quiet_s=0.4, speech_s=0.2)
    assert pace.armed is False


@pytest.mark.parametrize(
    ("start", "quiet_s", "speech_s", "runtime"),
    [
        (1.2, 0.4, 1.0, {}),  # not enough quiet before the speech
        (0.2, 0.8, 1.0, {"barge_arm_min_playback_s": 2.0}),  # reply not audible long enough
        (1.2, 0.8, 0.2, {}),  # backchannel
        (1.2, 0.8, 1.0, {"barge_cut_on_model_yield": False}),
    ],
)
def test_arming_conditions(start: float, quiet_s: float, speech_s: float, runtime: dict) -> None:
    pace = _pace(**runtime)
    _feed(pace, start, quiet_s=quiet_s, speech_s=speech_s)
    assert pace.armed is False


def test_continuous_speech_from_before_the_reply_never_arms() -> None:
    session = SimpleNamespace(active_response_id=None)
    pace = SessionPacing(DuplexSessionRuntimeConfig(barge_cut_on_model_yield=True), session)
    # The user is still talking when the reply's first audio arrives.
    t = _feed(pace, 0.0, quiet_s=0.0, speech_s=1.0)
    session.active_response_id = "resp_1"
    pace.on_audio_emit(t, 1.0)
    _feed(pace, t, quiet_s=0.0, speech_s=2.0)
    assert pace.armed is False


def test_the_replay_keeps_the_latest_units_of_the_arming_utterance() -> None:
    pace = _pace()
    pace.on_input_unit({"seq": "before"})  # no utterance running: not kept
    t = _feed(pace, 1.2, quiet_s=0.8, speech_s=0.2)
    for index in range(8):
        pace.on_input_unit({"seq": index})
    assert pace.take_replay() == []  # not armed: nothing to replay
    for index in range(8):
        pace.on_input_unit({"seq": index})
    _feed(pace, t, quiet_s=0.0, speech_s=0.4)
    assert pace.armed is True
    assert [unit["seq"] for unit in pace.take_replay()] == [2, 3, 4, 5, 6, 7]
    assert pace.take_replay() == []


def test_silence_after_the_armed_utterance_never_pushes_its_start_out_of_the_replay() -> None:
    pace = _pace()
    t = _feed(pace, 1.2, quiet_s=0.8, speech_s=0.6)
    assert pace.armed is True
    for index in range(4):
        pace.on_input_unit({"seq": f"speech{index}", "is_speech": True})
    _feed(pace, t, quiet_s=1.0, speech_s=0.0)  # the utterance ended; the model has not yielded yet
    for index in range(4):
        pace.on_input_unit({"seq": f"quiet{index}", "is_speech": False})
    # Trailing silence fills the free slots only (the model yields on it); speech still evicts the oldest.
    pace.on_input_unit({"seq": "speech4", "is_speech": True})
    assert [unit["seq"] for unit in pace.take_replay()] == [
        "speech1",
        "speech2",
        "speech3",
        "quiet0",
        "quiet1",
        "speech4",
    ]


def test_take_replay_forgets_the_arming_of_a_reply_that_already_ended() -> None:
    """The route-M listen abort takes the replay without asking ``cut_ready``: a stale arming is not replayed."""
    pace = _pace()
    _feed(pace, 1.2, quiet_s=0.8, speech_s=0.6)
    pace.on_input_unit({"seq": 0})
    assert pace.armed is True
    pace.session.active_response_id = "resp_2"  # a new reply opened; nothing synced the pacing since
    assert pace.take_replay() == []
    assert pace.armed is False


def test_cut_needs_unplayed_audio_and_disarms_on_epoch_and_new_reply() -> None:
    pace = _pace()
    t = _feed(pace, 1.2, quiet_s=0.8, speech_s=0.6)
    assert pace.armed is True
    assert pace.cut_ready(100.0) is False  # everything played out
    pace.on_epoch_advance(t)
    assert pace.armed is False
    pace = _pace()
    _feed(pace, 1.2, quiet_s=0.8, speech_s=0.6)
    pace.session.active_response_id = "resp_2"
    assert pace.cut_ready(2.0) is False
    assert pace.armed is False


async def _armed_harness(**runtime: object):
    config: dict[str, object] = {"barge_cut_on_model_yield": True, "barge_arm_min_playback_s": 0.0}
    config.update(runtime)
    h = await open_harness(runtime_config=DuplexSessionRuntimeConfig(**config))
    await h.run(append_audio())
    request_id = h.stage0_request_id()
    spoken = await h.deliver_and_settle(tts_output(request_id, samples=240000, text="hello", tts_is_last_chunk=True))
    response_id = find(spoken, "response.created").response_id
    await h.run(append_audio(12800, value=0.0, is_speech=False))
    await h.run(append_audio(8000, value=0.05, is_speech=True))
    return h, request_id, response_id


def _unit_audio(submission) -> str:
    return submission.prompt["model_intermediate_buffer"]["duplex"]["payload"]["audio"]


def _epoch_units(h, epoch: int) -> list[str]:
    """The audio of every Stage-0 append submitted in ``epoch``, in submission order."""
    return [_unit_audio(sub) for sub in h.port.submissions if f".e.{epoch}." in sub.context.request_id]


def _samples(audio: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(audio), dtype="<f4")


async def _model_listens(h, request_id: str):
    return await h.deliver_and_settle(
        listen_output(request_id),
        stage_id=0,
        segment_finished=True,
        segment_token_ids=[11, 12, LISTEN_TOKEN_ID],
        segment_output_metadata={"meta.listen_token_id": LISTEN_TOKEN_ID},
    )


async def _cut_with_an_append_in_flight(h, request_id: str):
    """Park the next unit mid-submit, queue one more behind it, then let the model yield (the cut).

    Returns the replay's audio and the events so far; the parked append
    resumes once the caller sets ``h.port.submit_gate``.
    """
    h.port.submit_gate = asyncio.Event()
    h.submit(append_audio(11200, value=0.06))  # 0.3 s buffered + 0.7 s: a unit, parked in the stage port
    await asyncio.wait_for(h.port.submit_started.wait(), timeout=2.0)
    h.submit(append_audio(16000, value=0.07))  # the next unit, queued behind it
    events = await h.settle(timeout_s=0.3)  # the parked append keeps the runner busy
    replay = [unit["audio"] for unit in h.runner.run.pace._utterance_units]
    assert len(replay) == 3
    h.deliver(
        listen_output(request_id),
        stage_id=0,
        segment_finished=True,
        segment_token_ids=[11, 12, LISTEN_TOKEN_ID],
        segment_output_metadata={"meta.listen_token_id": LISTEN_TOKEN_ID},
    )
    events += await h.settle(timeout_s=0.3)
    assert h.session.epoch == 1
    return replay, events


@pytest.mark.asyncio
async def test_model_listen_cuts_an_armed_reply_without_tts_in_flight() -> None:
    h, request_id, response_id = await _armed_harness()
    try:
        assert h.runner.run.pace.armed is True
        events = await _model_listens(h, request_id)
        cancelled = find(events, "response.done")
        assert cancelled.status == "cancelled"
        assert cancelled.response_id == response_id
        assert h.session.epoch == 1
        assert h.port.aborts and h.port.aborts[0] == [request_id]
        assert h.runner.run.pace.cuts == 1
        assert h.runner.run.pace.armed is False
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_the_cut_replays_the_interrupting_utterance_into_the_new_epoch() -> None:
    """Stage 0 drops the session context with the aborted request: the utterance's units go again."""
    h, request_id, _ = await _armed_harness()
    try:
        # The 0.5 s of speech completed a unit (0.8 s quiet + 0.2 s speech) Stage 0 took in epoch 0.
        assert len(h.runner.run.pace._utterance_units) == 1
        await _model_listens(h, request_id)
        assert h.session.epoch == 1
        assert h.runner.tasks.append_tail is not None
        assert await h.runner.tasks.append_tail
        new_epoch = [sub for sub in h.port.submissions if sub.context.request_id != request_id]
        assert len(new_epoch) == 1
        assert ".e.1." in new_epoch[0].context.request_id
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_natural_turn_end_cuts_an_armed_reply() -> None:
    h, request_id, response_id = await _armed_harness()
    try:
        natural = SimpleNamespace(
            request_id=request_id,
            finished=True,
            outputs=[SimpleNamespace(text="hello", token_ids=[], multimodal_output={})],
            multimodal_output={"meta.turn_eos_token_id": 9},
        )
        h.deliver(
            natural,
            stage_id=0,
            segment_finished=True,
            segment_token_ids=[9, 10],
            segment_output_metadata={"meta.turn_eos_token_id": 9},
        )
        events = await h.settle()
        cancelled = find(events, "response.done")
        assert cancelled.status == "cancelled"
        assert cancelled.response_id == response_id
        assert h.session.epoch == 1
        tail = await h.deliver_and_settle(tts_output(request_id, samples=24000, text="tail"), epoch=0)
        assert "response.output_audio.delta" not in types(tail)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_unarmed_reply_drains_as_before() -> None:
    # Speech straight after the reply opened (no quiet run): not a new utterance.
    h = await open_harness(
        runtime_config=DuplexSessionRuntimeConfig(barge_cut_on_model_yield=True, barge_arm_min_playback_s=0.0)
    )
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        await h.deliver_and_settle(tts_output(request_id, samples=240000, text="hello", tts_is_last_chunk=True))
        response_id = h.session.active_response_id
        await h.run(append_audio(8000, value=0.05, is_speech=True))
        assert h.runner.run.pace.armed is False
        await _model_listens(h, request_id)
        assert h.port.aborts == []
        assert h.session.epoch == 0
        assert h.session.active_response_id == response_id
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_switch_off_drains_the_same_utterance_as_before() -> None:
    """The armed sequence with the switch off: no pacing state, the listen drains the reply."""
    h, request_id, response_id = await _armed_harness(barge_cut_on_model_yield=False)
    try:
        assert h.runner.run.pace is None
        events = await _model_listens(h, request_id)
        assert h.port.aborts == []
        assert h.session.epoch == 0
        assert "audio.cancelled" not in types(events)
        assert h.session.active_response_id == response_id
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_the_replay_goes_before_the_buffered_tail_and_the_tail_goes_once() -> None:
    """Audio still buffered at the cut is no unit yet: not replayed, it opens the new epoch after the replay."""
    h, request_id, _ = await _armed_harness()
    try:
        replay = [unit["audio"] for unit in h.runner.run.pace._utterance_units]
        assert h.runner.model_state.audio_buffer.pending_byte_count == 4800 * 4  # 0.3 s of the interruption
        await _model_listens(h, request_id)
        assert h.session.epoch == 1
        await h.run(append_audio(11200, value=0.07))
        units = _epoch_units(h, 1)
        assert units[: len(replay)] == replay
        assert len(units) == len(replay) + 1
        tail = _samples(units[-1])
        assert tail.size == 16000
        assert np.allclose(tail[:4800], 0.05) and np.allclose(tail[4800:], 0.07)
        assert h.session.pending_input_bytes == 0
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_the_replay_includes_the_unit_a_commit_flushed_before_the_cut() -> None:
    """An explicit commit flushes the utterance's last, zero-padded unit into the old epoch: the cut replays it."""
    h, request_id, _ = await _armed_harness()
    try:
        events = await h.run(commands.Commit())
        assert "error" not in types(events)
        flushed = _epoch_units(h, 0)[-1]
        samples = _samples(flushed)
        assert samples.size == 16000
        assert np.allclose(samples[:4800], 0.05) and not samples[4800:].any()
        replay = [unit["audio"] for unit in h.runner.run.pace._utterance_units]
        assert len(replay) == 2 and replay[-1] == flushed
        await _model_listens(h, request_id)
        assert h.session.epoch == 1
        assert h.runner.tasks.append_tail is not None and await h.runner.tasks.append_tail
        new_epoch = [sub for sub in h.port.submissions if ".e.1." in sub.context.request_id]
        assert [_unit_audio(sub) for sub in new_epoch] == replay
        last = new_epoch[-1].prompt["model_intermediate_buffer"]["duplex"]
        assert last["final"] is False and "final" not in last["payload"]
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_units_still_in_the_append_chain_at_the_cut_reach_the_new_epoch_once() -> None:
    """Units planned in the old epoch but not accepted there are dropped by the epoch check: the replay delivers them."""
    h, request_id, _ = await _armed_harness()
    try:
        replay, _ = await _cut_with_an_append_in_flight(h, request_id)
        h.port.submit_gate.set()
        await h.settle()
        assert _epoch_units(h, 1) == replay
        assert replay[-1] not in _epoch_units(h, 0)  # the queued unit never reached the old epoch's Stage 0
        assert h.session.pending_input_bytes == 0
        assert h.runner.model_state.audio_buffer.pending_byte_count == 0
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_an_append_in_flight_at_the_cut_does_not_fail_the_replay_behind_it() -> None:
    """The parked append fails its epoch check after submit: dropped like a queued one, not a runtime failure.

    Failing it rolled its PCM back and stopped the append chain, so the whole
    replay behind it was abandoned silently, the client got a non-retryable
    ``runtime_append_failed`` and its next commit ``commit_aborted``.
    """
    h, request_id, _ = await _armed_harness()
    try:
        replay, events = await _cut_with_an_append_in_flight(h, request_id)
        h.port.submit_gate.set()
        events += await h.settle()
        assert "error" not in types(events)
        tail = h.runner.tasks.append_tail
        assert tail is not None and tail.done() and tail.result() is True
        assert _epoch_units(h, 1) == replay
        events = await h.run(commands.Commit())
        assert "error" not in types(events)
        assert "input_audio_buffer.committed" in types(events)
        assert h.session.state == DuplexSessionState.OPEN
    finally:
        await close_harness(h)


async def _cut_after_a_commit(h, request_id: str) -> list[str]:
    """Armed, the client commits (flushing its buffer) and stops sending; the model listens: the cut."""
    await h.run(commands.Commit())
    replay = [unit["audio"] for unit in h.runner.run.pace._utterance_units]
    await _model_listens(h, request_id)
    assert h.session.epoch == 1
    assert _epoch_units(h, 1) == replay
    return replay


@pytest.mark.asyncio
async def test_a_quiet_client_still_gets_silence_units_after_the_cut() -> None:
    """The cut reply's continuations kept Stage 0 of a quiet client moving; the replayed epoch has none.

    Without them nothing would submit another unit after the replay's
    listen: no answer until the client sends audio again.
    """
    h, request_id, _ = await _armed_harness()
    try:
        replay = await _cut_after_a_commit(h, request_id)
        events = await _model_listens(h, h.stage0_request_id())  # Stage 0 listens on the replay
        assert "response.listen" in types(events)
        units = _epoch_units(h, 1)
        assert len(units) == len(replay) + 1
        assert not _samples(units[-1]).any()
        assert h.runner.model_state.continuation_units == 1
        assert h.session.active_response_id is None
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_a_client_that_keeps_streaming_after_the_cut_gets_no_silence_units() -> None:
    h, request_id, _ = await _armed_harness()
    try:
        replay = await _cut_after_a_commit(h, request_id)
        await h.run(append_audio(16000, value=0.0, is_speech=False))  # the microphone stays on
        await _model_listens(h, h.stage0_request_id())
        assert len(_epoch_units(h, 1)) == len(replay) + 1  # the client's own unit, no silence after it
        assert h.runner.model_state.continuation_units == 0
    finally:
        await close_harness(h)
