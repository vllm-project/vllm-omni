# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Cancellation closes a native reply in order, retaining its resident request."""

import asyncio

import pybase64 as base64
import pytest

from tests.engine.duplex.test_minicpmo_input_clock import _speak_segment_end
from tests.engine.duplex.test_session_runner import (
    append_audio,
    close_harness,
    find,
    open_harness,
    tts_output,
    types,
)
from tests.engine.duplex.test_session_runner_minicpmo_thinker_only import (
    CHUNK_EOS,
    LISTEN,
    SPEAK,
    TURN_EOS,
    A,
    _stage0,
    _thinker_only_plugin,
)
from tests.engine.test_duplex_orchestrator import (
    _append_audio,
    _build,
    _close,
    _open,
    _settle,
    _stage0_request_id,
    _submit,
)
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.plugin import DuplexInputCapacityError
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import MiniCPMO45DuplexPlugin
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import INTERRUPTION_KEY
from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _plugin(thinker_only: bool) -> MiniCPMO45DuplexPlugin:
    plugin = _thinker_only_plugin()
    if not thinker_only:
        plugin.configure_pipeline(MINICPMO_4_5_PIPELINE)
    return plugin


@pytest.mark.asyncio
@pytest.mark.parametrize("thinker_only", [False, True])
@pytest.mark.parametrize("cancel", [commands.CancelResponse(), commands.BargeIn(), commands.ClearOutputAudio()])
async def test_cancel_keeps_request_context_and_filters_late_output(thinker_only, cancel):
    h = await open_harness(
        plugin=_plugin(thinker_only),
        stage_count=1 if thinker_only else 2,
        extra_body={"silence_continuation": False},
    )
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        old_tokens = [SPEAK, A, CHUNK_EOS]
        if thinker_only:
            await h.deliver_and_settle(_stage0(request_id, old_tokens, "嗯，", stop=True), stage_id=0)
        else:
            await h.deliver_and_settle(tts_output(request_id, text="hello"))
        old_response = h.session.active_response_id
        assert old_response is not None
        events = await h.run(cancel)
        assert find(events, "response.done").status == "cancelled"
        assert h.session.epoch == 0
        assert h.session.active_request_id == request_id
        assert h.runner.run.stream_request_id == request_id
        assert not h.port.aborts and not h.port.cleanups
        control = h.port.submissions[-1]
        duplex = control.prompt["model_intermediate_buffer"]["duplex"]
        assert duplex["payload"][INTERRUPTION_KEY] is True
        assert duplex["payload"]["force_listen"] is True
        assert duplex["payload"]["type"] == "text"
        assert control.context.request_id == request_id and control.already_submitted
        assert duplex["seq"] == 2

        # Old speech can arrive after cancellation, before the control unit.
        events = await h.deliver_and_settle(tts_output(request_id, text="late"))
        assert not {
            "response.created",
            "response.output_audio_transcript.delta",
            "response.output_audio.delta",
        }.intersection(types(events))
        control_output = _stage0(request_id, [*old_tokens, LISTEN] if thinker_only else [LISTEN], "", stop=True)
        control_output.multimodal_output[INTERRUPTION_KEY] = True
        await h.deliver_and_settle(control_output, stage_id=0, segment_finished=True, model_turn_id=0)
        assert not h.runner.run.interruption_pending
        assert h.session.turn_id == 1

        # A downstream tail still belongs to turn 0 after the control is done.
        if not thinker_only:
            events = await h.deliver_and_settle(tts_output(request_id, text="tail", turn_end=True))
            assert "response.created" not in types(events)
        await h.run(append_audio())
        assert h.port.submissions[-1].context.request_id == request_id
        assert h.port.submissions[-1].already_submitted
        assert h.port.submissions[-1].prompt["model_intermediate_buffer"]["duplex"]["seq"] == 3
        if thinker_only:
            tokens = [*old_tokens, LISTEN, SPEAK, A, CHUNK_EOS]
            events = await h.deliver_and_settle(
                _stage0(request_id, tokens, "嗯，", stop=True), stage_id=0, model_turn_id=1
            )
        else:
            events = await h.deliver_and_settle(tts_output(request_id, samples=48000, text="new answer", turn_id=1))
        assert "response.created" in types(events)
        assert h.session.active_response_id != old_response
        assert not h.port.aborts and not h.port.cleanups
        await h.run(commands.CloseSession())
        assert h.port.aborts or any(abort for _, abort in h.port.cleanups)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_clear_after_full_playback_skips_control_until_new_input_arrives():
    h = await open_harness(plugin=_plugin(False), extra_body={"silence_continuation": False})
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        events = await h.deliver_and_settle(tts_output(request_id, turn_end=True))
        response_id = find(events, "response.done").response_id
        await h.run(commands.AckPlayback(response_id=response_id, played_ms=1000))
        count = len(h.port.submissions)
        events = await h.run(commands.ClearOutputAudio())
        assert types(events) == ["output_audio_buffer.cleared"]
        assert events[0].response_id == response_id
        assert len(h.port.submissions) == count
        assert not h.runner.run.interruption_pending

        await h.run(append_audio())
        assert h.session.active_response_id is None
        await h.run(commands.ClearOutputAudio())
        assert h.runner.run.interruption_pending
        assert h.port.submissions[-1].prompt["model_intermediate_buffer"]["duplex"]["payload"][INTERRUPTION_KEY]
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("second_stage0_turn", [None, 0, 1], ids=["queued", "completed", "awaiting-tts"])
async def test_clear_while_pipelined_audio_is_draining(second_stage0_turn):
    h = await open_harness(plugin=_plugin(False), extra_body={"silence_continuation": False})
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        await h.deliver_and_settle(_speak_segment_end(request_id), stage_id=0, segment_finished=True, model_turn_id=0)
        await h.deliver_and_settle(tts_output(request_id))
        await h.run(append_audio())
        if second_stage0_turn is not None:
            await h.deliver_and_settle(
                _speak_segment_end(request_id), stage_id=0, segment_finished=True, model_turn_id=second_stage0_turn
            )
        events = await h.deliver_and_settle(tts_output(request_id, turn_end=True))
        await h.run(commands.AckPlayback(response_id=find(events, "response.done").response_id, played_ms=1000))
        count = len(h.port.submissions)
        events = await h.run(commands.ClearOutputAudio())
        assert types(events) == ["output_audio_buffer.cleared"]
        needs_control = second_stage0_turn != 0
        assert h.runner.run.interruption_pending is needs_control
        assert len(h.port.submissions) == count + int(needs_control)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_interrupt_does_not_cancel_an_inflight_append():
    h = await open_harness(plugin=_plugin(False), extra_body={"silence_continuation": False})
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        await h.deliver_and_settle(tts_output(request_id))
        h.port.submit_gate = asyncio.Event()
        h.submit(append_audio())
        await asyncio.wait_for(h.port.submit_started.wait(), 2)
        h.submit(commands.CancelResponse())
        await h.settle(timeout_s=0.1)
        assert h.runner.run.interruption_pending
        assert not h.port.aborts and not h.port.cleanups
        h.port.submit_gate.set()
        await h.settle()
        payloads = [
            submission.prompt["model_intermediate_buffer"]["duplex"]["payload"] for submission in h.port.submissions
        ]
        assert len(payloads) == 3
        assert payloads[1]["type"] == "audio"
        assert payloads[2][INTERRUPTION_KEY] is True
        assert not h.port.aborts and not h.port.cleanups
    finally:
        if h.port.submit_gate is not None:
            h.port.submit_gate.set()
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_predecessor", [False, True])
async def test_failed_interruption_submission_allows_the_next_reply(mocker, fail_predecessor):
    h = await open_harness(plugin=_plugin(False), extra_body={"silence_continuation": False})
    release = asyncio.Event()
    started = asyncio.Event()
    try:
        await h.run(append_audio())
        await h.deliver_and_settle(tts_output(h.stage0_request_id()))
        original = h.runner.model.append_runtime_input

        async def reject_once(*args, **kwargs):
            started.set()
            await release.wait()
            raise DuplexInputCapacityError("Capacity exhausted", code="test_capacity")

        mocked = mocker.patch.object(h.runner.model, "append_runtime_input", side_effect=reject_once)
        if fail_predecessor:
            h.submit(append_audio())
            await asyncio.wait_for(started.wait(), 2)
        h.submit(commands.CancelResponse())
        await h.settle(timeout_s=0.1)
        assert h.runner.run.interruption_pending
        h.submit(commands.AppendText(text="Queued behind the rejected control"))
        release.set()
        failure_events = await h.settle()
        mocked.side_effect = original
        assert not h.runner.closing
        assert not h.runner.run.interruption_pending
        assert h.session.epoch == 1
        assert find(failure_events, "input_audio_buffer.cleared").epoch == 1

        await h.run(append_audio())
        events = await h.deliver_and_settle(
            tts_output(h.stage0_request_id(), text="recovered", epoch=h.session.epoch, turn_id=h.session.turn_id)
        )
        assert "response.created" in types(events)
        assert "response.output_audio.delta" in types(events)
        assert find(events, "response.output_audio_transcript.delta").delta == "recovered"
    finally:
        release.set()
        await close_harness(h)


@pytest.mark.asyncio
async def test_cancelled_control_task_does_not_cancel_the_replacement_epoch():
    h = await open_harness(plugin=_plugin(False), extra_body={"silence_continuation": False})
    try:
        await h.run(append_audio())
        await h.deliver_and_settle(tts_output(h.stage0_request_id()))
        h.port.submit_gate = asyncio.Event()
        h.submit(commands.CancelResponse())
        await asyncio.wait_for(h.port.submit_started.wait(), 2)
        h.submit(commands.CancelInput())
        events = await h.settle()
        assert h.session.epoch == 1
        assert not h.runner.run.interruption_pending
        assert types(events).count("input_audio_buffer.cleared") == 1
        h.port.submit_gate.set()
        await h.run(append_audio())
        events = await h.deliver_and_settle(tts_output(h.stage0_request_id(), epoch=h.session.epoch))
        assert "response.output_audio.delta" in types(events)
        assert h.session.epoch == 1
    finally:
        if h.port.submit_gate is not None:
            h.port.submit_gate.set()
        await close_harness(h)


@pytest.mark.asyncio
async def test_cancelled_audio_cannot_consume_next_turn_text():
    h = await open_harness(plugin=_plugin(False), stage_count=3, extra_body={"silence_continuation": False})
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        # Cancellation may precede the first transcript, while codec output
        # from the interrupted turn is still in flight.
        await h.run(commands.BargeIn())
        events = await h.deliver_and_settle(tts_output(request_id, text=""), stage_id=2)
        assert "response.output_audio.delta" not in types(events)
        control = _stage0(request_id, [LISTEN], "", stop=True)
        control.multimodal_output[INTERRUPTION_KEY] = True
        await h.deliver_and_settle(control, stage_id=0, segment_finished=True, model_turn_id=0)
        await h.run(commands.AppendText(text="New question"))
        events = await h.deliver_and_settle(tts_output(request_id, samples=0, text="new answer", turn_id=1), stage_id=2)
        assert "response.output_audio.delta" not in types(events)
        events += await h.deliver_and_settle(
            tts_output(request_id, samples=48000, text="new answer", turn_id=1), stage_id=2
        )
        assert types(events).count("response.output_audio_transcript.delta") == 1
        assert find(events, "response.output_audio_transcript.delta").delta == "new answer"
        assert "response.output_audio.delta" in types(events)
        assert not h.port.aborts and not h.port.cleanups
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("clear_input", [False, True])
async def test_barge_in_retains_partial_input_until_explicit_clear(clear_input):
    h = await open_harness(plugin=_plugin(False), extra_body={"silence_continuation": False})
    try:
        await h.run(append_audio())
        await h.run(append_audio(samples=4000))  # 250 ms, below the native unit size.
        await h.run(commands.BargeIn())
        assert len(h.port.submissions) == 2  # First unit, then interruption control.
        if clear_input:
            await h.run(commands.ClearInput())
        await h.run(append_audio(samples=12000))  # Another 750 ms.
        if clear_input:
            assert len(h.port.submissions) == 2
            await h.run(append_audio(samples=4000))
        assert len(h.port.submissions) == 3
        payload = h.port.submissions[-1].prompt["model_intermediate_buffer"]["duplex"]["payload"]
        assert len(base64.b64decode(payload["audio"])) == 16000 * 4
        assert not h.port.aborts and not h.port.cleanups
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_control_completion_during_next_submit_keeps_resident_request():
    h = await open_harness(plugin=_plugin(False), extra_body={"silence_continuation": False})
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        await h.deliver_and_settle(tts_output(request_id))
        await h.run(commands.CancelResponse())
        h.port.submit_gate = asyncio.Event()
        h.submit(append_audio())
        await asyncio.wait_for(h.port.submit_started.wait(), 2)
        control = _stage0(request_id, [LISTEN], "", stop=True)
        control.multimodal_output[INTERRUPTION_KEY] = True
        h.deliver(control, stage_id=0, segment_finished=True, model_turn_id=0)
        await h.settle(timeout_s=0.1)
        h.port.submit_gate.set()
        await h.settle()
        assert not h.port.aborts and not h.port.cleanups
        assert not h.runner.closing
        assert h.session.input_seq == 3
    finally:
        if h.port.submit_gate is not None:
            h.port.submit_gate.set()
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_followup", [False, True])
async def test_text_queued_during_interruption_keeps_a_cancellable_response(cancel_followup):
    h = await open_harness(plugin=_plugin(False), extra_body={"silence_continuation": False})
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        await h.deliver_and_settle(tts_output(request_id))
        await h.run(commands.CancelResponse())
        events = await h.run(commands.AppendText(text="New question"))
        response_id = find(events, "response.created").response_id
        if cancel_followup:
            cancelled = find(await h.run(commands.CancelResponse()), "response.done")
            assert cancelled.response_id == response_id and cancelled.status == "cancelled"
        control = _stage0(request_id, [LISTEN], "", stop=True)
        control.multimodal_output[INTERRUPTION_KEY] = True
        await h.deliver_and_settle(control, stage_id=0, segment_finished=True, model_turn_id=0)
        events = await h.deliver_and_settle(tts_output(request_id, samples=48000, text="answer", turn_id=1))
        assert "response.created" not in types(events)
        if cancel_followup:
            assert "response.output_audio.delta" not in types(events)
            await h.deliver_and_settle(control, stage_id=0, segment_finished=True, model_turn_id=1)
            assert h.session.active_response_id is None
            assert not h.runner.run.interruption_pending
        else:
            assert find(events, "response.output_audio.delta").response_id == response_id
            assert h.session.active_response_turn_id == 1
        assert not h.port.aborts and not h.port.cleanups
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("old_tail", [[TURN_EOS, CHUNK_EOS], [CHUNK_EOS, LISTEN]])
async def test_control_uses_producer_turn_when_snapshot_includes_cancelled_units(old_tail):
    h = await open_harness(plugin=_plugin(True), stage_count=1, extra_body={"silence_continuation": False})
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        await h.deliver_and_settle(_stage0(request_id, [SPEAK, A], "嗯，", stop=False), stage_id=0)
        await h.run(commands.CancelResponse())
        # The cancelled unit ended naturally before the control. A cumulative
        # snapshot may deliver both boundaries at once; only the control's
        # producer identity owns the transition out of suppression.
        control = _stage0(request_id, [SPEAK, A, *old_tail, LISTEN], "", stop=True)
        control.multimodal_output[INTERRUPTION_KEY] = True
        await h.deliver_and_settle(control, stage_id=0, segment_finished=True, model_turn_id=0)
        assert h.session.turn_id == 1
        assert not h.runner.run.interruption_pending
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("interruptions", [1, 2])
@pytest.mark.parametrize("submit_inflight", [False, True])
async def test_explicit_audio_commit_keeps_response_across_control_completion(interruptions, submit_inflight):
    h = await open_harness(plugin=_plugin(False), extra_body={"silence_continuation": False})
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        await h.deliver_and_settle(tts_output(request_id))
        await h.run(commands.CancelResponse())
        for _ in range(1, interruptions):
            await h.run(commands.BargeIn())
        assert h.runner.run.pending_interruptions == interruptions
        start = len(h.events)
        await h.run(append_audio(samples=3200, is_speech=True))
        if submit_inflight:
            h.port.submit_gate = asyncio.Event()
            h.port.submit_started.clear()
            h.submit(commands.Commit(create_response=True))
            await asyncio.wait_for(h.port.submit_started.wait(), 2)
            events = await h.settle(timeout_s=0.1)
        else:
            events = await h.run(commands.Commit(create_response=True))
        created = find(events, "response.created")
        assert created.response_id == h.session.active_response_id

        for turn in range(interruptions):
            control = _stage0(request_id, [LISTEN], "", stop=True)
            control.multimodal_output[INTERRUPTION_KEY] = True
            events = await h.deliver_and_settle(control, stage_id=0, segment_finished=True, model_turn_id=turn)
            assert "response.done" not in types(events)
        if submit_inflight:
            h.port.submit_gate.set()
            await h.settle()
        final_append = h.port.submissions[-1].prompt["model_intermediate_buffer"]["duplex"]
        assert final_append["final"] is True
        assert final_append["payload"]["_native_explicit_reply"] is True

        events = await h.deliver_and_settle(
            tts_output(request_id, samples=48000, text="new answer", turn_id=interruptions)
        )
        assert find(events, "response.output_audio.delta").response_id == created.response_id
        assert find(events, "response.output_audio_transcript.delta").delta == "new answer"
        assert h.session.active_response_turn_id == interruptions
        await h.deliver_and_settle(
            tts_output(request_id, samples=48000, text="new answer", turn_id=interruptions, turn_end=True)
        )
        followup_events = h.events[start:]
        assert types(followup_events).count("response.created") == 1
        assert types(followup_events).count("response.done") == 1
        done = find(followup_events, "response.done")
        assert done.response_id == created.response_id
        assert done.status == "completed"
        assert h.session.epoch == 0
        assert not h.port.aborts and not h.port.cleanups
    finally:
        if h.port.submit_gate is not None:
            h.port.submit_gate.set()
        await close_harness(h)


@pytest.mark.asyncio
async def test_overlap_barge_in_queues_control_before_user_audio():
    h = await open_harness(plugin=_plugin(False), extra_body={"silence_continuation": False})
    try:
        await h.run(commands.UpdateSession(patch={"overlap_policy": "barge_in_on_speech", "overlap_short_ack_ms": 0}))
        await h.run(append_audio())
        await h.deliver_and_settle(tts_output(h.stage0_request_id()))
        events = await h.run(append_audio(is_speech=True))
        assert find(events, "response.done").status == "cancelled"
        assert h.session.epoch == 0
        assert not h.port.aborts and not h.port.cleanups
        payloads = [
            submission.prompt["model_intermediate_buffer"]["duplex"]["payload"] for submission in h.port.submissions
        ]
        assert [payload["type"] for payload in payloads] == ["audio", "text", "audio"]
        assert payloads[1][INTERRUPTION_KEY] is True
        await h.run(commands.BargeIn())
        assert len(h.port.submissions) == 4
        assert h.runner.run.pending_interruptions == 2
        for turn in (0, 1):
            control = _stage0(h.stage0_request_id(), [LISTEN], "", stop=True)
            control.multimodal_output[INTERRUPTION_KEY] = True
            await h.deliver_and_settle(control, stage_id=0, segment_finished=True, model_turn_id=turn)
            assert h.runner.run.pending_interruptions == 1 - turn
        assert h.session.turn_id == 2
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("thinker_only", [False, True])
async def test_orchestrator_advances_turn_only_after_control_output(thinker_only):
    orchestrator, clients, rpc, _ = _build(stages=1 if thinker_only else 2, plugin=_plugin(thinker_only))
    try:
        await _open(orchestrator, rpc, extra_body={"silence_continuation": False})
        await _submit(orchestrator, _append_audio())
        request_id = _stage0_request_id()
        state = orchestrator.request_states[request_id]
        bridge = state.streaming.bridge_states["duplex"]
        old_tokens = [SPEAK, A, CHUNK_EOS]
        output = _stage0(request_id, old_tokens, "嗯，", stop=True) if thinker_only else tts_output(request_id)
        output.is_segment_finished = True
        await orchestrator._intercept_stage_output(0 if thinker_only else 1, 0, output, state, None, None)
        await _settle(orchestrator)
        await _submit(orchestrator, commands.CancelResponse())
        assert bridge["model_turn_id"] == 0
        assert orchestrator.request_states[request_id] is state
        control = _stage0(request_id, [*old_tokens, LISTEN] if thinker_only else [LISTEN], "", stop=True)
        control.multimodal_output[INTERRUPTION_KEY] = True
        control.is_segment_finished = True
        await orchestrator._intercept_stage_output(0, 0, control, state, None, None)
        assert bridge["model_turn_id"] == 1
        await _settle(orchestrator)
        await _submit(orchestrator, _append_audio())
        assert orchestrator.request_states[request_id] is state
        assert bridge["model_turn_id"] == 1
        assert all(not client.abort_calls for client in clients)
        await _close(orchestrator, rpc)
        assert clients[0].abort_calls == [[request_id]]
    finally:
        await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
async def test_input_clock_keeps_submitted_units_and_control_adds_no_input_duration():
    h = await open_harness(plugin=_plugin(False), stage_count=3, extra_body={"clock": "input"})
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        h.deliver(_speak_segment_end(request_id), stage_id=0, segment_finished=True)
        h.deliver(_stage0(request_id, [], "", stop=False), stage_id=1, segment_finished=True)
        await h.deliver_and_settle(tts_output(request_id), stage_id=2)
        assert "input_audio_buffer.processed" not in types(h.events)
        await h.run(commands.CancelResponse())
        await h.run(append_audio())
        control = _stage0(request_id, [LISTEN], "", stop=True)
        control.multimodal_output[INTERRUPTION_KEY] = True
        await h.deliver_and_settle(control, stage_id=0, segment_finished=True, model_turn_id=0)
        # Even discarded old output completes the work owed to its input.
        await h.deliver_and_settle(tts_output(request_id, tts_is_last_chunk=True), stage_id=2)
        h.deliver(_speak_segment_end(request_id), stage_id=0, segment_finished=True, model_turn_id=1)
        h.deliver(_stage0(request_id, [], "", stop=False), stage_id=1, segment_finished=True)
        await h.deliver_and_settle(tts_output(request_id, samples=48000, turn_id=1, tts_is_last_chunk=True), stage_id=2)
        acks = [event for event in h.events if event.type == "input_audio_buffer.processed"]
        assert [(ack.input_index, ack.units) for ack in acks] == [
            (1, ({"end_ms": 1000, "decision": "speak"},)),
            (2, ({"end_ms": 1000, "decision": "listen"}, {"end_ms": 2000, "decision": "speak"})),
        ]
        assert h.session.epoch == 0
        assert not h.port.aborts and not h.port.cleanups
    finally:
        await close_harness(h)
