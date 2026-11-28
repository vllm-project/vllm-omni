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
    B,
    _stage0,
    _thinker_only_plugin,
)
from tests.engine.test_duplex_orchestrator import (
    SESSION_ID,
    _append_audio,
    _build,
    _close,
    _drain_events,
    _open,
    _settle,
    _stage0_request_id,
    _submit,
)
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.delivery import DuplexOutputBuffer
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


@pytest.fixture
async def h(request):
    """Own the runner and release any blocked submit before teardown."""
    options = getattr(request, "param", {})
    harness = await open_harness(
        plugin=_plugin(options.get("thinker_only", False)),
        stage_count=options.get("stages", 2),
        extra_body={"silence_continuation": False, **options.get("extra_body", {})},
    )
    try:
        yield harness
    finally:
        if harness.port.submit_gate is not None:
            harness.port.submit_gate.set()
        await close_harness(harness)


def _control(request_id, tokens=(LISTEN,)):
    output = _stage0(request_id, list(tokens), "", stop=True)
    output.multimodal_output[INTERRUPTION_KEY] = True
    return output


def _capped_reply(request_id):
    output = _stage0(request_id, [SPEAK, A, B, TURN_EOS], "嗯，好的", stop=True)
    output.outputs[0].finish_reason = "length"
    output.outputs[0].stop_reason = None
    return output


@pytest.mark.asyncio
async def test_interrupt_does_not_cancel_an_inflight_append(h):
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


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_predecessor", [False, True])
async def test_failed_interruption_submission_allows_the_next_reply(h, mocker, fail_predecessor):
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


@pytest.mark.asyncio
async def test_cancelled_control_task_does_not_cancel_the_replacement_epoch(h):
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


@pytest.mark.asyncio
@pytest.mark.parametrize("h", [{"stages": 3}], indirect=True)
async def test_cancelled_audio_cannot_consume_next_turn_text(h):
    await h.run(append_audio())
    request_id = h.stage0_request_id()
    # Cancellation may precede the first transcript, while codec output
    # from the interrupted turn is still in flight.
    await h.run(commands.BargeIn())
    events = await h.deliver_and_settle(tts_output(request_id, text=""), stage_id=2)
    assert "response.output_audio.delta" not in types(events)
    control = _control(request_id)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("h", [{"stages": 3}], indirect=True)
@pytest.mark.parametrize("clear_input", [False, True])
async def test_barge_in_retains_partial_input_until_explicit_clear(h, clear_input):
    h.runner.plugin._tokenizers.clear()  # Audio-only serving can run without an orchestrator tokenizer.
    await h.run(append_audio())
    request_id = h.port.submissions[0].context.request_id
    await h.run(append_audio(samples=4000))  # 250 ms, below the native unit size.
    await h.run(commands.BargeIn())
    assert h.session.epoch == 0
    assert len(h.port.submissions) == 2  # First unit, then interruption control.
    await h.deliver_and_settle(_control(request_id), stage_id=0, segment_finished=True, model_turn_id=0)
    assert h.session.turn_id == 1 and not h.runner.run.interruption_pending
    if clear_input:
        await h.run(commands.ClearInput())
    await h.run(append_audio(samples=12000))  # Another 750 ms.
    if clear_input:
        assert len(h.port.submissions) == 2
        await h.run(append_audio(samples=4000))
    assert len(h.port.submissions) == 3
    assert all(submission.context.request_id == request_id for submission in h.port.submissions)
    payload = h.port.submissions[-1].prompt["model_intermediate_buffer"]["duplex"]["payload"]
    assert len(base64.b64decode(payload["audio"])) == 16000 * 4
    assert not h.port.aborts and not h.port.cleanups


@pytest.mark.asyncio
@pytest.mark.parametrize("old_tail", [[TURN_EOS, CHUNK_EOS], [CHUNK_EOS, LISTEN]])
@pytest.mark.parametrize("h", [{"thinker_only": True, "stages": 1}], indirect=True)
async def test_control_uses_producer_turn_when_snapshot_includes_cancelled_units(h, old_tail):
    await h.run(append_audio())
    request_id = h.stage0_request_id()
    await h.deliver_and_settle(_stage0(request_id, [SPEAK, A], "嗯，", stop=False), stage_id=0)
    await h.run(commands.CancelResponse())
    # The cancelled unit ended naturally before the control. A cumulative
    # snapshot may deliver both boundaries at once; only the control's
    # producer identity owns the transition out of suppression.
    control = _control(request_id, [SPEAK, A, *old_tail, LISTEN])
    await h.deliver_and_settle(control, stage_id=0, segment_finished=True, model_turn_id=0)
    assert h.session.turn_id == 1
    assert not h.runner.run.interruption_pending


@pytest.mark.parametrize("h", [{"thinker_only": True, "stages": 1}, {"stages": 3}], indirect=True)
@pytest.mark.parametrize("control_confirmed", [False, True])
async def test_only_new_thinker_text_response_after_interrupt_gets_a_user_turn(h, control_confirmed):
    await h.run(append_audio())
    request_id = h.stage0_request_id()
    await h.run(commands.BargeIn())
    if control_confirmed:
        await h.deliver_and_settle(_control(request_id), stage_id=0, segment_finished=True, model_turn_id=0)

    for text, forced in (("context", True), ("new question", False), ("continuation", False)):
        await h.run(commands.AppendText(text=text, force_listen=forced))
        payload = h.port.submissions[-1].prompt["model_intermediate_buffer"]["duplex"]["payload"]
        framed = h.runner.plugin.data_plane.thinker_only and text == "new question"
        assert bool(payload.get("text_before_unit")) is framed
        expected = f"\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n" if framed else text
        tokenizer = next(iter(h.runner.plugin._tokenizers.values()))
        assert payload["text_token_ids"] == tokenizer.encode(expected, add_special_tokens=False)
    assert not h.port.aborts and all(s.context.request_id == request_id for s in h.port.submissions)


@pytest.mark.parametrize("early_listen_step", [True, False])
@pytest.mark.parametrize("capped_at_turn_eos", [True, False])
async def test_cancelled_thinker_length_stop_preserves_the_followup(capped_at_turn_eos, early_listen_step):
    orchestrator, clients, rpc, output_q = _build(stages=1, plugin=_plugin(True))
    buffer = DuplexOutputBuffer(max_bytes=2 * 1024 * 1024, max_events=512)
    try:
        await _open(orchestrator, rpc, output_buffer=buffer, extra_body={"silence_continuation": False})
        await _submit(orchestrator, _append_audio())
        request_id = _stage0_request_id()
        state = orchestrator.request_states[request_id]
        session = orchestrator.session_manager.get(SESSION_ID)
        bridge = state.streaming.bridge_states["duplex"]

        async def deliver(output, *, finished=True):
            output.is_segment_finished = finished
            await orchestrator._intercept_stage_output(0, 0, output, state, None, None)
            await _settle(orchestrator)
            return await _drain_events(buffer, output_q)

        await deliver(_stage0(request_id, [SPEAK, A], "嗯，", stop=False), finished=False)
        await _submit(orchestrator, commands.CancelResponse())
        cancelled = find(await _drain_events(buffer, output_q), "response.done")
        assert cancelled.status == "cancelled"
        await _submit(orchestrator, commands.AppendText(text="New question"))
        followup = await _drain_events(buffer, output_q)
        new_id = find(followup, "response.created").response_id
        assert new_id != cancelled.response_id

        capped = _capped_reply(request_id)
        if not capped_at_turn_eos:
            capped.outputs[0].token_ids.pop()
        late = await deliver(capped)
        tokens = [*capped.outputs[0].token_ids, LISTEN]
        control = _control(request_id, tokens)
        control.outputs[0].cumulative_token_ids = tokens
        control.outputs[0].token_ids = [LISTEN]
        # A streaming step can already contain LISTEN before the segment stop.
        if early_listen_step:
            late += await deliver(control, finished=False)
            assert bridge["model_turn_id"] == session.turn_id == 0
        late += await deliver(control)
        assert not {"error", "response.done", "response.output_text.delta"}.intersection(types(late))
        assert bridge["model_turn_id"] == session.turn_id == 1
        assert session.active_response_id == new_id

        reply = _stage0(request_id, [*tokens, SPEAK, B, TURN_EOS, CHUNK_EOS], "好的", stop=True)
        reply.multimodal_output["duplex_text_input"] = True
        followup += await deliver(reply)
        assert types(followup).count("response.created") == types(followup).count("response.done") == 1
        delta = find(followup, "response.output_text.delta")
        done = find(followup, "response.done")
        assert delta.response_id == done.response_id == new_id
        assert delta.delta == "好的" and done.status == "completed"
        assert orchestrator.request_states[request_id] is state
        assert all(not client.abort_calls for client in clients)
    finally:
        await orchestrator.session_manager.shutdown()


@pytest.mark.asyncio
async def test_overlap_barge_in_queues_control_before_user_audio(h):
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
        control = _control(h.stage0_request_id())
        await h.deliver_and_settle(control, stage_id=0, segment_finished=True, model_turn_id=turn)
        assert h.runner.run.pending_interruptions == 1 - turn
    assert h.session.turn_id == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "h", [{"stages": 3, "extra_body": {"clock": "input", "silence_continuation": True}}], indirect=True
)
async def test_input_clock_keeps_submitted_units_and_control_adds_no_input_duration(h):
    await h.run(append_audio())
    request_id = h.stage0_request_id()
    h.deliver(_speak_segment_end(request_id), stage_id=0, segment_finished=True)
    h.deliver(_stage0(request_id, [], "", stop=False), stage_id=1, segment_finished=True)
    await h.deliver_and_settle(tts_output(request_id), stage_id=2)
    assert "input_audio_buffer.processed" not in types(h.events)
    await h.run(commands.CancelResponse())
    await h.run(append_audio())
    control = _control(request_id)
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


@pytest.mark.parametrize(
    "thinker_only,cancel",
    [
        (False, commands.CancelResponse()),
        (False, commands.BargeIn()),
        (False, commands.ClearOutputAudio()),
        (True, commands.CancelResponse()),
    ],
)
async def test_cancel_keeps_request_and_filters_output_through_orchestrator(thinker_only, cancel):
    final_stage = 0 if thinker_only else 1
    orchestrator, clients, rpc, output_q = _build(stages=final_stage + 1, plugin=_plugin(thinker_only))
    buffer = DuplexOutputBuffer(max_bytes=2 * 1024 * 1024, max_events=512)
    try:
        await _open(orchestrator, rpc, output_buffer=buffer, extra_body={"silence_continuation": False})
        await _submit(orchestrator, _append_audio())
        request_id = _stage0_request_id()
        state = orchestrator.request_states[request_id]
        session = orchestrator.session_manager.get(SESSION_ID)
        bridge = state.streaming.bridge_states["duplex"]

        async def deliver(output, stage_id=final_stage):
            output.is_segment_finished = True
            await orchestrator._intercept_stage_output(stage_id, 0, output, state, None, None)
            await _settle(orchestrator)
            return await _drain_events(buffer, output_q)

        old_tokens = [SPEAK, A, CHUNK_EOS]
        await deliver(_stage0(request_id, old_tokens, "嗯，", stop=True) if thinker_only else tts_output(request_id))
        old_response = session.active_response_id
        assert old_response is not None
        await _submit(orchestrator, cancel)
        assert find(await _drain_events(buffer, output_q), "response.done").status == "cancelled"
        assert session.epoch == 0 and session.active_request_id == request_id
        assert bridge["model_turn_id"] == 0  # Enqueuing the control cannot advance the producer.
        submitted = clients[0].add_request_calls[-1][0]
        control = submitted.model_intermediate_buffer["duplex"]
        assert submitted.request_id == request_id and control["seq"] == 2
        assert control["payload"][INTERRUPTION_KEY] and control["payload"]["force_listen"]
        assert control["payload"]["type"] == "text"
        late = await deliver(tts_output(request_id, text="late"))
        assert not {
            "response.created",
            "response.output_audio_transcript.delta",
            "response.output_audio.delta",
        }.intersection(types(late))
        await deliver(_control(request_id, [*old_tokens, LISTEN] if thinker_only else [LISTEN]), stage_id=0)
        assert bridge["model_turn_id"] == session.turn_id == 1
        if not thinker_only:
            assert "response.created" not in types(await deliver(tts_output(request_id, turn_end=True)))
        await _submit(orchestrator, _append_audio())
        assert orchestrator.request_states[request_id] is state
        assert clients[0].add_request_calls[-1][0].model_intermediate_buffer["duplex"]["seq"] == 3
        reply = (
            _stage0(request_id, [*old_tokens, LISTEN, SPEAK, A, CHUNK_EOS], "嗯，", stop=True)
            if thinker_only
            else tts_output(request_id, samples=48000, turn_id=1)
        )
        assert "response.created" in types(await deliver(reply))
        assert session.active_response_id != old_response
        assert all(not client.abort_calls for client in clients)
        await _close(orchestrator, rpc)
        assert clients[0].abort_calls == [[request_id]]
    finally:
        await orchestrator.session_manager.shutdown()


@pytest.mark.parametrize(
    "h", [{"thinker_only": True, "stages": 1}, {"stages": 3}], indirect=True, ids=["text", "cancelled"]
)
async def test_repeated_clear_after_a_drained_reply_without_pcm(h):
    await h.run(append_audio())
    request_id = h.stage0_request_id()
    if h.runner.plugin.data_plane.thinker_only:
        events = await h.deliver_and_settle(
            _stage0(request_id, [SPEAK, A, TURN_EOS, LISTEN], "嗯，", stop=True),
            stage_id=0,
            segment_finished=True,
            model_turn_id=0,
        )
        assert find(events, "response.done").status == "completed"
    else:
        await h.deliver_and_settle(_speak_segment_end(request_id), stage_id=0, segment_finished=True, model_turn_id=0)
        await h.deliver_and_settle(tts_output(request_id), stage_id=2)
        events = await h.run(commands.CancelResponse())
        assert find(events, "response.done").status == "cancelled"
        await h.deliver_and_settle(_control(request_id), stage_id=0, segment_finished=True, model_turn_id=0)

    response_id = find(events, "response.done").response_id
    assert h.session.playback.sent_ms == h.session.playback.generated_ms == 0
    submissions = len(h.port.submissions)
    turn_id = h.session.turn_id
    for _ in range(2):
        cleared = await h.run(commands.ClearOutputAudio())
        assert types(cleared) == ["output_audio_buffer.cleared"] and cleared[0].response_id == response_id
        assert not h.runner.run.interruption_pending
        assert h.session.turn_id == turn_id
        assert len(h.port.submissions) == submissions


@pytest.mark.parametrize("h", [{"thinker_only": True, "stages": 1}], indirect=True)
@pytest.mark.parametrize("next_input", ["none", "submitted", "output_queued"])
async def test_clear_after_failed_thinker_reply(h, next_input):
    await h.run(append_audio())
    request_id = h.stage0_request_id()
    await h.deliver_and_settle(_stage0(request_id, [SPEAK, A], "嗯，", stop=False), stage_id=0, model_turn_id=0)
    if next_input != "none":
        await h.run(append_audio())
    events = await h.deliver_and_settle(_capped_reply(request_id), stage_id=0, segment_finished=True, model_turn_id=0)
    failed = find(events, "response.done")
    assert failed.status == "failed"
    submissions = len(h.port.submissions)
    h.submit(commands.ClearOutputAudio())
    if next_input == "output_queued":
        # Stage 0 has produced the next input's output, but the session
        # mailbox must process the client's clear before delivering it.
        tokens = [SPEAK, A, B, TURN_EOS, SPEAK, B, TURN_EOS, CHUNK_EOS]
        h.deliver(_stage0(request_id, tokens, "好的", stop=True), stage_id=0, segment_finished=True, model_turn_id=0)
    events = await h.settle()
    assert types(events) == ["output_audio_buffer.cleared"] and events[0].response_id == failed.response_id
    assert h.runner.run.interruption_pending is (next_input != "none")
    assert len(h.port.submissions) == submissions + int(next_input != "none")
    if next_input == "none":
        events = await h.run(commands.ClearOutputAudio())
        assert types(events) == ["output_audio_buffer.cleared"] and events[0].response_id == failed.response_id
        assert len(h.port.submissions) == submissions


@pytest.mark.parametrize("h", [{"thinker_only": True, "stages": 1}, {"stages": 3}], indirect=True)
@pytest.mark.parametrize("after_reply", [False, True])
async def test_clear_after_passive_listen(h, after_reply):
    request_id = h.stage0_request_id()
    tokens = []
    response_id = None
    if after_reply:
        await h.run(append_audio())
        tokens = [SPEAK, A, TURN_EOS, CHUNK_EOS]
        events = await h.deliver_and_settle(
            _stage0(request_id, tokens, "嗯，", stop=True), stage_id=0, segment_finished=True, model_turn_id=0
        )
        if not h.runner.plugin.data_plane.thinker_only:
            events = await h.deliver_and_settle(tts_output(request_id, turn_end=True), stage_id=2)
        response_id = find(events, "response.done").response_id
        await h.run(commands.AckPlayback(response_id=response_id, played_ms=1000))

    await h.run(append_audio())
    turn_id = h.session.turn_id
    await h.deliver_and_settle(
        _stage0(request_id, [*tokens, LISTEN], "", stop=True),
        stage_id=0,
        segment_finished=True,
        model_turn_id=turn_id,
    )
    submissions = len(h.port.submissions)
    for _ in range(2):
        cleared = await h.run(commands.ClearOutputAudio())
        assert types(cleared) == ["output_audio_buffer.cleared"] and cleared[0].response_id == response_id
        assert h.session.turn_id == turn_id and not h.runner.run.interruption_pending
        assert len(h.port.submissions) == submissions


@pytest.mark.parametrize("h", [{"stages": 3}], indirect=True)
async def test_clear_precedes_a_queued_passive_listen(h):
    await h.run(append_audio())
    count = len(h.port.submissions)
    h.submit(commands.ClearOutputAudio())
    h.deliver(_stage0(h.stage0_request_id(), [LISTEN], "", stop=True), stage_id=0, segment_finished=True)
    assert types(await h.settle()) == ["output_audio_buffer.cleared"]
    assert h.runner.run.interruption_pending and len(h.port.submissions) == count + 1


@pytest.mark.parametrize("h", [{"stages": 3}], indirect=True)
async def test_passive_listen_does_not_hide_pending_tts(h):
    await h.run(append_audio())
    request_id = h.stage0_request_id()
    await h.deliver_and_settle(_speak_segment_end(request_id), stage_id=0, segment_finished=True, model_turn_id=0)
    await h.run(append_audio())
    await h.deliver_and_settle(
        _stage0(request_id, [LISTEN], "", stop=True), stage_id=0, segment_finished=True, model_turn_id=0
    )
    count = len(h.port.submissions)
    assert types(await h.run(commands.ClearOutputAudio())) == ["output_audio_buffer.cleared"]
    assert h.runner.run.interruption_pending and len(h.port.submissions) == count + 1


@pytest.mark.parametrize("h", [{"stages": 3}], indirect=True)
@pytest.mark.parametrize("last_chunk", [True, False])
async def test_retired_turn_releases_audio_and_terminal_text_flushes_once(h, last_chunk):
    request_id = h.stage0_request_id()
    samples = 0
    for turn_id in range(4):
        await h.run(append_audio())
        samples += 24000
        events = await h.deliver_and_settle(
            tts_output(request_id, samples=samples, text="", turn_id=turn_id), stage_id=2
        )
        assert "response.output_audio.delta" not in types(events)
        turn_state = h.runner.plugin.data_plane._requests[request_id].turns[turn_id]
        assert turn_state.pending_audio_without_text

        text = "answer" if turn_id % 2 else ""
        # A turn boundary may also bring the last piece of the waveform.
        if turn_id >= 2:
            samples += 12000
        terminal = tts_output(
            request_id, samples=samples, text=text, turn_end=True, turn_id=turn_id, tts_is_last_chunk=last_chunk
        )
        events = await h.deliver_and_settle(terminal, stage_id=2)
        assert h.session.turn_id == turn_id + 1
        assert not turn_state.pending_audio_without_text
        if text:
            done = find(events, "response.done")
            assert done.status == "completed"
            transcript = [event for event in events if event.type == "response.output_audio_transcript.delta"]
            assert "".join(event.delta for event in transcript) == text
            assert all(event.response_id == done.response_id for event in transcript)
            packets = [event for event in events if event.type == "response.output_audio.delta"]
            assert packets and all(packet.response_id == done.response_id for packet in packets)
            assert [packet.delta for packet in packets] == (
                ["wav-24000", "wav-12000"] if turn_id >= 2 else ["wav-24000"]
            )
            await h.run(commands.AckPlayback(response_id=done.response_id, played_ms=1500 if turn_id >= 2 else 1000))
        else:
            assert "response.created" not in types(events) and "response.output_audio.delta" not in types(events)
        replay = await h.deliver_and_settle(terminal, stage_id=2)
        assert not {"response.output_audio.delta", "response.done"}.intersection(types(replay))


@pytest.mark.parametrize("pending", ["none", "queued", "completed", "awaiting-tts"])
async def test_clear_only_interrupts_undrained_input(h, pending):
    await h.run(append_audio())
    request_id = h.stage0_request_id()
    if pending != "none":
        await h.deliver_and_settle(_speak_segment_end(request_id), stage_id=0, segment_finished=True, model_turn_id=0)
        await h.deliver_and_settle(tts_output(request_id))
        await h.run(append_audio())
        if pending != "queued":
            await h.deliver_and_settle(
                _speak_segment_end(request_id),
                stage_id=0,
                segment_finished=True,
                model_turn_id=int(pending == "awaiting-tts"),
            )
    events = await h.deliver_and_settle(tts_output(request_id, turn_end=True))
    response_id = find(events, "response.done").response_id
    await h.run(commands.AckPlayback(response_id=response_id, played_ms=1000))
    count = len(h.port.submissions)
    events = await h.run(commands.ClearOutputAudio())
    assert types(events) == ["output_audio_buffer.cleared"] and events[0].response_id == response_id
    needs_control = pending in {"queued", "awaiting-tts"}
    assert h.runner.run.interruption_pending is needs_control
    assert len(h.port.submissions) == count + int(needs_control)
    if pending == "none":
        await h.run(append_audio())
        assert h.session.active_response_id is None
        await h.run(commands.ClearOutputAudio())
        assert h.runner.run.interruption_pending
        assert h.port.submissions[-1].prompt["model_intermediate_buffer"]["duplex"]["payload"][INTERRUPTION_KEY]


@pytest.mark.parametrize(
    "input_kind,interruptions,submit_inflight",
    [
        ("append", 1, True),
        ("text", 1, False),
        ("commit", 1, False),
        ("commit", 1, True),
        ("commit", 2, False),
        ("commit", 2, True),
    ],
)
async def test_followup_keeps_response_across_control_completion(h, input_kind, interruptions, submit_inflight):
    await h.run(append_audio())
    request_id = h.stage0_request_id()
    await h.deliver_and_settle(tts_output(request_id))
    await h.run(commands.CancelResponse())
    for _ in range(1, interruptions):
        await h.run(commands.BargeIn())
    assert h.runner.run.pending_interruptions == interruptions
    start = len(h.events)
    if input_kind == "commit":
        await h.run(append_audio(samples=3200, is_speech=True))
        command = commands.Commit(create_response=True)
    else:
        command = commands.AppendText(text="New question") if input_kind == "text" else append_audio()
    if submit_inflight:
        h.port.submit_gate = asyncio.Event()
        h.port.submit_started.clear()
        h.submit(command)
        await asyncio.wait_for(h.port.submit_started.wait(), 2)
        await h.settle(timeout_s=0.1)
    else:
        await h.run(command)
    precreated = h.session.active_response_id
    assert (precreated is not None) is (input_kind != "append")
    if precreated is not None:
        assert find(h.events[start:], "response.created").response_id == precreated
    for turn in range(interruptions):
        events = await h.deliver_and_settle(_control(request_id), stage_id=0, segment_finished=True, model_turn_id=turn)
        assert "response.done" not in types(events)
    if submit_inflight:
        h.port.submit_gate.set()
        await h.settle()
    assert h.session.input_seq == interruptions + 2 and not h.runner.closing
    if input_kind == "commit":
        final = h.port.submissions[-1].prompt["model_intermediate_buffer"]["duplex"]
        assert final["final"] is True and final["payload"]["_native_explicit_reply"] is True
    events = await h.deliver_and_settle(tts_output(request_id, samples=48000, text="new answer", turn_id=interruptions))
    response_id = find(events, "response.output_audio.delta").response_id
    assert precreated is None or response_id == precreated
    assert find(events, "response.output_audio_transcript.delta").delta == "new answer"
    assert h.session.active_response_turn_id == interruptions
    await h.deliver_and_settle(
        tts_output(request_id, samples=48000, text="new answer", turn_id=interruptions, turn_end=True)
    )
    followup = h.events[start:]
    assert types(followup).count("response.created") == types(followup).count("response.done") == 1
    done = find(followup, "response.done")
    assert done.response_id == response_id and done.status == "completed"
    assert h.session.epoch == 0 and not h.port.aborts and not h.port.cleanups


async def test_text_queued_during_interruption_can_be_cancelled(h):
    await h.run(append_audio())
    request_id = h.stage0_request_id()
    await h.deliver_and_settle(tts_output(request_id))
    await h.run(commands.CancelResponse())
    response_id = find(await h.run(commands.AppendText(text="New question")), "response.created").response_id
    cancelled = find(await h.run(commands.CancelResponse()), "response.done")
    assert cancelled.response_id == response_id and cancelled.status == "cancelled"
    await h.deliver_and_settle(_control(request_id), stage_id=0, segment_finished=True, model_turn_id=0)
    events = await h.deliver_and_settle(tts_output(request_id, samples=48000, text="answer", turn_id=1))
    assert not {"response.created", "response.output_audio.delta"}.intersection(types(events))
    await h.deliver_and_settle(_control(request_id), stage_id=0, segment_finished=True, model_turn_id=1)
    assert h.session.active_response_id is None and not h.runner.run.interruption_pending
    assert not h.port.aborts and not h.port.cleanups
