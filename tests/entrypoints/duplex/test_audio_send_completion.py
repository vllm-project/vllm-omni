# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Send completion through the real handle, manager and PersonaPlex runner.

The engine command bridge is same-loop, and stage execution and sockets are
recording doubles. No GPU, live serving or acoustic-drain result is asserted.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.engine.duplex.test_session_runner import close_harness
from tests.engine.duplex.test_session_runner_personaplex import code2wav_output, frame, open_personaplex_harness
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.events import AudioDelta
from vllm_omni.engine.duplex.messages import DuplexSessionCommandMessage
from vllm_omni.engine.duplex_omni_engine import DuplexOmniEngine
from vllm_omni.entrypoints.duplex.serving import OmniDuplexSessionHandler
from vllm_omni.entrypoints.duplex_omni import DuplexOmni, DuplexSessionHandle
from vllm_omni.model_executor.models.personaplex.duplex import stage0

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_transport_completion_cannot_be_forged_as_a_client_realtime_command():
    with pytest.raises(commands.DuplexCommandError):
        commands.command_from_realtime({"type": "transport.audio_send_completed", "samples": 100000})


async def _until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), 3)


async def _next_audio(buffer):
    async def read():
        while True:
            event = await buffer.get()
            if isinstance(event, AudioDelta):
                return event

    return await asyncio.wait_for(read(), 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("journal", [False, True])
async def test_older_send_credits_its_samples_not_newer_projected_audio(monkeypatch, mocker, journal):
    monkeypatch.setattr(stage0, "personaplex_prefill_slots", lambda *_: 4)
    h = await open_personaplex_harness()
    # Fix the accepted-input target; wall-clock silence generation is tested
    # separately, not allowed to change this deterministic delivery scenario.
    mocker.patch.object(h.runner.model, "_schedule_silence_continuation", new=mocker.AsyncMock(return_value=False))
    started, release = asyncio.Event(), asyncio.Event()
    sent = []

    async def submit_command_async(session_id, command):
        h.manager.dispatch(DuplexSessionCommandMessage(session_id=session_id, command=command))

    async def send(payload):
        started.set()
        await release.wait()
        sent.append(payload)

    # Keep real frontend/config access; replace only the same-loop transport.
    engine = DuplexOmniEngine.__new__(DuplexOmniEngine)
    engine.duplex_session_config = h.manager.runtime_config
    mocker.patch.object(engine, "submit_command_async", new=submit_command_async)
    omni = DuplexOmni.__new__(DuplexOmni)
    omni.engine = engine
    handle = DuplexSessionHandle(omni, h.session.session_id)
    handle._outbox = h.output_buffer
    handler = OmniDuplexSessionHandler(duplex_omni=omni)
    task = None
    try:
        for _ in range(3):
            await h.run(frame())
        assert h.session.audio_delivery.accepted_seq == 3
        request_id = h.stage0_request_id()
        h.deliver(code2wav_output(request_id, samples=1920, text="he"))
        first = await _next_audio(h.output_buffer)
        h.deliver(code2wav_output(request_id, samples=1920, text="hello"))
        await _until(lambda: h.session.audio_delivery.projected_samples == 3840)
        assert h.session.playback.sent_ms == 160
        assert h.session.audio_delivery.completed_samples == 0

        await handler._attachment_registry.create(h.session.session_id, send=send, close=mocker.AsyncMock())
        if not journal:
            handler._resync_required_sessions.add(h.session.session_id)
        task = asyncio.create_task(handler._send_event(h.session.session_id, first, handle=handle))
        await asyncio.wait_for(started.wait(), 3)
        assert h.session.audio_delivery.completed_samples == 0
        release.set()
        await task
        await _until(lambda: h.session.audio_delivery.completed_samples == 1920)
        # Duplicate receipt for this event cannot credit the newer queued PCM.
        await handle.confirm_output_sent(first)
        await handle.confirm_output_sent(first)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert h.session.audio_delivery.completed_samples == 1920
        assert h.session.playback.played_ms == 0
        assert h.session.playback.sent_ms == 160
        receipt = h.output_buffer.send_receipt(first)

        second = await _next_audio(h.output_buffer)
        await handler._send_event(h.session.session_id, second, handle=handle)
        await _until(lambda: h.session.audio_delivery.completed_samples == 3840)
        assert len(sent) == 2
        assert "_audio_send_watermark" not in str(sent)
        assert "watermark" not in second.to_realtime()

        await h.run(commands.CancelResponse())
        await h.run(frame())
        assert h.session.audio_delivery.epoch == 1
        await h.run(commands.AudioSendCompleted(receipt=receipt))
        assert h.session.audio_delivery.completed_samples == 0
        assert h.session.audio_delivery.accepted_seq == 1
    finally:
        release.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await close_harness(h)


@pytest.mark.asyncio
async def test_failed_submission_cannot_advance_accepted_pcm_sequence(monkeypatch):
    monkeypatch.setattr(stage0, "personaplex_prefill_slots", lambda *_: 4)
    h = await open_personaplex_harness()
    try:
        await h.run(frame(960))
        assert h.session.audio_delivery is None
        h.port.fail_submit = RuntimeError("stage rejected frame")
        await h.run(frame(960))
        assert h.port.submissions == []
        assert h.session.audio_delivery is None
        assert h.runner.model_state.audio_buffer.pending_byte_count == 1920 * 4
        h.port.fail_submit = None
        await h.run(frame(960))
        assert h.session.audio_delivery is not None
        assert h.session.audio_delivery.accepted_seq == 1
        assert h.session.audio_delivery.projected_samples == 0
        assert h.session.audio_delivery.completed_samples == 0
        assert len(h.port.submissions) == 1
        assert h.runner.model_state.audio_buffer.pending_byte_count == 960 * 4
    finally:
        await close_harness(h)
