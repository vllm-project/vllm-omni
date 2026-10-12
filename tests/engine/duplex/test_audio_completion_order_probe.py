# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression coverage for ordered generation and event-specific send receipts.

Real runner, manager, plugin and handle; stage outputs and transport are
deterministic doubles. These are not GPU or live-model evidence.
"""

import asyncio

import pytest
import torch

from tests.engine.duplex.test_session_runner import close_harness
from tests.engine.duplex.test_session_runner_personaplex import code2wav_output, frame, open_personaplex_harness
from vllm_omni.engine.duplex.events import AudioDelta
from vllm_omni.engine.duplex.messages import DuplexSessionCommandMessage
from vllm_omni.engine.duplex_omni_engine import DuplexOmniEngine
from vllm_omni.entrypoints.duplex.serving import OmniDuplexSessionHandler
from vllm_omni.entrypoints.duplex_omni import DuplexOmni, DuplexSessionHandle
from vllm_omni.model_executor.models.personaplex.duplex import stage0
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


async def _until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), 3)


def _generated(h, sequence):
    return h.deliver(
        OmniRequestOutput(request_id=h.stage0_request_id(), finished=False),
        stage_id=0,
        segment_finished=True,
        segment_token_ids=(11,),
        segment_output_metadata={
            "meta.duplex_epoch": torch.tensor([h.session.epoch]),
            "meta.duplex_generated_seq": torch.tensor([sequence]),
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("second_submission_fails", [False, True])
async def test_fast_second_generation_does_not_overtake_deferred_first_receipt(
    monkeypatch, mocker, second_submission_fails
):
    monkeypatch.setattr(stage0, "personaplex_prefill_slots", lambda *_: 4)
    h = await open_personaplex_harness()
    mocker.patch.object(h.runner.model, "_schedule_silence_continuation", new=mocker.AsyncMock(return_value=False))
    started, release = asyncio.Event(), asyncio.Event()
    original_submit = h.port.submit
    credited = mocker.spy(h.session, "complete_audio_generation")

    async def submit(submission):
        sequence = submission.prompt["model_intermediate_buffer"]["duplex"]["seq"]
        assert not _generated(h, sequence)
        if sequence == 1:
            started.set()
            await release.wait()
        elif second_submission_fails:
            raise RuntimeError("second submission failed after generation notification")
        return await original_submit(submission)

    mocker.patch.object(h.port, "submit", new=submit)
    try:
        h.submit(frame())
        h.submit(frame())
        await asyncio.wait_for(started.wait(), 3)
        await _until(lambda: any(task.get_name() == "duplex-generation-receipt" for task in h.runner._background_tasks))
        release.set()
        await h.settle()
        state = h.session.audio_delivery
        expected = 1 if second_submission_fails else 2
        assert state is not None and state.accepted_seq == expected
        order = [call.kwargs["sequence"] for call in credited.call_args_list]
        print({"accepted": state.accepted_seq, "generated": state.generated_seq, "receipt_credit_order": order})
        assert state.generated_seq == expected, order
    finally:
        release.set()
        await close_harness(h)


@pytest.mark.asyncio
async def test_earlier_committed_generation_does_not_wait_for_later_pending_append(monkeypatch, mocker):
    monkeypatch.setattr(stage0, "personaplex_prefill_slots", lambda *_: 4)
    h = await open_personaplex_harness()
    mocker.patch.object(h.runner.model, "_schedule_silence_continuation", new=mocker.AsyncMock(return_value=False))
    first_started, release_first = asyncio.Event(), asyncio.Event()
    later_started, release_later = asyncio.Event(), asyncio.Event()
    original_submit = h.port.submit

    async def submit(submission):
        sequence = submission.prompt["model_intermediate_buffer"]["duplex"]["seq"]
        if sequence == 1:
            assert not _generated(h, sequence)
            first_started.set()
            await release_first.wait()
        else:
            later_started.set()
            await release_later.wait()
        return await original_submit(submission)

    mocker.patch.object(h.port, "submit", new=submit)
    try:
        h.submit(frame())
        h.submit(frame())
        await asyncio.wait_for(first_started.wait(), 3)
        await _until(lambda: any(task.get_name() == "duplex-generation-receipt" for task in h.runner._background_tasks))
        release_first.set()
        await asyncio.wait_for(later_started.wait(), 3)
        assert h.session.audio_delivery.accepted_seq == 1
        try:
            await asyncio.wait_for(_until(lambda: h.session.audio_delivery.generated_seq == 1), 0.5)
        except TimeoutError:
            pass
        state = h.session.audio_delivery
        print(
            {
                "accepted": state.accepted_seq,
                "generated": state.generated_seq,
                "later_append_still_pending": not release_later.is_set(),
            }
        )
        assert state.generated_seq == 1, "A later unaccepted append cannot hold the earlier generation receipt."
    finally:
        release_first.set()
        release_later.set()
        await h.settle()
        await close_harness(h)


@pytest.mark.asyncio
async def test_dequeued_unsent_first_delta_cannot_be_credited_by_later_send(monkeypatch, mocker):
    monkeypatch.setattr(stage0, "personaplex_prefill_slots", lambda *_: 4)
    h = await open_personaplex_harness()
    mocker.patch.object(h.runner.model, "_schedule_silence_continuation", new=mocker.AsyncMock(return_value=False))
    mocker.patch.object(h.port, "flush_audio_prefix", new=mocker.AsyncMock(return_value=3))

    async def submit_command_async(session_id, command):
        h.manager.dispatch(DuplexSessionCommandMessage(session_id=session_id, command=command))

    engine = DuplexOmniEngine.__new__(DuplexOmniEngine)
    engine.duplex_session_config = h.manager.runtime_config
    mocker.patch.object(engine, "submit_command_async", new=submit_command_async)
    omni = DuplexOmni.__new__(DuplexOmni)
    omni.engine = engine
    handle = DuplexSessionHandle(omni, h.session.session_id)
    handle._outbox = h.output_buffer
    handler = OmniDuplexSessionHandler(duplex_omni=omni)
    sends = []

    async def send(payload):
        sends.append(payload)

    drain = None
    consumer = handle.events()
    try:
        for _ in range(3):
            await h.run(frame())
        for sequence in (1, 2, 3):
            _generated(h, sequence)
        await _until(lambda: h.session.audio_delivery.generated_seq == 3)
        await handler._attachment_registry.create(h.session.session_id, send=send, close=mocker.AsyncMock())
        drain = h.runner.begin_audio_drain(timeout=2)
        request_id = h.stage0_request_id()
        h.deliver(code2wav_output(request_id, samples=1920, text="one"))
        h.deliver(code2wav_output(request_id, samples=1920, text="one two"))
        await _until(lambda: h.session.audio_delivery.projected_samples == 3840)

        async def next_audio():
            while True:
                event = await anext(consumer)
                if isinstance(event, AudioDelta):
                    return event

        first = await asyncio.wait_for(next_audio(), 1)
        # A server-side output consumer is permitted to dequeue without send;
        # the new API promises that this does not satisfy acoustic drain.
        assert h.output_buffer.send_receipt(first).watermark.samples == 1920
        second = await asyncio.wait_for(next_audio(), 1)
        assert h.output_buffer.send_receipt(first) is None
        assert h.output_buffer.send_receipt(second).watermark.samples == 3840
        await handler._send_event(h.session.session_id, second, handle=handle)
        await _until(lambda: h.runner._mailbox.empty())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        state = h.session.audio_delivery
        print(
            {
                "socket_sends": len(sends),
                "actually_sent_samples": 1920,
                "projected_samples": state.projected_samples,
                "completed_samples": state.completed_samples,
                "drain_done": drain.done(),
            }
        )
        assert len(sends) == 1
        assert state.completed_samples <= 1920
        assert not drain.done()
    finally:
        await consumer.aclose()
        if drain is not None:
            if drain.done() and not drain.cancelled():
                drain.exception()
            else:
                drain.cancel()
        await close_harness(h)
