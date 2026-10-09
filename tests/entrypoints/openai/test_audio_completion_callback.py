# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Completion is after a valid socket send, never journal acceptance."""

import asyncio

import pytest

from vllm_omni.engine.duplex.delivery import AudioSampleWatermark, DuplexOutputBuffer
from vllm_omni.engine.duplex.events import AudioDelta
from vllm_omni.entrypoints.duplex.session_attachment import DuplexSessionAttachmentRegistry

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize("journal", [False, True])
@pytest.mark.parametrize("outcome", ["success", "failure", "detached", "invalidated", "detached_during_send"])
async def test_only_current_successful_send_reports_completion(mocker, journal, outcome):
    registry = DuplexSessionAttachmentRegistry(replay_ttl_s=60, replay_max_bytes_per_session=4096)
    output = DuplexOutputBuffer(max_bytes=4096, max_events=8)
    started, release = asyncio.Event(), asyncio.Event()

    async def send(_payload):
        started.set()
        await release.wait()
        if outcome == "failure":
            raise RuntimeError("socket failed")

    await registry.create("s", send=send, close=mocker.AsyncMock())
    event = AudioDelta(session_id="s", response_id="r", epoch=0, delta="AAAA")
    watermark = AudioSampleWatermark(request_id="request", epoch=0, samples=1920)
    output.put(event, audio_watermark=watermark)
    assert await output.get() is event
    receipt = output.send_receipt(event)
    assert receipt.event_id == event.event_id and receipt.watermark == watermark
    completed = mocker.AsyncMock()
    if outcome == "detached":
        await registry.detach("s")
    delivery = asyncio.create_task(
        registry.send_event(
            "s", event.to_realtime(), journal=journal, event_guard=lambda: output.guard(event), on_sent=completed
        )
    )
    try:
        if outcome != "detached":
            await asyncio.wait_for(started.wait(), 3)
            completed.assert_not_awaited()
            if outcome == "invalidated":
                output.invalidate("r", through_epoch=0)
            if outcome == "detached_during_send":
                await registry.detach("s")
        release.set()
        if outcome == "failure":
            with pytest.raises(RuntimeError, match="socket failed"):
                await delivery
        else:
            await delivery
        assert completed.await_count == int(outcome == "success")
        if outcome == "invalidated":
            assert output.send_receipt(event) is None
    finally:
        release.set()
        await asyncio.gather(delivery, return_exceptions=True)
