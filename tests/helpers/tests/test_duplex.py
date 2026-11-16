# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Behavior and bounded-wait contracts for shared duplex test helpers."""

import asyncio

import pytest

from tests.helpers.duplex import next_audio, wait_until
from vllm_omni.engine.duplex.delivery import DuplexOutputBuffer
from vllm_omni.engine.duplex.events import AudioDelta, TextDelta

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
async def test_wait_until_accepts_an_already_satisfied_condition():
    await wait_until(lambda: True)


@pytest.mark.asyncio
async def test_wait_until_yields_to_the_condition_producer():
    ready = asyncio.Event()
    asyncio.get_running_loop().call_soon(ready.set)
    await wait_until(ready.is_set)


@pytest.mark.asyncio
async def test_wait_until_times_out_on_an_unsatisfied_condition():
    with pytest.raises(TimeoutError):
        await wait_until(lambda: False, timeout=0.01)


@pytest.mark.asyncio
async def test_wait_until_preserves_predicate_errors():
    def failed() -> bool:
        raise ValueError("predicate failed")

    with pytest.raises(ValueError, match="predicate failed"):
        await wait_until(failed)


@pytest.mark.asyncio
async def test_next_audio_skips_text_and_preserves_event_identity():
    buffer = DuplexOutputBuffer(max_bytes=4096, max_events=4)
    audio = AudioDelta(response_id="r", delta="AAAA")
    buffer.put(TextDelta(response_id="r", delta="text"))
    buffer.put(audio)
    assert await next_audio(buffer) is audio
    assert buffer.pending_events == 0


@pytest.mark.asyncio
async def test_next_audio_times_out_and_leaves_buffer_usable():
    buffer = DuplexOutputBuffer(max_bytes=4096, max_events=4)
    buffer.put(TextDelta(response_id="r", delta="text"))
    with pytest.raises(TimeoutError):
        await next_audio(buffer, timeout=0.01)
    audio = AudioDelta(response_id="r", delta="AAAA")
    buffer.put(audio)
    assert await next_audio(buffer) is audio


@pytest.mark.asyncio
async def test_next_audio_rejects_a_closed_buffer():
    buffer = DuplexOutputBuffer(max_bytes=4096, max_events=4)
    buffer.close()
    with pytest.raises(AssertionError, match="closed before an audio event"):
        await next_audio(buffer)


@pytest.mark.asyncio
async def test_next_audio_preserves_caller_cancellation():
    buffer = DuplexOutputBuffer(max_bytes=4096, max_events=4)
    waiting = asyncio.create_task(next_audio(buffer))
    await asyncio.sleep(0)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    audio = AudioDelta(response_id="r", delta="AAAA")
    buffer.put(audio)
    assert await next_audio(buffer) is audio
