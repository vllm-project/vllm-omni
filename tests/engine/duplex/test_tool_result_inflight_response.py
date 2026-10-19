# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Accepted results must not finalize a non-resumable request still producing output.

The existing runner harness uses a real MiniCPM plugin with a non-resumable
capability override, as other generic lifecycle tests do. This checks the
shared result-admission boundary, not Qwen parsing or model inference.
"""

from collections.abc import Coroutine
from dataclasses import replace

import pytest

from tests.engine.duplex.test_session_runner import (
    append_audio,
    close_harness,
    open_harness,
    tts_output,
    types,
)
from tests.engine.duplex.test_session_runner_nemotron_voicechat import error_codes, output_item
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.session.tool_calls import ToolCallState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize("auto_response", [False, True])
async def test_fast_tool_result_preserves_inflight_nonresumable_response(auto_response):
    h = await open_harness(auto_response=auto_response)
    try:
        h.session.capabilities = replace(h.session.capabilities, supports_core_resumable_request=False)
        assert error_codes(await h.run(append_audio())) == []
        if not auto_response:
            assert error_codes(await h.run(commands.Commit(create_response=True))) == []
        request_id = h.port.submissions[-1].context.request_id
        await h.deliver_and_settle(
            tts_output(request_id, text="partial", turn_id=h.session.turn_id, epoch=h.session.epoch)
        )
        response_id = h.session.active_response_id
        assert response_id is not None and h.session.active_request_id == request_id
        assert not h.session.response_has_draining_request(response_id)
        stream_id = h.runner.run.stream_request_id
        await h.runner.model._expose_function_call({"call_id": "call_fast", "name": "lookup", "arguments": "{}"})
        await h.settle()
        events = await h.run(commands.CreateItem(item=output_item("call_fast")))
        assert error_codes(events) == []
        assert h.runner.tool_calls.record("call_fast").state is ToolCallState.ACCEPTED
        assert "conversation.item.created" in types(events)
        assert "response.done" not in types(events)
        assert h.session.active_response_id == response_id
        assert h.session.active_request_id == request_id
        assert h.runner.run.stream_request_id == stream_id

        final = await h.deliver_and_settle(
            tts_output(
                request_id,
                text=" final",
                finished=True,
                tts_is_last_chunk=True,
                turn_id=h.session.turn_id,
                epoch=h.session.epoch,
            )
        )
        assert sum(event.type == "response.done" and event.response_id == response_id for event in final) == 1
        if not auto_response:
            assert h.runner.run.stream_request_id is None
        assert error_codes(await h.run(commands.AckPlayback(played_ms=2000, response_id=response_id))) == []
        before = len(h.port.submissions)
        assert error_codes(await h.run(append_audio())) == []
        if not auto_response:
            assert error_codes(await h.run(commands.Commit(create_response=True))) == []
        assert len(h.port.submissions) > before
        assert h.port.submissions[-1].context.request_id != request_id
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_stale_finished_segment_does_not_close_the_next_response():
    """A segment-end continuation must not finish a response opened later.

    The continuation is spawned, so a fast tool result can replace the active
    response before it runs. Response B's stage0 is still producing and must
    not receive that stale ``response.done``.
    """
    h = await open_harness(auto_response=True)
    held: list[Coroutine[object, object, None]] = []
    try:
        h.session.capabilities = replace(h.session.capabilities, supports_core_resumable_request=False)
        assert error_codes(await h.run(append_audio())) == []
        request_a = h.port.submissions[-1].context.request_id
        await h.deliver_and_settle(
            tts_output(request_a, text="partial", turn_id=h.session.turn_id, epoch=h.session.epoch)
        )
        response_a = h.session.active_response_id
        assert response_a is not None and h.session.active_request_id == request_a

        def hold_continuation(coro: Coroutine[object, object, None], *, name: str) -> None:
            if name == "duplex-continue":
                held.append(coro)
                return
            h.runner.__class__.spawn(h.runner, coro, name=name)

        h.runner.spawn = hold_continuation  # type: ignore[method-assign]
        h.deliver(
            tts_output(
                request_a,
                text=" final",
                finished=True,
                tts_is_last_chunk=True,
                turn_id=h.session.turn_id,
                epoch=h.session.epoch,
            )
        )
        await h.settle()
        assert len(held) == 1
        response_b = h.session.begin_response(turn_id=h.session.turn_id)
        h.session.bind_request("request-b")
        assert not h.runner.plugin.data_plane.is_terminal("request-b")
        await held.pop()
        events = await h.settle()
        assert not any(event.type == "response.done" and event.response_id == response_b for event in events)
        assert h.session.active_response_id == response_b
        assert h.session.active_request_id == "request-b"
    finally:
        for coro in held:
            coro.close()
        await close_harness(h)


async def _hold_one_final_segment(h, held: list[Coroutine[object, object, None]]) -> tuple[str, str]:
    """Deliver one final TTS segment and hold its continuation before it runs."""
    h.session.capabilities = replace(h.session.capabilities, supports_core_resumable_request=False)
    assert error_codes(await h.run(append_audio())) == []
    request_a = h.port.submissions[-1].context.request_id
    await h.deliver_and_settle(tts_output(request_a, text="partial", turn_id=h.session.turn_id, epoch=h.session.epoch))
    response_a = h.session.active_response_id
    assert response_a is not None and h.session.active_request_id == request_a

    def hold_continuation(coro: Coroutine[object, object, None], *, name: str) -> None:
        if name == "duplex-continue":
            held.append(coro)
            return
        h.runner.__class__.spawn(h.runner, coro, name=name)

    h.runner.spawn = hold_continuation  # type: ignore[method-assign]
    h.deliver(
        tts_output(
            request_a,
            text=" final",
            finished=True,
            tts_is_last_chunk=True,
            turn_id=h.session.turn_id,
            epoch=h.session.epoch,
        )
    )
    await h.settle()
    assert len(held) == 1
    return request_a, response_a


@pytest.mark.asyncio
async def test_finished_segment_does_not_close_when_active_request_changed():
    """Same response, but the active request is a different one still running."""
    h = await open_harness(auto_response=True)
    held: list[Coroutine[object, object, None]] = []
    try:
        _request_a, response_a = await _hold_one_final_segment(h, held)
        h.session.bind_request("request-b")
        assert not h.runner.plugin.data_plane.is_terminal("request-b")
        await held.pop()
        events = await h.settle()
        assert not any(event.type == "response.done" and event.response_id == response_a for event in events)
        assert h.session.active_response_id == response_a
        assert h.session.active_request_id == "request-b"
    finally:
        for coro in held:
            coro.close()
        await close_harness(h)


@pytest.mark.asyncio
async def test_finished_segment_does_not_close_a_terminal_foreign_request():
    """A finished request is not a reason to close a segment that does not own it."""
    h = await open_harness(auto_response=True)
    held: list[Coroutine[object, object, None]] = []
    try:
        _request_a, response_a = await _hold_one_final_segment(h, held)
        h.session.bind_request("request-b")
        h.runner.plugin.data_plane.mark_terminal("request-b")
        await held.pop()
        events = await h.settle()
        assert not any(event.type == "response.done" and event.response_id == response_a for event in events)
        assert h.session.active_response_id == response_a
        assert h.session.active_request_id == "request-b"
    finally:
        for coro in held:
            coro.close()
        await close_harness(h)


@pytest.mark.asyncio
async def test_finished_segment_still_closes_its_own_request():
    h = await open_harness(auto_response=True)
    held: list[Coroutine[object, object, None]] = []
    try:
        request_a, response_a = await _hold_one_final_segment(h, held)
        assert h.session.active_request_id == request_a
        await held.pop()
        events = await h.settle()
        assert sum(event.type == "response.done" and event.response_id == response_a for event in events) == 1
    finally:
        for coro in held:
            coro.close()
        await close_harness(h)
