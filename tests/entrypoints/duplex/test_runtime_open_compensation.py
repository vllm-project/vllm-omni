# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio

import pytest

from tests.entrypoints.openai_api.test_duplex_handler import (
    FakeChatService,
    FakeEngineClient,
    OmniDuplexSessionHandler,
    TimedWebSocket,
    _native_session_create,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["timeout", "cancel"])
async def test_accepted_open_without_reply_is_compensated(outcome):
    accepted = asyncio.Event()
    release_reply = asyncio.Event()
    closed = asyncio.Event()

    class LostOpenReplyEngine(FakeEngineClient):
        async def open_duplex_session_async(self, session_id, **kwargs):
            result = await super().open_duplex_session_async(session_id, **kwargs)
            accepted.set()
            if outcome == "timeout":
                raise TimeoutError("open accepted, reply not received")
            await release_reply.wait()
            return result

        async def close_duplex_session_async(self, session_id, **kwargs):
            result = await super().close_duplex_session_async(session_id, **kwargs)
            closed.set()
            return result

    engine = LostOpenReplyEngine()
    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(engine), config_timeout_s=0.1, idle_timeout_s=1)
    websocket = TimedWebSocket()
    websocket.put(_native_session_create("abandoned-open"))
    task = asyncio.create_task(handler.handle_session(websocket))
    await asyncio.wait_for(accepted.wait(), timeout=2)
    if outcome == "cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release_reply.set()
        await asyncio.wait_for(closed.wait(), 2)
    else:
        await task
    assert engine.opened == ["abandoned-open"]
    assert handler._registry.get("abandoned-open") is None
    assert engine.closed == [("abandoned-open", "open_abandoned")]
    assert handler._runtime_open_attempts.pending_count == 0
