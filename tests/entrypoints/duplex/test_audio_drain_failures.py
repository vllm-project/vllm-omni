# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drain must not accept blocked, failed, detached or invalidated sends.

Uses the nine-case composition harness: normal production imports and real
runner/codec adapter/serving handler, with same-loop command/close bridges
and synthetic model, decoder and socket execution. No GPU/live IPC claim.
"""

import asyncio

import pytest

from tests.engine.duplex.test_audio_drain import _until
from tests.engine.duplex.test_session_runner import close_harness
from tests.engine.duplex.test_session_runner_personaplex import frame
from tests.entrypoints.duplex.test_audio_drain_accounting import (
    _open_composition,
)
from tests.entrypoints.duplex.test_audio_drain_accounting import (
    build_adapter as _build_adapter_fixture,
)
from vllm_omni.engine.duplex import commands
from vllm_omni.model_executor.models.personaplex.duplex import stage0

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
build_adapter = _build_adapter_fixture


@pytest.fixture(autouse=True)
def _prefill(monkeypatch):
    monkeypatch.setattr(stage0, "personaplex_prefill_slots", lambda *_: 4)


@pytest.mark.asyncio
@pytest.mark.parametrize("journal", [False, True])
@pytest.mark.parametrize("outcome", ["success", "failure", "detached", "invalidated", "detached_during_send"])
async def test_acoustic_drain_requires_current_successful_send(build_adapter, mocker, journal, outcome):
    c = await _open_composition(build_adapter, mocker, attach=False)
    h = c.harnesses[0]
    started, release = asyncio.Event(), asyncio.Event()
    send_task = drain = None

    async def close_session(session_id, *, reason, timeout):
        assert session_id == h.session.session_id
        await h.runner.close(reason)
        h.handle._mark_closed(reason)

    async def send(payload):
        started.set()
        await release.wait()
        if outcome == "failure":
            raise RuntimeError("socket failed")
        h.sent.append(payload)

    h.handle.capabilities = h.session.capabilities
    h.handle._omni.get_session = lambda session_id: h.handle if session_id == h.session.session_id else None
    h.handle._omni.close_session = close_session
    registry = h.handler._attachment_registry
    await registry.create(h.session.session_id, send=send, close=mocker.AsyncMock())
    if not journal:
        h.handler._resync_required_sessions.add(h.session.session_id)
    try:
        for _ in range(2):
            await h.run(frame())
        drain = h.runner.begin_audio_drain(timeout=5)
        await c.generate(h)
        await _until(lambda: c.flushes)
        assert c.submitted_samples(h) == 1920
        event = await c.project(h, 1920)
        state = h.session.audio_delivery
        if outcome == "detached":
            await registry.detach(h.session.session_id)
        send_task = asyncio.create_task(h.handler._send_event(h.session.session_id, event, handle=h.handle))
        if outcome != "detached":
            await asyncio.wait_for(started.wait(), 2)
            assert not drain.done() and state.completed_samples == 0
            if outcome == "invalidated":
                await h.run(commands.CancelResponse())
            elif outcome == "detached_during_send":
                await registry.detach(h.session.session_id)
        release.set()
        await send_task
        if outcome == "success":
            target = await drain
            assert (target.accepted_seq, target.expected_samples, state.completed_samples) == (2, 1920, 1920)
            assert h.session.playback.played_ms == 0
        elif outcome in {"failure", "invalidated"}:
            with pytest.raises(RuntimeError, match="invalidated|closed"):
                await drain
            assert state.completed_samples == 0
        else:
            await asyncio.sleep(0)
            assert not drain.done() and state.completed_samples == 0
            drain.cancel()
            with pytest.raises(asyncio.CancelledError):
                await drain
        assert len(h.sent) == int(outcome not in {"failure", "detached"})
    finally:
        release.set()
        if send_task is not None:
            await asyncio.gather(send_task, return_exceptions=True)
        if drain is not None and not drain.done():
            drain.cancel()
        if drain is not None:
            await asyncio.gather(drain, return_exceptions=True)
        await close_harness(h)
