# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-session lease teardown: a cancelled engine call must be drained.

Cancelling an asyncio task blocked on a worker-thread engine call returns
immediately while the thread keeps running (executor futures cannot be
interrupted). The server therefore tracks the in-flight future on the session
state and waits it out before releasing the capacity-one lease; otherwise a
fast reconnect could call ``open()`` on the shared engine while the previous
connection's final ``step()`` is still mutating it.
"""

import asyncio
import contextlib
import sys
import threading
import types

import pytest

try:
    import sphn  # noqa: F401
except ImportError:
    # These lease tests never encode/decode Opus. Keep the optional serving
    # dependency from hiding dependency-free teardown regressions.
    sys.modules["sphn"] = types.ModuleType("sphn")

from vllm_omni.experimental.fullduplex.personaplex.config import PersonaPlexConfig
from vllm_omni.experimental.fullduplex.personaplex.serving.server import DuplexServer
from vllm_omni.experimental.fullduplex.personaplex.session import PersonaPlexServingSessionState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
async def test_cancelled_engine_call_is_drained_before_lease_release():
    server = DuplexServer(PersonaPlexConfig())  # engine never loaded: stub calls only
    state = PersonaPlexServingSessionState()

    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def slow_engine_op():
        started.set()
        release.wait(timeout=5.0)
        finished.set()
        return []

    task = asyncio.create_task(server._engine_call(state, slow_engine_op))
    await asyncio.to_thread(started.wait, 5.0)

    # Cancel while the worker thread is mid-call: the await returns at once,
    # the thread keeps running, and its completion event stays tracked on the
    # state (the asyncio future itself reads as cancelled/done here, which is
    # exactly why the drain must not rely on it).
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert state.inflight is not None
    assert not state.inflight.is_set()

    # The drain must genuinely wait for the worker, not just return.
    drain = asyncio.create_task(server._drain_inflight(state))
    await asyncio.sleep(0.05)
    assert not drain.done()

    release.set()
    await asyncio.wait_for(drain, timeout=5.0)
    assert finished.is_set()
    assert state.inflight.is_set()


@pytest.mark.asyncio
async def test_completed_engine_call_clears_inflight():
    server = DuplexServer(PersonaPlexConfig())
    state = PersonaPlexServingSessionState()
    result = await server._engine_call(state, lambda: "ok")
    assert result == "ok"
    assert state.inflight is None
    await server._drain_inflight(state)  # no-op on a clean state


@pytest.mark.asyncio
async def test_drain_timeout_keeps_lease_until_worker_finishes():
    server = DuplexServer(PersonaPlexConfig())
    server._active = True
    state = PersonaPlexServingSessionState()
    state.inflight = threading.Event()

    drained = await server._release_lease_after_drain(state, timeout=0.01)

    assert drained is False
    assert server._active is True

    state.inflight.set()
    for _ in range(100):
        if not server._active:
            break
        await asyncio.sleep(0.01)
    assert server._active is False
