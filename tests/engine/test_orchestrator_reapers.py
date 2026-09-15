# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from vllm_omni.engine import orchestrator as orchestrator_module
from vllm_omni.engine.orchestrator import Orchestrator

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("reaper", ["duplex", "cfg_companion"])
@pytest.mark.parametrize("legacy_timeout", [False, True])
async def test_reaper_continues_after_poll_timeouts(monkeypatch, reaper, legacy_timeout):
    orch = Orchestrator.__new__(Orchestrator)
    orch._shutdown_event = asyncio.Event()
    setattr(orch, f"_{reaper}_reaper_interval_s", 0.001)
    calls = 0

    async def reap():
        nonlocal calls
        calls += 1
        if calls == 2:
            orch._shutdown_event.set()

    monkeypatch.setattr(orch, "duplex_control_plane", SimpleNamespace(reap_expired=reap), raising=False)
    monkeypatch.setattr(orch, "_reap_expired_cfg_parents", reap)
    if legacy_timeout:
        # Python 3.10's asyncio timeout is distinct from the builtin. Exercise
        # that exception identity even when the suite runs on newer Python.
        class LegacyAsyncioTimeoutError(Exception):
            pass

        async def wait_for(awaitable, *, timeout):
            awaitable.close()
            raise LegacyAsyncioTimeoutError

        monkeypatch.setattr(
            orchestrator_module,
            "asyncio",
            SimpleNamespace(wait_for=wait_for, TimeoutError=LegacyAsyncioTimeoutError),
        )

    await asyncio.wait_for(getattr(orch, f"_{reaper}_reaper_loop")(), timeout=1)
    assert calls == 2


@pytest.mark.parametrize("reaper", ["duplex", "cfg_companion"])
@pytest.mark.parametrize("already_stopped", [False, True])
async def test_reaper_shutdown_does_not_reap(monkeypatch, reaper, already_stopped):
    orch = Orchestrator.__new__(Orchestrator)
    orch._shutdown_event = asyncio.Event()
    setattr(orch, f"_{reaper}_reaper_interval_s", 60)
    reap = AsyncMock()
    monkeypatch.setattr(orch, "duplex_control_plane", SimpleNamespace(reap_expired=reap), raising=False)
    monkeypatch.setattr(orch, "_reap_expired_cfg_parents", reap)
    if already_stopped:
        orch._shutdown_event.set()
    else:
        asyncio.get_running_loop().call_soon(orch._shutdown_event.set)

    await asyncio.wait_for(getattr(orch, f"_{reaper}_reaper_loop")(), timeout=1)
    reap.assert_not_awaited()


async def test_duplex_reaper_retries_cleanup_failure(monkeypatch, caplog):
    orch = Orchestrator.__new__(Orchestrator)
    orch._shutdown_event = asyncio.Event()
    orch._duplex_reaper_interval_s = 0.001
    calls = 0

    async def reap():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary cleanup failure")
        orch._shutdown_event.set()

    monkeypatch.setattr(orch, "duplex_control_plane", SimpleNamespace(reap_expired=reap), raising=False)
    await asyncio.wait_for(orch._duplex_reaper_loop(), timeout=1)
    assert calls == 2
    assert "Duplex expiry cleanup failed; retrying on next tick" in caplog.text
