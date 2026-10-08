# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import concurrent.futures

import janus
import pytest

from vllm_omni.engine import orchestrator as orchestrator_module
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.orchestrator import Orchestrator
from vllm_omni.engine.stage_runtime import StageRuntime

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("legacy_timeout", [False, True])
async def test_reaper_continues_after_poll_timeouts(monkeypatch, mocker, legacy_timeout):
    orch = Orchestrator.__new__(Orchestrator)
    orch._shutdown_event = asyncio.Event()
    orch._cfg_companion_reaper_interval_s = 0.001
    calls = 0

    async def reap():
        nonlocal calls
        calls += 1
        if calls == 2:
            orch._shutdown_event.set()

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
            mocker.Mock(spec=asyncio, wait_for=wait_for, TimeoutError=LegacyAsyncioTimeoutError),
        )

    await asyncio.wait_for(orch._cfg_companion_reaper_loop(), timeout=1)
    assert calls == 2


@pytest.mark.parametrize("already_stopped", [False, True])
async def test_reaper_shutdown_does_not_reap(monkeypatch, mocker, already_stopped):
    orch = Orchestrator.__new__(Orchestrator)
    orch._shutdown_event = asyncio.Event()
    orch._cfg_companion_reaper_interval_s = 60
    reap = mocker.AsyncMock()
    monkeypatch.setattr(orch, "_reap_expired_cfg_parents", reap)
    if already_stopped:
        orch._shutdown_event.set()
    else:
        asyncio.get_running_loop().call_soon(orch._shutdown_event.set)

    await asyncio.wait_for(orch._cfg_companion_reaper_loop(), timeout=1)
    reap.assert_not_awaited()


async def test_cfg_reaper_propagates_cleanup_failure(monkeypatch, mocker):
    orch = Orchestrator.__new__(Orchestrator)
    orch._shutdown_event = asyncio.Event()
    orch._cfg_companion_reaper_interval_s = 0.001
    monkeypatch.setattr(orch, "_reap_expired_cfg_parents", mocker.AsyncMock(side_effect=RuntimeError("cleanup failed")))

    with pytest.raises(RuntimeError, match="cleanup failed"):
        await asyncio.wait_for(orch._cfg_companion_reaper_loop(), timeout=1)


def test_cfg_reaper_runs_alongside_subclass_background_tasks(monkeypatch, mocker):
    async def exercise():
        orch = Orchestrator(
            request_async_queue=asyncio.Queue(),
            output_async_queue=asyncio.Queue(),
            rpc_async_queue=asyncio.Queue(),
            stage_pools=[],
            cfg_companion_timeout=0.001,
        )
        background_started = asyncio.Event()

        async def background():
            background_started.set()
            await orch._shutdown_event.wait()

        async def reap():
            await background_started.wait()
            raise RuntimeError("cleanup failed")

        monkeypatch.setattr(orch, "_background_tasks", lambda: [background()])
        monkeypatch.setattr(orch, "_request_handler", orch._shutdown_event.wait)
        monkeypatch.setattr(orch, "_orchestration_output_handler", orch._shutdown_event.wait)
        monkeypatch.setattr(orch, "_reap_expired_cfg_parents", reap)
        shutdown = mocker.Mock()
        monkeypatch.setattr(orch, "_shutdown_stages", shutdown)
        # If the CFG task is accidentally omitted, stop the other loops and
        # fail the exception assertion instead of hanging the regression.
        stop = asyncio.get_running_loop().call_later(1, orch._shutdown_event.set)
        try:
            with pytest.raises(RuntimeError, match="cleanup failed"):
                await orch.run()
        finally:
            stop.cancel()
        assert background_started.is_set()
        assert orch._shutdown_event.is_set()
        shutdown.assert_called_once()

    # run() owns its loop and cancels pending tasks during teardown.
    asyncio.run(exercise())


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_engine_rejects_invalid_cfg_timeout_before_startup(value):
    with pytest.raises(ValueError, match="cfg-companion-timeout must be finite and > 0"):
        AsyncOmniEngine("unused", cfg_companion_timeout=value)


def test_engine_bootstrap_passes_cfg_timeout_to_orchestrator(monkeypatch, mocker):
    engine = AsyncOmniEngine.__new__(AsyncOmniEngine)
    engine._cfg_companion_timeout_s = 0.125
    engine._event_driven_orch_default = False
    engine._engines_waiting_counter = None
    engine._running_counter = None
    engine.async_chunk = False
    engine.stage_pools = []
    engine._runtime = mocker.Mock(spec=StageRuntime)
    engine._runtime.create_membership_controller.return_value = None
    engine.request_queue = engine.output_queue = engine.rpc_output_queue = mocker.Mock(spec=janus.Queue, async_q=None)
    monkeypatch.setattr(engine, "_initialize_stages", lambda timeout: None)
    monkeypatch.setattr(engine, "_detect_pd_config", lambda: None)
    run = mocker.patch.object(Orchestrator, "run", new_callable=mocker.AsyncMock)
    factory = mocker.Mock(wraps=engine._create_orchestrator)
    monkeypatch.setattr(engine, "_create_orchestrator", factory)
    startup: concurrent.futures.Future[asyncio.AbstractEventLoop] = concurrent.futures.Future()

    engine._bootstrap_orchestrator(1, startup)

    assert startup.done() and startup.exception() is None
    assert factory.call_args.kwargs["cfg_companion_timeout"] == 0.125
    run.assert_awaited_once()
