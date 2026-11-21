# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import multiprocessing
import os
import signal
import socket
import sys
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from tests.entrypoints.openai_api.test_api_server_guards import (
    _FakeParallelConfig,
    _FakeSocket,
    _FakeVllmConfig,
    _minimal_args,
)
from tests.entrypoints.test_async_omni_idle_sleep import make_omni
from vllm_omni.config.omni_config import OmniStageModelConfig
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.async_omni_base import AsyncOmniBase
from vllm_omni.entrypoints.omni_base import OmniEngineDeadError
from vllm_omni.entrypoints.openai import api_server
from vllm_omni.entrypoints.utils import PureDiffusionLauncherAdapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _prepare_server(monkeypatch, omni, app, *, timeout, pure_diffusion):
    """Share only server setup; each test owns its shutdown behavior and assertions."""
    omni._idle_timeout_s = None
    omni.engine.tracer_provider = None
    for idx, stage in enumerate(omni.engine.stage_configs):
        stage.stage_id = idx
        stage.model_config = OmniStageModelConfig(enable_sleep_mode=True)
    config = None if pure_diffusion else _FakeVllmConfig(_FakeParallelConfig(0))
    if config is not None:
        config.shutdown_timeout = timeout

    async def init_state(engine, state, args):
        state.engine_client = engine

    monkeypatch.setattr(AsyncOmni, "vllm_config", property(lambda _: config))
    monkeypatch.setattr(AsyncOmni, "weight_transfer_enabled", property(lambda _: False))
    monkeypatch.setattr(omni, "get_supported_tasks", AsyncMock(return_value=("generate",)))
    monkeypatch.setattr(omni, "get_diffusion_od_config", lambda: None)
    monkeypatch.setattr(omni, "_get_pd_separation_pair", lambda: None)
    monkeypatch.setattr(api_server, "build_openai_app", lambda *_: app)
    monkeypatch.setattr(api_server, "omni_init_app_state", init_state)
    monkeypatch.setattr(api_server, "shutdown_unsupported_routes", lambda *_: None)
    monkeypatch.setattr(api_server, "_should_enable_profiler_endpoints", lambda *_: False)
    monkeypatch.setattr(api_server.openai_app_state, "_get_vllm_config", AsyncMock(return_value=config))
    monkeypatch.setattr(api_server.STORAGE_MANAGER, "start", AsyncMock())
    monkeypatch.setattr(api_server.STORAGE_MANAGER, "stop", AsyncMock())
    return config


@pytest.mark.parametrize(
    "timeout,pure_diffusion,startup_failure",
    [(0.0, False, False), (1.0, True, False), (0.0, False, True), (0.0, True, True)],
)
async def test_serving_closes_idle_policy_before_engine_context(timeout, pure_diffusion, startup_failure, monkeypatch):
    omni = make_omni(("diffusion",) if pure_diffusion else ("llm",))
    events = []
    config = _prepare_server(monkeypatch, omni, FastAPI(), timeout=timeout, pure_diffusion=pure_diffusion)

    @asynccontextmanager
    async def build(*args, **kwargs):
        try:
            yield omni
        finally:
            omni.shutdown()
            assert omni._idle_closed
            assert omni._idle_timer is None
            events.append("engine_context")

    def base_shutdown(self, timeout=None):
        assert self._idle_closed
        events.append(("engine_shutdown", timeout))

    async def serve(asgi, **kwargs):
        assert omni._idle_timeout_s == 300.0
        client = asgi.state.engine_client
        if pure_diffusion:
            assert isinstance(client, PureDiffusionLauncherAdapter)
            assert client._wrapped is omni
        else:
            assert client is omni
            assert client.vllm_config is config
        if startup_failure:
            raise RuntimeError("listener failed")
        await asyncio.to_thread(client.shutdown, timeout=timeout)
        return asyncio.create_task(asyncio.sleep(0))

    monkeypatch.setattr(AsyncOmniBase, "shutdown", base_shutdown)
    monkeypatch.setattr(api_server, "build_async_omni", build)
    monkeypatch.setattr(api_server, "serve_http", serve)
    args = _minimal_args(idle_sleep_timeout=300.0, shutdown_timeout=timeout)
    if startup_failure:
        with pytest.raises(RuntimeError, match="listener failed"):
            await api_server.omni_run_server_worker("localhost:8000", _FakeSocket(), args)
        assert events == [("engine_shutdown", None), "engine_context"]
    else:
        await api_server.omni_run_server_worker("localhost:8000", _FakeSocket(), args)
        assert events == [("engine_shutdown", timeout), ("engine_shutdown", None), "engine_context"]


@pytest.mark.parametrize("keep_alive", ["none", "environment", "engine"])
def test_upstream_watchdog_observes_idle_failure_with_live_engine(keep_alive, monkeypatch, mocker):
    from vllm.entrypoints.launchers import launcher

    omni = make_omni()
    omni.engine.is_alive = lambda: True
    omni.final_output_task = mocker.Mock()
    omni.final_output_task.done.return_value = False
    omni._idle_error = RuntimeError("automatic sleep failed")
    monkeypatch.setattr(launcher.envs, "VLLM_KEEP_ALIVE_ON_ENGINE_DEATH", keep_alive == "environment")
    monkeypatch.setattr(AsyncOmni, "should_keep_api_server_alive", keep_alive == "engine", raising=False)
    server = mocker.Mock(should_exit=False)
    launcher.terminate_if_errored(server, omni)
    assert server.should_exit == (keep_alive == "none")


def _signal_server(connection, timeout, pure_diffusion):
    """Exercise the installed launcher and production shutdown callback in a child."""

    async def run():
        omni = make_omni(("diffusion",) if pure_diffusion else ("llm",))
        started, destroyed = asyncio.Event(), asyncio.Event()
        loop = asyncio.get_running_loop()

        async def wake():
            started.set()
            while not destroyed.is_set():
                try:
                    await destroyed.wait()
                except asyncio.CancelledError:
                    # Model a device operation that local cancellation cannot stop.
                    continue

        def base_shutdown(self, timeout=None):
            if not destroyed.is_set():
                connection.send(("destroyed", self._idle_closed, timeout))
                loop.call_soon_threadsafe(destroyed.set)

        @asynccontextmanager
        async def lifespan(app):
            omni._sleeping_tags = {"weights", "kv_cache"}
            omni._paused = True
            waiter = asyncio.create_task(omni._wait_for_admission())
            try:
                yield
            finally:
                try:
                    await waiter
                except (OmniEngineDeadError, asyncio.CancelledError):
                    pass

        app = FastAPI(lifespan=lifespan)

        @app.get("/_ready")
        async def _ready():
            return JSONResponse({"ready": started.is_set()}, status_code=200 if started.is_set() else 503)

        @asynccontextmanager
        async def build(*args, **kwargs):
            try:
                yield omni
            finally:
                omni.shutdown()
                assert omni._idle_closed
                connection.send(("context_closed",))

        with pytest.MonkeyPatch.context() as patcher:
            _prepare_server(patcher, omni, app, timeout=timeout, pure_diffusion=pure_diffusion)
            patcher.setattr(omni, "wake_up", wake)
            patcher.setattr(AsyncOmniBase, "shutdown", base_shutdown)
            patcher.setattr(api_server, "build_async_omni", build)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
                connection.send(("listening", port))
                args = _minimal_args(idle_sleep_timeout=300.0, shutdown_timeout=timeout, port=port)
                await api_server.omni_run_server_worker(f"127.0.0.1:{port}", sock, args)
        connection.send(("finished",))

    try:
        asyncio.run(run())
    except BaseException as exc:
        connection.send(("error", repr(exc)))
        raise
    finally:
        connection.close()


def _wait_for_ready(port: int, receive, deadline: float) -> None:
    url = f"http://127.0.0.1:{port}/_ready"
    while time.monotonic() < deadline:
        if receive.poll():
            raise AssertionError(f"child reported unexpected event before ready: {receive.recv()}")
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.05)
    raise AssertionError(f"server on port {port} did not reach blocked wake")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX launcher signals")
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
@pytest.mark.parametrize("timeout", [0.0, 0.05])
@pytest.mark.parametrize("pure_diffusion", [False, True])
def test_real_launcher_signal_closes_before_destroying_uncooperative_engine(signum, timeout, pure_diffusion):
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_signal_server, args=(send, timeout, pure_diffusion))
    process.start()
    send.close()
    try:
        assert receive.poll(60), "server did not start listening"
        event, port = receive.recv()
        assert event == "listening"
        _wait_for_ready(port, receive, time.monotonic() + 30.0)
        os.kill(process.pid, signum)
        assert receive.poll(10), "shutdown did not reach engine destruction within its budget"
        assert receive.recv() == ("destroyed", True, timeout)
        assert receive.poll(10)
        assert receive.recv() == ("context_closed",)
        assert receive.poll(10)
        assert receive.recv() == ("finished",)
        process.join(timeout=5)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        receive.close()
