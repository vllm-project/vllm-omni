# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Session-only backend through public handles and the standard HTTP server."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI

from tests.engine.duplex.test_process_engine import fake_factory
from vllm_omni.config.endpoint_policy import shutdown_unsupported_routes
from vllm_omni.engine.duplex.events import AudioDelta, SessionClosed, SessionCreated
from vllm_omni.entrypoints.duplex.warmup import lookup_duplex_plugin, warmup_silence_unit
from vllm_omni.entrypoints.duplex_omni import DuplexOmni
from vllm_omni.entrypoints.openai import api_server

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def process_omni(monkeypatch):
    monkeypatch.setattr("vllm_omni.entrypoints.duplex_omni._ProcessWorkerRuntime", fake_factory)
    omni = DuplexOmni("fake", duplex_engine_backend="process", init_timeout=30)
    yield omni
    omni.shutdown()


@pytest.mark.parametrize("backend", ["invalid", "", None])
def test_invalid_backend_fails_before_startup(backend):
    with pytest.raises(ValueError, match="duplex_engine_backend"):
        DuplexOmni("fake", duplex_engine_backend=backend)


@pytest.mark.asyncio
async def test_public_handles_and_session_only_guard(process_omni):
    omni = process_omni
    assert omni.session_only
    assert omni.duplex_capabilities.supports_chat_completions
    assert await omni.get_supported_tasks() == ()
    assert await omni.get_vllm_config() is None
    assert omni.vllm_config is None
    assert omni.model_config is None
    assert omni.get_diffusion_od_config() is None
    with pytest.raises(NotImplementedError, match="sessions only"):
        await anext(omni.generate("hello"))
    handles = await asyncio.gather(*(omni.open_session() for _ in range(4)))

    async def exercise(handle):
        stream = handle.events()
        assert isinstance(await anext(stream), SessionCreated)
        for i in range(20):
            await handle.append_audio(bytes([i]) * 1920)
        for i in range(20):
            event = await asyncio.wait_for(anext(stream), 5)
            assert isinstance(event, AudioDelta)
            assert event.audio == bytes([i]) * 1920
        await omni.resume_session(handle.session_id, expected_lease_generation=0)
        assert handle.lease_generation == 1
        await omni.detach_session(handle.session_id, expected_lease_generation=1)
        await handle.close()
        assert isinstance(await anext(stream), SessionClosed)
        await stream.aclose()

    await asyncio.gather(*(exercise(handle) for handle in handles))
    assert not omni.sessions
    await omni.aclose()
    assert not omni.engine.is_alive()


@pytest.mark.asyncio
async def test_server_models_health_and_chat_restriction(process_omni):
    from tests.entrypoints.openai_api.test_duplex_api_server import _minimal_args

    app = FastAPI()
    app.include_router(api_server.router)
    await api_server.omni_init_app_state(process_omni, app.state, _minimal_args(model="fake"))
    shutdown_unsupported_routes(app, process_omni.endpoint_restrictions)
    assert app.state.openai_serving_chat is None
    assert app.state.openai_serving_duplex is not None
    assert app.state.online_renderer is None
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/health")).status_code == 200
        models = await client.get("/v1/models")
        assert models.status_code == 200
        assert models.json()["data"][0]["id"] == "fake"
        chat = await client.post("/v1/chat/completions", json={"model": "fake", "messages": []})
        assert chat.status_code == 400
        assert "sessions only" in chat.text
    plugin = lookup_duplex_plugin(process_omni)
    assert warmup_silence_unit(plugin)["sample_rate_hz"] == 24000
    assert plugin.capabilities(max_sessions=1).supports_chat_completions
    await process_omni.aclose()


@pytest.mark.asyncio
async def test_slow_frontend_session_does_not_kill_healthy_session(monkeypatch):
    monkeypatch.setattr("vllm_omni.entrypoints.duplex_omni._ProcessWorkerRuntime", fake_factory)
    omni = DuplexOmni("fake", duplex_engine_backend="process", init_timeout=30, small_buffers=True)
    try:
        slow = await omni.open_session()
        for _ in range(10):
            if slow.closed:
                break
            await slow.append_audio(b"audio")
            await asyncio.sleep(0.05)
        assert await asyncio.wait_for(slow.wait_closed(), 5) == "frontend_output_overflow"
        assert omni.engine.is_alive()
        healthy = await omni.open_session()
        await healthy.close()
    finally:
        await omni.aclose()


@pytest.mark.asyncio
async def test_worker_death_closes_handles(process_omni):
    handle = await process_omni.open_session()
    stream = handle.events()
    await anext(stream)
    await handle.append_text("crash")
    assert "engine_dead" in await asyncio.wait_for(handle.wait_closed(), 5)
    assert not process_omni.sessions
    await stream.aclose()
