# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""L1 regressions using the real vLLM HTTP reader and loopback HTTP."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pybase64
import pytest
from aiohttp import web
from vllm import envs
from vllm.connections import MediaRequestDownloadBudget

from vllm_omni.entrypoints.openai import media_input
from vllm_omni.entrypoints.openai.errors import InvalidInputReferenceError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
MIB = 1024 * 1024


@pytest.fixture(autouse=True)
def limits(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB", 1)
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DOWNLOAD_SIZE_MB", 1)
    monkeypatch.setattr(envs, "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB", 1)
    monkeypatch.setattr(envs, "VLLM_MEDIA_FETCH_MAX_RETRIES", 1)


@asynccontextmanager
async def serve(handler) -> AsyncIterator[str]:
    app = web.Application()
    app.router.add_get("/{path:.*}", handler)
    runner = web.AppRunner(app, shutdown_timeout=0.1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        host, port = runner.addresses[0]
        yield f"http://{host}:{port}"
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_configured_download_limit_can_be_raised(monkeypatch):
    payload = b"x" * (MIB + 1)

    async def handler(request):
        return web.Response(body=payload)

    async with serve(handler) as url:
        with pytest.raises(InvalidInputReferenceError, match="exceed"):
            await media_input.download_media_bytes(url, kind="video", source="video")
        monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB", 2)
        assert await media_input.download_media_bytes(url, kind="video", source="video") == payload
        monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB", 1)
        with pytest.raises(InvalidInputReferenceError, match="exceed"):
            await media_input.download_media_bytes(url, kind="video", source="video")
        monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB", 0)
        assert await media_input.download_media_bytes(url, kind="video", source="video") == payload


@pytest.mark.asyncio
@pytest.mark.parametrize("declared", [False, True])
async def test_stream_is_closed_before_eof_on_overflow(declared):
    disconnected = asyncio.Event()
    chunks_sent = 0

    async def handler(request):
        nonlocal chunks_sent
        headers = {"Content-Length": str(8 * MIB)} if declared else {}
        response = web.StreamResponse(headers=headers)
        await response.prepare(request)
        try:
            for _ in range(128):
                await response.write(b"x" * 65536)
                chunks_sent += 1
                await asyncio.sleep(0.001)
        except ConnectionResetError:
            disconnected.set()
        return response

    async with serve(handler) as url:
        with pytest.raises(InvalidInputReferenceError, match="exceed"):
            await media_input.download_media_bytes(url, kind="image", source="image")
        await asyncio.wait_for(disconnected.wait(), 2)
        assert chunks_sent < 128


@pytest.mark.asyncio
async def test_audio_uses_the_stricter_upstream_limit(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB", 2)
    payload = b"x" * (MIB + 1)

    async def handler(request):
        return web.Response(body=payload)

    async with serve(handler) as url:
        assert await media_input.download_media_bytes(url, kind="image", source="image") == payload
        with pytest.raises(InvalidInputReferenceError, match="exceed"):
            await media_input.download_media_bytes(url, kind="audio", source="audio")


@pytest.mark.asyncio
async def test_request_budget_is_shared_and_new_requests_are_independent():
    payload = b"x" * (3 * MIB // 4)

    async def handler(request):
        return web.Response(body=payload)

    async with serve(handler) as url:
        budget = MediaRequestDownloadBudget()
        assert (
            await media_input.download_media_bytes(url, kind="image", source="image", request_budget=budget) == payload
        )
        with pytest.raises(InvalidInputReferenceError, match="Request media downloads"):
            await media_input.download_media_bytes(url, kind="video", source="video", request_budget=budget)
        assert (
            await media_input.download_media_bytes(
                url, kind="image", source="image", request_budget=MediaRequestDownloadBudget()
            )
            == payload
        )


@pytest.mark.asyncio
async def test_gzip_redirect_body_is_not_decompressed_or_drained():
    disconnected = asyncio.Event()

    async def handler(request):
        if request.path == "/final":
            return web.Response(body=b"ok")
        response = web.StreamResponse(status=302, headers={"Location": "/final", "Content-Encoding": "gzip"})
        await response.prepare(request)
        try:
            while True:
                await response.write(b"not-a-gzip-body")
                await asyncio.sleep(0.001)
        except ConnectionResetError:
            disconnected.set()
        return response

    async with serve(handler) as url:
        assert (
            await media_input.download_media_bytes(
                url, kind="image", source="image", follow_redirects=True, timeout_s=2
            )
            == b"ok"
        )
        await asyncio.wait_for(disconnected.wait(), 2)


@pytest.mark.asyncio
async def test_final_compressed_response_is_rejected_before_decoding():
    async def handler(request):
        return web.Response(body=b"not-a-gzip-body", headers={"Content-Encoding": "gzip"})

    async with serve(handler) as url:
        with pytest.raises(InvalidInputReferenceError, match="compressed responses"):
            await media_input.download_media_bytes(url, kind="image", source="image")


@pytest.mark.asyncio
async def test_disabled_redirects_do_not_reach_target():
    paths = []

    async def handler(request):
        paths.append(request.path)
        return web.Response(status=302, headers={"Location": "/final"})

    async with serve(handler) as url:
        with pytest.raises(InvalidInputReferenceError, match="redirect"):
            await media_input.download_media_bytes(url, kind="image", source="image")
    assert paths == ["/"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [304, 404])
async def test_http_status_and_connection_errors_are_input_errors(status):
    async def handler(request):
        return web.Response(status=status)

    async with serve(handler) as url:
        with pytest.raises(InvalidInputReferenceError, match=f"server returned HTTP {status}"):
            await media_input.download_media_bytes(url, kind="image", source="image")
    # The server's port is now closed; exercise the real connector failure.
    with pytest.raises(InvalidInputReferenceError, match="failed to download audio"):
        await media_input.download_media_bytes(url, kind="audio", source="audio")


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_deadline_and_cancellation_close_the_session(monkeypatch, cancel):
    sessions = []
    session_factory = media_input.aiohttp.ClientSession

    def record_session(*args, **kwargs):
        session = session_factory(*args, **kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(media_input.aiohttp, "ClientSession", record_session)

    async def handler(request):
        await asyncio.sleep(5)
        return web.Response(body=b"late")

    async with serve(handler) as url:
        task = asyncio.create_task(media_input.download_media_bytes(url, kind="image", source="image", timeout_s=0.1))
        if cancel:
            await asyncio.sleep(0.02)
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else InvalidInputReferenceError):
            await task
        assert sessions and all(session.closed for session in sessions)


@pytest.mark.parametrize("kind", ["image", "video", "audio"])
def test_data_url_size_is_checked_before_base64_allocation(monkeypatch, kind):
    def unexpected_decode(*args, **kwargs):
        pytest.fail("oversized encoded input reached the base64 decoder")

    monkeypatch.setattr(media_input.pybase64, "b64decode", unexpected_decode)
    with pytest.raises(InvalidInputReferenceError, match="exceeds"):
        media_input.decode_base64_media("A" * (4 * ((MIB + 2) // 3) + 4), kind=kind, source="data URL")


def test_data_url_uses_audio_limit_and_detects_padding_boundary(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB", 2)
    payload = b"x" * (MIB + 1)
    encoded = pybase64.b64encode(payload).decode()
    assert media_input.decode_base64_media(encoded, kind="image", source="image") == payload
    with pytest.raises(InvalidInputReferenceError, match="exceeds"):
        media_input.decode_base64_media(encoded, kind="audio", source="audio")
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB", 1)
    with pytest.raises(InvalidInputReferenceError, match="exceeds"):
        media_input.decode_base64_media(encoded, kind="image", source="image")
