# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import web

from benchmarks.diffusion.backends import (
    RequestFuncInput,
    async_request_chat_completions,
    async_request_image_edits,
    async_request_openai_image_generations,
    async_request_v1_videos,
    endpoint_filename_token,
    normalize_endpoint,
)


@pytest_asyncio.fixture
async def delayed_generation_server(unused_tcp_port: int) -> AsyncIterator[str]:
    async def respond(request: web.Request) -> web.Response:
        video_status = "completed"
        if request.method == "POST" and request.path == "/v1/videos":
            form = await request.post()
            video_status = "queued" if form.get("model") == "queued" else "completed"
        else:
            await request.read()
        await asyncio.sleep(0.05)
        if request.method == "POST" and request.path == "/v1/videos":
            return web.json_response({"id": "job", "status": video_status})
        if request.path == "/v1/videos/job":
            return web.json_response({"id": "job", "status": "completed"})
        if request.path.endswith("/content"):
            return web.Response(body=b"video")
        return web.json_response({"data": [], "choices": []})

    application = web.Application()
    application.router.add_route("*", "/{path:.*}", respond)
    runner = web.AppRunner(application)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", unused_tcp_port)
    await site.start()
    try:
        yield f"http://127.0.0.1:{unused_tcp_port}"
    finally:
        await runner.cleanup()


@pytest.mark.core_model
@pytest.mark.benchmark
@pytest.mark.cpu
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "request_function"),
    [
        ("/v1/chat/completions", async_request_chat_completions),
        ("/v1/images/generations", async_request_openai_image_generations),
        ("/v1/images/edits", async_request_image_edits),
        ("/v1/videos", async_request_v1_videos),
    ],
)
@pytest.mark.parametrize(("timeout_seconds", "success"), [(0.01, False), (0.0, True), (10.0, True)])
async def test_request_timeout_controls_real_http_requests(
    delayed_generation_server, tmp_path: Path, endpoint, request_function, timeout_seconds, success
):
    """Every backend must honor the request budget, including disabling its total limit."""
    image = tmp_path / "reference.png"
    image.write_bytes(b"reference")
    request = RequestFuncInput(
        prompt="hello",
        api_url=delayed_generation_server + endpoint,
        model="test",
        image_paths=[str(image)],
        request_timeout=timeout_seconds,
    )
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=0.01)) as session:
        output = await request_function(request, session)
    assert output.success is success


@pytest.mark.core_model
@pytest.mark.benchmark
@pytest.mark.cpu
@pytest.mark.asyncio
async def test_unset_request_timeout_preserves_the_session_budget(delayed_generation_server):
    request = RequestFuncInput(
        prompt="hello", api_url=delayed_generation_server + "/v1/images/generations", model="test"
    )
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=0.01)) as session:
        output = await async_request_openai_image_generations(request, session)
    assert not output.success


@pytest.mark.core_model
@pytest.mark.benchmark
@pytest.mark.cpu
@pytest.mark.asyncio
async def test_disabling_total_timeout_preserves_socket_read_limit(delayed_generation_server):
    request = RequestFuncInput(
        prompt="hello",
        api_url=delayed_generation_server + "/v1/images/generations",
        model="test",
        request_timeout=0.0,
    )
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10.0, sock_read=0.01)) as session:
        output = await async_request_openai_image_generations(request, session)
    assert not output.success


@pytest.mark.core_model
@pytest.mark.benchmark
@pytest.mark.cpu
@pytest.mark.asyncio
async def test_request_timeout_reaches_video_polling(delayed_generation_server):
    request = RequestFuncInput(
        prompt="hello",
        api_url=delayed_generation_server + "/v1/videos",
        model="queued",
        request_timeout=0.0,
        video_job_timeout=10.0,
    )
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=0.01)) as session:
        output = await async_request_v1_videos(request, session)
    assert output.success


class _MockResponse:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = payload
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def json(self):
        return self._payload

    async def text(self):
        return str(self._payload)


class _MockSession:
    def __init__(self, payload: dict):
        self._payload = payload

    def post(self, *args, **kwargs):
        return _MockResponse(self._payload)


@pytest.mark.core_model
@pytest.mark.benchmark
@pytest.mark.cpu
def test_endpoint_normalization_accepts_optional_leading_slash():
    assert normalize_endpoint("v1/videos") == "/v1/videos"
    assert normalize_endpoint("/v1/videos") == "/v1/videos"
    assert normalize_endpoint("v1/chat/completions") == "/v1/chat/completions"
    assert normalize_endpoint("v1/images/generations") == "/v1/images/generations"


@pytest.mark.core_model
@pytest.mark.benchmark
@pytest.mark.cpu
def test_endpoint_normalization_accepts_legacy_backend_aliases():
    assert normalize_endpoint("vllm-omni") == "/v1/chat/completions"
    assert normalize_endpoint("openai") == "/v1/images/generations"


@pytest.mark.core_model
@pytest.mark.benchmark
@pytest.mark.cpu
def test_endpoint_filename_token_drops_leading_slash():
    assert endpoint_filename_token("/v1/videos") == "v1_videos"
    assert endpoint_filename_token("v1/chat/completions") == "v1_chat_completions"


@pytest.mark.core_model
@pytest.mark.benchmark
@pytest.mark.cpu
@pytest.mark.asyncio
async def test_chat_completions_metrics_fallback_to_top_level():
    payload = {
        "choices": [
            {
                "message": {
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,abc"},
                        }
                    ]
                }
            }
        ],
        "metrics": {
            "stage_durations": {"diffusion": 1.25},
            "peak_memory_mb": 4096.0,
        },
    }

    output = await async_request_chat_completions(
        RequestFuncInput(
            prompt="draw a cat",
            api_url="http://test.local/v1/chat/completions",
            model="ByteDance-Seed/BAGEL-7B-MoT",
        ),
        session=_MockSession(payload),
    )

    assert output.success is True
    assert output.stage_durations == {"diffusion": 1.25}
    assert output.peak_memory_mb == 4096.0


@pytest.mark.core_model
@pytest.mark.benchmark
@pytest.mark.cpu
@pytest.mark.asyncio
async def test_chat_completions_metrics_message_level_takes_precedence():
    payload = {
        "choices": [
            {
                "message": {
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,abc"},
                            "stage_durations": {"message_stage": 0.7},
                            "peak_memory_mb": 1234.0,
                        }
                    ]
                }
            }
        ],
        "metrics": {
            "stage_durations": {"top_level_stage": 9.9},
            "peak_memory_mb": 9999.0,
        },
    }

    output = await async_request_chat_completions(
        RequestFuncInput(
            prompt="draw a dog",
            api_url="http://test.local/v1/chat/completions",
            model="ByteDance-Seed/BAGEL-7B-MoT",
        ),
        session=_MockSession(payload),
    )

    assert output.success is True
    assert output.stage_durations == {"message_stage": 0.7}
    assert output.peak_memory_mb == 1234.0
