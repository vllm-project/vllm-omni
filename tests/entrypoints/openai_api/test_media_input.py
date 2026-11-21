# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import socket
from functools import partial
from unittest.mock import AsyncMock

import httpx
import pytest

from vllm_omni.entrypoints.openai import media_input
from vllm_omni.entrypoints.openai.errors import InvalidInputReferenceError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def install_transport(monkeypatch):
    def install(handler):
        client = partial(httpx.AsyncClient, transport=httpx.MockTransport(handler))
        monkeypatch.setattr(media_input.httpx, "AsyncClient", client)

    return install


def _address_info(address: str, port: int):
    if ":" in address:
        return (socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, port, 0, 0))
    return (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, port))


@pytest.mark.asyncio
async def test_media_request_uses_checked_ip_with_original_host(monkeypatch, install_transport):
    resolve = AsyncMock(return_value=[_address_info("93.184.216.34", 8443), _address_info("1.1.1.1", 8443)])
    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
    attempted_hosts = []

    def handle(request):
        attempted_hosts.append(request.url.host)
        assert request.url.port == 8443
        assert request.url.raw_path == b"/asset?signature=a%2Fb"
        assert request.headers["host"] == "media.example.com:8443"
        assert request.extensions["sni_hostname"] == "media.example.com"
        if request.url.host == "93.184.216.34":
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, content=b"media")

    install_transport(handle)

    assert (
        await media_input.download_media_bytes(
            "https://media.example.com:8443/asset?signature=a%2Fb", max_bytes=5, source="reference"
        )
        == b"media"
    )
    resolve.assert_awaited_once_with("media.example.com", 8443, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP)
    assert attempted_hosts == ["93.184.216.34", "1.1.1.1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.1", "::1", "::ffff:10.0.0.1", "ff02::1"])
async def test_media_request_rejects_non_public_ip(monkeypatch, install_transport, address):
    resolve = AsyncMock(return_value=[_address_info(address, 443)])
    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
    install_transport(lambda request: pytest.fail("Non-public media addresses must not be contacted"))

    with pytest.raises(InvalidInputReferenceError, match="public IP address"):
        await media_input.download_media_bytes("https://media.example.com/asset", max_bytes=5, source="reference")


@pytest.mark.asyncio
async def test_redirect_media_request_checks_new_destination(monkeypatch, install_transport):
    resolve = AsyncMock(side_effect=[[_address_info("93.184.216.34", 443)], [_address_info("10.0.0.1", 443)]])
    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
    contacted = []

    def handle(request):
        contacted.append(request.headers["host"])
        return httpx.Response(302, headers={"location": "https://private.example.com/asset"})

    install_transport(handle)

    with pytest.raises(InvalidInputReferenceError, match="public IP address"):
        await media_input.download_media_bytes(
            "https://media.example.com/asset", max_bytes=5, source="reference", follow_redirects=True
        )

    assert contacted == ["media.example.com"]
    assert [call.args[0] for call in resolve.await_args_list] == ["media.example.com", "private.example.com"]


@pytest.mark.asyncio
async def test_media_download_timeout_includes_address_resolution(monkeypatch, install_transport):
    async def resolve(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", resolve)
    install_transport(lambda request: pytest.fail("Resolution must finish before connecting"))

    with pytest.raises(media_input.MediaInputTimeoutError):
        await media_input.download_media_bytes(
            "https://media.example.com/asset", max_bytes=5, source="reference", timeout_s=0.05
        )
