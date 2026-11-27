# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import socket
from functools import partial

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
async def test_media_request_uses_checked_ip_with_original_host(mocker, install_transport):
    resolve = mocker.patch.object(
        asyncio.get_running_loop(),
        "getaddrinfo",
        return_value=[_address_info("93.184.216.34", 8443), _address_info("1.1.1.1", 8443)],
    )
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
async def test_media_request_tries_backup_after_connect_timeout(mocker, install_transport):
    first_address = "2606:4700:4700::1111"
    backup_address = "93.184.216.34"
    mocker.patch.object(
        asyncio.get_running_loop(),
        "getaddrinfo",
        return_value=[
            _address_info(first_address, 443),
            _address_info("10.0.0.1", 443),
            _address_info(backup_address, 443),
        ],
    )
    attempted_hosts = []

    async def handle(request):
        attempted_hosts.append(request.url.host)
        if request.url.host == first_address:
            try:
                await asyncio.wait_for(asyncio.Event().wait(), request.extensions["timeout"]["connect"])
            except TimeoutError as exc:
                raise httpx.ConnectTimeout("connection timed out", request=request) from exc
        assert request.url.host == backup_address
        assert request.headers["host"] == "media.example.com"
        assert request.extensions["sni_hostname"] == "media.example.com"
        return httpx.Response(200, content=b"media")

    install_transport(handle)

    assert (
        await media_input.download_media_bytes(
            "https://media.example.com/asset", max_bytes=5, source="reference", timeout_s=0.4
        )
        == b"media"
    )
    assert attempted_hosts == [first_address, backup_address]


@pytest.mark.asyncio
@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.1", "::1", "::ffff:10.0.0.1", "ff02::1"])
async def test_media_request_rejects_non_public_ip(mocker, install_transport, address):
    mocker.patch.object(asyncio.get_running_loop(), "getaddrinfo", return_value=[_address_info(address, 443)])
    install_transport(lambda request: pytest.fail("Non-public media addresses must not be contacted"))

    with pytest.raises(InvalidInputReferenceError, match="public IP address"):
        await media_input.download_media_bytes("https://media.example.com/asset", max_bytes=5, source="reference")


@pytest.mark.asyncio
async def test_redirect_media_request_checks_new_destination(mocker, install_transport):
    resolve = mocker.patch.object(
        asyncio.get_running_loop(),
        "getaddrinfo",
        side_effect=[[_address_info("93.184.216.34", 443)], [_address_info("10.0.0.1", 443)]],
    )
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
@pytest.mark.parametrize(
    ("location", "expect_cookies"),
    [
        pytest.param("/assets/media", True, id="same-origin"),
        pytest.param("/other/media", False, id="different-path"),
        pytest.param("https://cdn.media.example.com/assets/media", False, id="different-host"),
        pytest.param("https://media.example.com:8443/assets/media", False, id="different-port"),
        pytest.param("http://media.example.com/assets/media", False, id="different-scheme"),
    ],
)
async def test_redirect_media_cookies_follow_the_logical_origin_and_path(
    mocker, install_transport, location, expect_cookies
):
    mocker.patch.object(asyncio.get_running_loop(), "getaddrinfo", return_value=[_address_info("93.184.216.34", 443)])

    def handle(request):
        assert request.url.host == "93.184.216.34"
        if request.url.path == "/start":
            assert "cookie" not in request.headers
            return httpx.Response(
                302,
                headers=[
                    ("location", location),
                    ("set-cookie", "host_session=valid; Path=/assets"),
                    ("set-cookie", "domain_session=valid; Domain=media.example.com; Path=/assets"),
                ],
            )
        if expect_cookies:
            assert set(request.headers["cookie"].split("; ")) == {"host_session=valid", "domain_session=valid"}
        else:
            assert "cookie" not in request.headers
        return httpx.Response(200, content=b"media")

    install_transport(handle)

    assert (
        await media_input.download_media_bytes(
            "https://media.example.com/start", max_bytes=5, source="reference", follow_redirects=True
        )
        == b"media"
    )


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


@pytest.mark.asyncio
async def test_media_download_timeout_closes_a_stalled_body(mocker, install_transport):
    class StalledBody(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield b"partial media"
            await asyncio.Event().wait()

        async def aclose(self):
            self.closed = True

    body = StalledBody()
    mocker.patch.object(asyncio.get_running_loop(), "getaddrinfo", return_value=[_address_info("93.184.216.34", 443)])
    install_transport(lambda request: httpx.Response(200, stream=body))

    with pytest.raises(media_input.MediaInputTimeoutError):
        await media_input.download_media_bytes(
            "https://media.example.com/asset", max_bytes=None, source="reference", timeout_s=0.05
        )

    assert body.closed
