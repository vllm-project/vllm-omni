# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded input-media reads for OpenAI-compatible endpoints.

This module owns the byte and time limits applied before image, video, or audio
input reaches a decoder or a request-scoped temporary file. It deliberately
does not own endpoint parsing, model policy, or output encoding.
"""

from __future__ import annotations

import asyncio
import binascii
import ipaddress
import socket
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Final

import httpx
import pybase64 as base64
from vllm import envs

from vllm_omni.entrypoints.openai.errors import InvalidInputReferenceError

_MIB: Final = 1024 * 1024


def _general_limit_bytes(megabytes: int) -> int | None:
    """Convert the optional general media ceiling to bytes."""
    return None if megabytes <= 0 else megabytes * _MIB


def _audio_limit_bytes(megabytes: int, general_limit: int | None) -> int:
    """Keep the audio ceiling in effect even when it allows zero bytes."""
    audio_limit = max(0, megabytes) * _MIB
    return audio_limit if general_limit is None else min(general_limit, audio_limit)


# Keep these limits aligned with the vLLM 0.34.0 media configuration.
_GENERAL_INPUT_BYTES: Final = _general_limit_bytes(envs.VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB)
MAX_IMAGE_INPUT_BYTES: Final = _GENERAL_INPUT_BYTES
MAX_VIDEO_INPUT_BYTES: Final = _GENERAL_INPUT_BYTES
MAX_AUDIO_INPUT_BYTES: Final = _audio_limit_bytes(envs.VLLM_MAX_AUDIO_CLIP_FILESIZE_MB, _GENERAL_INPUT_BYTES)
MEDIA_DOWNLOAD_TIMEOUT_S: Final = 60.0
_READ_CHUNK_SIZE: Final = 64 * 1024
_MAX_REDIRECTS: Final = 20


class MediaInputTooLargeError(InvalidInputReferenceError):
    """An input media payload exceeded its byte limit."""


class MediaInputTimeoutError(InvalidInputReferenceError):
    """An input media payload exceeded its total download time limit."""


def decode_base64_media(data: str, *, max_bytes: int | None, source: str) -> bytes:
    """Decode base64 media while bounding the decoded payload size.

    Args:
        data: Base64 payload without a data-URL prefix.
        max_bytes: Maximum number of decoded bytes permitted, or ``None`` to
            disable the byte limit.
        source: User-facing input name for error messages.

    Returns:
        The decoded media bytes.

    Raises:
        InvalidInputReferenceError: If the payload is malformed or empty.
        MediaInputTooLargeError: If the encoded or decoded payload exceeds the limit.
    """
    if max_bytes is not None and max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    if not data:
        raise InvalidInputReferenceError(f"Invalid {source}: media data is empty.")

    if max_bytes is not None:
        max_encoded_length = 4 * ((max_bytes + 2) // 3)
        if len(data) > max_encoded_length:
            raise MediaInputTooLargeError(
                f"Invalid {source}: media data exceeds the {max_bytes // _MIB} MiB size limit."
            )

    try:
        decoded = base64.b64decode(data)
    except (binascii.Error, ValueError) as exc:
        raise InvalidInputReferenceError(f"Invalid {source}: media data is not valid base64.") from exc

    if max_bytes is not None and len(decoded) > max_bytes:
        raise MediaInputTooLargeError(f"Invalid {source}: media data exceeds the {max_bytes // _MIB} MiB size limit.")
    return decoded


@asynccontextmanager
async def _stream_media_response(
    client: httpx.AsyncClient, url: httpx.URL, *, source: str, cookies: httpx.Cookies
) -> AsyncIterator[httpx.Response]:
    """Connect to a checked address and match cookies against the logical URL."""
    if url.scheme not in ("http", "https") or not url.host or url.userinfo:
        raise InvalidInputReferenceError(f"Invalid {source}: expected an HTTP(S) URL without credentials.")

    hostname = url.raw_host.decode("ascii")
    try:
        addresses = await asyncio.get_running_loop().getaddrinfo(
            hostname, url.port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP
        )
    except OSError as exc:
        raise InvalidInputReferenceError(f"Invalid {source}: could not resolve the media host.") from exc

    request = httpx.Request("GET", url, headers={"Accept-Encoding": "identity"})
    cookies.set_cookie_header(request)
    connect_error: httpx.ConnectError | None = None
    async with AsyncExitStack() as stack:
        for _, _, _, _, sockaddr in addresses:
            address = ipaddress.ip_address(sockaddr[0])
            if isinstance(address, ipaddress.IPv6Address):
                if address.is_site_local or address.sixtofour is not None:
                    continue
                address = address.ipv4_mapped or address
            if not address.is_global or address.is_multicast or address.is_reserved:
                continue

            # Connect to the checked address while preserving HTTP routing and TLS verification.
            target = httpx.URL(scheme=url.scheme, host=str(address), port=url.port, raw_path=url.raw_path)
            try:
                response = await stack.enter_async_context(
                    client.stream(
                        request.method,
                        target,
                        headers=request.headers,
                        extensions={"sni_hostname": hostname},
                    )
                )
            except httpx.ConnectError as exc:
                connect_error = exc
            else:
                # HTTPX's client jar is keyed to the connection address. Use the logical URL for cookies.
                client.cookies.clear()
                response.request = request
                cookies.extract_cookies(response)
                yield response
                return

    if connect_error is not None:
        raise connect_error
    raise InvalidInputReferenceError(f"Invalid {source}: media URL must resolve to a public IP address.")


async def _download_media_bytes(
    url: str, *, max_bytes: int | None, source: str, follow_redirects: bool, timeout_s: float
) -> bytes:
    async with httpx.AsyncClient(
        timeout=timeout_s,
        follow_redirects=False,
        trust_env=False,
        verify=httpx.create_ssl_context(),
        # Different hostnames can resolve to the same address and need distinct TLS connections.
        limits=httpx.Limits(max_keepalive_connections=0),
    ) as client:
        current_url = httpx.URL(url)
        redirect_count = 0
        cookies_by_origin: defaultdict[tuple[str, bytes], httpx.Cookies] = defaultdict(httpx.Cookies)
        while True:
            origin = (current_url.scheme, current_url.netloc)
            async with _stream_media_response(
                client, current_url, source=source, cookies=cookies_by_origin[origin]
            ) as response:
                if response.has_redirect_location:
                    if not follow_redirects:
                        response.raise_for_status()
                    if redirect_count >= _MAX_REDIRECTS:
                        raise httpx.TooManyRedirects(
                            f"Exceeded {_MAX_REDIRECTS} redirects.",
                            request=response.request,
                        )
                    current_url = current_url.join(response.headers["location"])
                    redirect_count += 1
                    continue

                response.raise_for_status()
                content_encoding = response.headers.get("content-encoding", "").strip().lower()
                if content_encoding and content_encoding != "identity":
                    raise InvalidInputReferenceError(f"Invalid {source}: compressed responses are not supported.")

                content_length = response.headers.get("content-length")
                declared_length = None
                if content_length is not None:
                    try:
                        declared_length = int(content_length)
                    except ValueError:
                        declared_length = None
                if max_bytes is not None and declared_length is not None and declared_length > max_bytes:
                    raise MediaInputTooLargeError(
                        f"Invalid {source}: response exceeds the {max_bytes // _MIB} MiB size limit."
                    )

                payload = bytearray()
                async for chunk in response.aiter_bytes(chunk_size=_READ_CHUNK_SIZE):
                    if max_bytes is not None and len(payload) + len(chunk) > max_bytes:
                        raise MediaInputTooLargeError(
                            f"Invalid {source}: response exceeds the {max_bytes // _MIB} MiB size limit."
                        )
                    payload.extend(chunk)
                return bytes(payload)


async def download_media_bytes(
    url: str,
    *,
    max_bytes: int | None,
    source: str,
    follow_redirects: bool = False,
    timeout_s: float = MEDIA_DOWNLOAD_TIMEOUT_S,
) -> bytes:
    """Download public HTTP(S) media with response-size and total-time limits.

    ``Content-Length`` is checked before the body is read. Responses without a
    trustworthy length are counted incrementally, so chunked and streaming
    responses cannot exceed ``max_bytes``. Redirects are followed one at a
    time so an intermediate response body is never read or decompressed. The
    request advertises identity encoding and rejects non-identity final
    responses before HTTPX can decompress them, so compressed payloads cannot
    bypass the limit. Each hop connects directly to a validated public address.

    Args:
        url: HTTP(S) media URL.
        max_bytes: Maximum number of decoded response bytes permitted, or
            ``None`` to disable the byte limit.
        source: User-facing input name for error messages.
        follow_redirects: Whether to follow redirects one at a time.
        timeout_s: Total time permitted for connection and body download.

    Returns:
        The bounded response body.

    Raises:
        httpx.HTTPError: For HTTP status and transport failures.
        InvalidInputReferenceError: If a URL or response violates the media policy.
        MediaInputTooLargeError: If the response exceeds ``max_bytes``.
        MediaInputTimeoutError: If the total download time is exceeded.
    """
    if max_bytes is not None and max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")

    try:
        return await asyncio.wait_for(
            _download_media_bytes(
                url,
                max_bytes=max_bytes,
                source=source,
                follow_redirects=follow_redirects,
                timeout_s=timeout_s,
            ),
            timeout=timeout_s,
        )
    except httpx.InvalidURL as exc:
        raise InvalidInputReferenceError(f"Invalid {source}: invalid media URL.") from exc
    except TimeoutError as exc:
        raise MediaInputTimeoutError(f"Invalid {source}: download exceeded the {timeout_s:g}s time limit.") from exc


__all__ = [
    "MAX_AUDIO_INPUT_BYTES",
    "MAX_IMAGE_INPUT_BYTES",
    "MAX_VIDEO_INPUT_BYTES",
    "MEDIA_DOWNLOAD_TIMEOUT_S",
    "MediaInputTimeoutError",
    "MediaInputTooLargeError",
    "decode_base64_media",
    "download_media_bytes",
]
