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
from typing import Final

import httpx
import pybase64 as base64

from vllm_omni.entrypoints.openai.errors import InvalidInputReferenceError

MAX_IMAGE_INPUT_BYTES: Final = 30 * 1024 * 1024
MAX_VIDEO_INPUT_BYTES: Final = 50 * 1024 * 1024
MAX_AUDIO_INPUT_BYTES: Final = 15 * 1024 * 1024
MEDIA_DOWNLOAD_TIMEOUT_S: Final = 60.0
_READ_CHUNK_SIZE: Final = 64 * 1024
_MAX_REDIRECTS: Final = 20


class MediaInputTooLargeError(InvalidInputReferenceError):
    """An input media payload exceeded its byte limit."""


class MediaInputTimeoutError(InvalidInputReferenceError):
    """An input media payload exceeded its total download time limit."""


def decode_base64_media(data: str, *, max_bytes: int, source: str) -> bytes:
    """Decode base64 media while bounding the decoded payload size.

    Args:
        data: Base64 payload without a data-URL prefix.
        max_bytes: Maximum number of decoded bytes permitted.
        source: User-facing input name for error messages.

    Returns:
        The decoded media bytes.

    Raises:
        InvalidInputReferenceError: If the payload is malformed or empty.
        MediaInputTooLargeError: If the encoded or decoded payload exceeds the limit.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if not data:
        raise InvalidInputReferenceError(f"Invalid {source}: media data is empty.")

    max_encoded_length = 4 * ((max_bytes + 2) // 3)
    if len(data) > max_encoded_length:
        raise MediaInputTooLargeError(
            f"Invalid {source}: media data exceeds the {max_bytes // (1024 * 1024)} MiB size limit."
        )

    try:
        decoded = base64.b64decode(data)
    except (binascii.Error, ValueError) as exc:
        raise InvalidInputReferenceError(f"Invalid {source}: media data is not valid base64.") from exc

    if len(decoded) > max_bytes:
        raise MediaInputTooLargeError(
            f"Invalid {source}: media data exceeds the {max_bytes // (1024 * 1024)} MiB size limit."
        )
    return decoded


async def download_media_bytes(
    url: str,
    *,
    max_bytes: int,
    source: str,
    follow_redirects: bool = False,
    timeout_s: float = MEDIA_DOWNLOAD_TIMEOUT_S,
) -> bytes:
    """Download media with response-size, compression, and total-time limits.

    ``Content-Length`` is checked before the body is read. Responses without a
    trustworthy length are counted incrementally, so chunked and streaming
    responses cannot exceed ``max_bytes``. Redirects are followed one at a
    time so an intermediate response body is never read or decompressed. The
    request advertises identity encoding and rejects non-identity final
    responses before HTTPX can decompress them, so compressed payloads cannot
    bypass the limit.

    Args:
        url: HTTP(S) media URL.
        max_bytes: Maximum number of decoded response bytes permitted.
        source: User-facing input name for error messages.
        follow_redirects: Whether to follow redirects one at a time.
        timeout_s: Total time permitted for connection and body download.

    Returns:
        The bounded response body.

    Raises:
        httpx.HTTPError: For HTTP status and transport failures.
        MediaInputTooLargeError: If the response exceeds ``max_bytes``.
        MediaInputTimeoutError: If the total download time is exceeded.
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")

    try:
        async with asyncio.timeout(timeout_s):
            async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=False) as client:
                current_url = url
                redirect_count = 0
                while True:
                    async with client.stream(
                        "GET",
                        current_url,
                        headers={"Accept-Encoding": "identity"},
                    ) as response:
                        if response.has_redirect_location:
                            if not follow_redirects:
                                response.raise_for_status()
                            if redirect_count >= _MAX_REDIRECTS:
                                raise httpx.TooManyRedirects(
                                    f"Exceeded {_MAX_REDIRECTS} redirects.",
                                    request=response.request,
                                )
                            current_url = response.url.join(response.headers["location"])
                            redirect_count += 1
                            continue

                        response.raise_for_status()
                        content_encoding = response.headers.get("content-encoding", "").strip().lower()
                        if content_encoding and content_encoding != "identity":
                            raise InvalidInputReferenceError(
                                f"Invalid {source}: compressed responses are not supported."
                            )

                        content_length = response.headers.get("content-length")
                        if content_length is not None:
                            try:
                                declared_length = int(content_length)
                            except ValueError:
                                declared_length = None
                            if declared_length is not None and declared_length > max_bytes:
                                raise MediaInputTooLargeError(
                                    f"Invalid {source}: response exceeds the "
                                    f"{max_bytes // (1024 * 1024)} MiB size limit."
                                )

                        payload = bytearray()
                        async for chunk in response.aiter_bytes(chunk_size=_READ_CHUNK_SIZE):
                            if len(payload) + len(chunk) > max_bytes:
                                raise MediaInputTooLargeError(
                                    f"Invalid {source}: response exceeds the "
                                    f"{max_bytes // (1024 * 1024)} MiB size limit."
                                )
                            payload.extend(chunk)
                        return bytes(payload)
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
