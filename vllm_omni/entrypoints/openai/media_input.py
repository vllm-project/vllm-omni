# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Input-media policy around vLLM's bounded HTTP reader.

vLLM owns download accounting and configuration. This adapter owns endpoint
errors, the total deadline, and rejection of compressed final responses before
decoding. aiohttp releases redirect bodies without materializing them.
"""

from __future__ import annotations

import asyncio
import binascii
from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Literal

import aiohttp
import pybase64
from vllm import envs
from vllm.connections import (
    HTTPConnection,
    HTTPResponseSizeExceededError,
    MediaDownloadSizeExceededError,
    MediaRequestDownloadBudget,
    MediaRequestDownloadSizeExceededError,
)
from vllm.utils.mem_constants import MiB_bytes

from vllm_omni.entrypoints.openai.errors import InvalidInputReferenceError

MediaKind = Literal["image", "video", "audio"]
MEDIA_DOWNLOAD_TIMEOUT_S = 60.0


def _input_limit(kind: MediaKind) -> int | None:
    # Match vLLM: a non-positive general download limit disables that ceiling.
    limit = envs.VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB
    max_bytes = limit * MiB_bytes if limit > 0 else None
    if kind == "audio":
        audio_limit = int(envs.VLLM_MAX_AUDIO_CLIP_FILESIZE_MB * MiB_bytes)
        return audio_limit if max_bytes is None else min(max_bytes, audio_limit)
    return max_bytes


def decode_base64_media(data: str, *, kind: MediaKind, source: str) -> bytes:
    """Apply the same encoded-media budget to data URLs before decoding."""
    if not data:
        raise InvalidInputReferenceError(f"Invalid {source}: {kind} data is empty.")
    max_bytes = _input_limit(kind)
    if max_bytes is not None and len(data) > 4 * ((max_bytes + 2) // 3):
        raise InvalidInputReferenceError(f"Invalid {source}: media data exceeds the {max_bytes}-byte size limit.")
    try:
        decoded = pybase64.b64decode(data)
    except (binascii.Error, ValueError) as exc:
        raise InvalidInputReferenceError(f"Invalid {source}: {kind} data is not valid base64.") from exc
    if max_bytes is not None and len(decoded) > max_bytes:
        raise InvalidInputReferenceError(f"Invalid {source}: media data exceeds the {max_bytes}-byte size limit.")
    return decoded


class _MediaConnection(HTTPConnection):
    """Supply a request-scoped session and guard headers before vLLM reads."""

    def __init__(self, session: aiohttp.ClientSession, source: str) -> None:
        super().__init__()
        self.session = session
        self.source = source

    async def get_async_client(self) -> aiohttp.ClientSession:
        return self.session

    async def get_async_response(
        self,
        url: str,
        *,
        timeout: float | None = None,
        extra_headers: Mapping[str, str] | None = None,
        allow_redirects: bool = True,
    ) -> AbstractAsyncContextManager[aiohttp.ClientResponse]:
        response_context = await super().get_async_response(
            url,
            timeout=timeout,
            extra_headers=extra_headers,
            allow_redirects=allow_redirects,
        )

        @asynccontextmanager
        async def guarded_response() -> AsyncIterator[aiohttp.ClientResponse]:
            async with response_context as response:
                if 300 <= response.status < 400:
                    if response.headers.get("Location"):
                        raise InvalidInputReferenceError(f"Invalid {self.source}: redirect response was rejected.")
                    raise InvalidInputReferenceError(f"Invalid {self.source}: server returned HTTP {response.status}.")
                encoding = response.headers.get("Content-Encoding", "").strip().lower()
                if encoding and encoding != "identity":
                    raise InvalidInputReferenceError(f"Invalid {self.source}: compressed responses are not supported.")
                yield response

        return guarded_response()


async def download_media_bytes(
    url: str,
    *,
    kind: MediaKind,
    source: str,
    follow_redirects: bool = False,
    request_budget: MediaRequestDownloadBudget | None = None,
    timeout_s: float = MEDIA_DOWNLOAD_TIMEOUT_S,
) -> bytes:
    """Delegate streaming, byte counting and retries to vLLM's HTTPConnection."""
    try:
        async with asyncio.timeout(timeout_s):
            # Disabling decompression also protects intermediate redirects,
            # which aiohttp releases before returning the final response.
            async with aiohttp.ClientSession(
                auto_decompress=False,
                headers={"Accept-Encoding": "identity"},
                trust_env=True,
            ) as session:
                connection = _MediaConnection(session, source)
                return await connection.async_get_bytes(
                    url,
                    timeout=timeout_s,
                    allow_redirects=follow_redirects,
                    max_bytes=_input_limit(kind),
                    request_budget=request_budget,
                )
    except (
        HTTPResponseSizeExceededError,
        MediaDownloadSizeExceededError,
        MediaRequestDownloadSizeExceededError,
    ) as exc:
        raise InvalidInputReferenceError(f"Invalid {source}: {exc}") from exc
    except TimeoutError as exc:
        raise InvalidInputReferenceError(f"Invalid {source}: download exceeded the {timeout_s:g}s time limit.") from exc
    except aiohttp.ClientResponseError as exc:
        raise InvalidInputReferenceError(f"Invalid {source}: server returned HTTP {exc.status}.") from exc
    except aiohttp.ClientError as exc:
        raise InvalidInputReferenceError(f"Invalid {source}: failed to download {kind}.") from exc
