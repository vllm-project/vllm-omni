# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import base64
import io
import zipfile

import pytest
from fastapi.responses import StreamingResponse

from vllm_omni.entrypoints.openai.protocol.images import (
    _FILE_RESPONSE_CHUNK_SIZE,
    ImageData,
    ImageGenerationResponse,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_NEWLINE_HEAVY_PAYLOAD = (b"\x89PNG\n\x1a\n" * 64) + bytes(range(256)) * 8


def _response(*payloads: bytes) -> ImageGenerationResponse:
    return ImageGenerationResponse(
        created=0,
        data=[ImageData(b64_json=base64.b64encode(payload).decode()) for payload in payloads],
        output_format="png",
        size="1x1",
    )


async def _collect(response: StreamingResponse) -> list[bytes]:
    return [bytes(chunk) async for chunk in response.body_iterator]


@pytest.mark.asyncio
async def test_single_file_response_chunks_ignore_newlines() -> None:
    payload = _NEWLINE_HEAVY_PAYLOAD
    assert b"\n" in payload

    response = _response(payload).stream_response()
    chunks = await _collect(response)

    assert len(chunks) == 1
    assert b"".join(chunks) == payload
    assert response.headers["content-length"] == str(len(payload))
    assert response.headers["content-type"] == "image/png"


@pytest.mark.asyncio
async def test_single_file_response_chunks_are_bounded() -> None:
    payload = _NEWLINE_HEAVY_PAYLOAD * 200
    expected_chunks = -(-len(payload) // _FILE_RESPONSE_CHUNK_SIZE)
    assert expected_chunks > 1

    chunks = await _collect(_response(payload).stream_response())

    assert len(chunks) == expected_chunks
    assert max(len(chunk) for chunk in chunks) <= _FILE_RESPONSE_CHUNK_SIZE
    assert b"".join(chunks) == payload


@pytest.mark.asyncio
async def test_zip_file_response_streams_whole_archive() -> None:
    payloads = (_NEWLINE_HEAVY_PAYLOAD, b"second image\nwith a newline")

    response = _response(*payloads).stream_response()
    chunks = await _collect(response)

    zip_bytes = b"".join(chunks)
    assert len(chunks) == -(-len(zip_bytes) // _FILE_RESPONSE_CHUNK_SIZE)
    assert response.headers["content-length"] == str(len(zip_bytes))
    assert response.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        assert archive.read("image_0.png") == payloads[0]
        assert archive.read("image_1.png") == payloads[1]
