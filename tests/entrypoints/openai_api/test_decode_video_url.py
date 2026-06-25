# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for decode_video_url helper and _VideoFramesMediaIO."""

import asyncio
import base64
import io

import pytest
from PIL import Image
from vllm.multimodal.media import MediaConnector

from vllm_omni.entrypoints.openai.errors import InvalidInputReferenceError
from vllm_omni.entrypoints.openai.video_api_utils import (
    VideoFrames,
    _VideoFramesMediaIO,
    decode_video_url,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _run(coro):
    return asyncio.run(coro)


def _permissive_connector():
    return MediaConnector(
        allowed_local_media_path="",
        allowed_media_domains=None,
    )


def _restricted_connector(allowed_domains=None):
    return MediaConnector(
        allowed_local_media_path="",
        allowed_media_domains=allowed_domains or [],
    )


def _make_mp4_bytes(width=4, height=4, num_frames=2):
    """Create a minimal valid MP4 video using PyAV."""
    import av

    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"preset": "ultrafast"}
        for i in range(num_frames):
            img = Image.new("RGB", (width, height), color=(i * 40, 128, 200))
            frame = av.VideoFrame.from_image(img)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# _VideoFramesMediaIO direct tests
# ---------------------------------------------------------------------------


class TestVideoFramesMediaIO:
    """Direct tests for _VideoFramesMediaIO methods."""

    def test_load_bytes_decodes_video(self):
        video_bytes = _make_mp4_bytes(num_frames=3)
        media_io = _VideoFramesMediaIO()
        result = media_io.load_bytes(video_bytes)
        assert isinstance(result, VideoFrames)
        assert len(result) >= 1
        assert all(isinstance(f, Image.Image) for f in result)

    def test_load_bytes_respects_max_frames(self):
        video_bytes = _make_mp4_bytes(num_frames=5)
        media_io = _VideoFramesMediaIO(max_frames=2)
        result = media_io.load_bytes(video_bytes)
        assert len(result) <= 2

    def test_load_bytes_invalid_data_raises(self):
        media_io = _VideoFramesMediaIO()
        with pytest.raises(InvalidInputReferenceError, match="not a valid video"):
            media_io.load_bytes(b"not a video")

    def test_load_base64_decodes_video(self):
        video_bytes = _make_mp4_bytes()
        b64 = base64.b64encode(video_bytes).decode()
        media_io = _VideoFramesMediaIO()
        result = media_io.load_base64("video/mp4", b64)
        assert isinstance(result, VideoFrames)
        assert len(result) >= 1

    def test_load_file_decodes_video(self, tmp_path):
        video_bytes = _make_mp4_bytes()
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(video_bytes)
        media_io = _VideoFramesMediaIO()
        result = media_io.load_file(video_file)
        assert isinstance(result, VideoFrames)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# decode_video_url with data URLs (real MediaConnector + MediaIO)
# ---------------------------------------------------------------------------


class TestDecodeVideoUrlDataUrl:
    """Tests for data URL decoding — no mocks, real code path."""

    def test_valid_video_data_url(self):
        video_bytes = _make_mp4_bytes()
        b64 = base64.b64encode(video_bytes).decode()
        url = f"data:video/mp4;base64,{b64}"

        result = _run(decode_video_url(url, _permissive_connector()))
        assert isinstance(result, VideoFrames)
        assert len(result) >= 1
        assert all(isinstance(f, Image.Image) for f in result)

    def test_invalid_video_data_url_raises(self):
        b64 = base64.b64encode(b"not a video").decode()
        url = f"data:video/mp4;base64,{b64}"
        with pytest.raises(InvalidInputReferenceError, match="not a valid video"):
            _run(decode_video_url(url, _permissive_connector()))


# ---------------------------------------------------------------------------
# decode_video_url with file:// URIs (exercises load_file through connector)
# ---------------------------------------------------------------------------


class TestDecodeVideoUrlFileUri:
    """Tests using file:// URIs — exercises MediaConnector + load_file."""

    def test_file_uri_with_allowed_path(self, tmp_path):
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(_make_mp4_bytes())
        file_uri = video_file.as_uri()
        connector = MediaConnector(
            allowed_local_media_path=str(tmp_path),
            allowed_media_domains=None,
        )
        result = _run(decode_video_url(file_uri, connector))
        assert isinstance(result, VideoFrames)
        assert len(result) >= 1

    def test_file_uri_without_allowed_path_raises(self, tmp_path):
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(_make_mp4_bytes())
        file_uri = video_file.as_uri()
        connector = MediaConnector(
            allowed_local_media_path="",
            allowed_media_domains=None,
        )
        with pytest.raises(InvalidInputReferenceError):
            _run(decode_video_url(file_uri, connector))


# ---------------------------------------------------------------------------
# decode_video_url — invalid schemes and SSRF
# ---------------------------------------------------------------------------


class TestDecodeVideoUrlInvalid:
    """Tests for invalid URL schemes."""

    def test_local_file_path_raises(self):
        with pytest.raises(InvalidInputReferenceError, match="must be either a HTTP, data or file URL"):
            _run(decode_video_url("/path/to/local/file.mp4", _permissive_connector()))

    def test_ftp_url_raises(self):
        with pytest.raises(InvalidInputReferenceError, match="must be either a HTTP, data or file URL"):
            _run(decode_video_url("ftp://example.com/file.mp4", _permissive_connector()))

    def test_empty_string_raises(self):
        with pytest.raises(InvalidInputReferenceError, match="must be either a HTTP, data or file URL"):
            _run(decode_video_url("", _permissive_connector()))


class TestDecodeVideoUrlSSRF:
    """Tests for SSRF protection via MediaConnector domain restrictions."""

    def test_restricted_domain_rejects_disallowed_url(self):
        connector = _restricted_connector(
            allowed_domains=["trusted.example.com"],
        )
        with pytest.raises(InvalidInputReferenceError, match="allowed domains"):
            _run(
                decode_video_url(
                    "https://evil.internal/admin/video.mp4",
                    connector,
                )
            )

    def test_data_url_bypasses_domain_restriction(self):
        video_bytes = _make_mp4_bytes()
        b64 = base64.b64encode(video_bytes).decode()
        url = f"data:video/mp4;base64,{b64}"

        connector = _restricted_connector(
            allowed_domains=["trusted.example.com"],
        )
        result = _run(decode_video_url(url, connector))
        assert isinstance(result, VideoFrames)
        assert len(result) >= 1
