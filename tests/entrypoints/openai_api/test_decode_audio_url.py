# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for decode_audio_url helper and _AudioFileMediaIO."""

import asyncio
import base64
import os

import pytest
from vllm.multimodal.media import MediaConnector

from vllm_omni.entrypoints.openai.errors import InvalidInputReferenceError
from vllm_omni.entrypoints.openai.video_api_utils import (
    _AudioFileMediaIO,
    decode_audio_url,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _run(coro):
    return asyncio.run(coro)


def _cleanup(path):
    if path and os.path.exists(path):
        os.unlink(path)


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


# ---------------------------------------------------------------------------
# _AudioFileMediaIO direct tests
# ---------------------------------------------------------------------------


class TestAudioFileMediaIO:
    """Direct tests for _AudioFileMediaIO methods."""

    def test_load_bytes_writes_temp_file(self):
        audio_bytes = b"\xff\xfb\x90\x00" * 50
        io = _AudioFileMediaIO()
        path = io.load_bytes(audio_bytes)
        try:
            assert os.path.isfile(path)
            assert path.endswith(".wav")
            with open(path, "rb") as f:
                assert f.read() == audio_bytes
        finally:
            _cleanup(path)

    def test_load_bytes_empty_raises(self):
        io = _AudioFileMediaIO()
        with pytest.raises(InvalidInputReferenceError, match="audio data is empty"):
            io.load_bytes(b"")

    def test_load_base64_mp3_suffix(self):
        audio_bytes = b"\xff\xfb\x90\x00" * 50
        b64 = base64.b64encode(audio_bytes).decode()
        io = _AudioFileMediaIO()
        path = io.load_base64("audio/mp3", b64)
        try:
            assert path.endswith(".mp3")
            with open(path, "rb") as f:
                assert f.read() == audio_bytes
        finally:
            _cleanup(path)

    def test_load_base64_mpeg_uses_mp3(self):
        io = _AudioFileMediaIO()
        path = io.load_base64("audio/mpeg", base64.b64encode(b"\x00" * 10).decode())
        try:
            assert path.endswith(".mp3")
        finally:
            _cleanup(path)

    def test_load_base64_wav_suffix(self):
        io = _AudioFileMediaIO()
        path = io.load_base64("audio/wav", base64.b64encode(b"\x00" * 10).decode())
        try:
            assert path.endswith(".wav")
        finally:
            _cleanup(path)

    def test_load_base64_flac_suffix(self):
        io = _AudioFileMediaIO()
        path = io.load_base64("audio/flac", base64.b64encode(b"\x00" * 10).decode())
        try:
            assert path.endswith(".flac")
        finally:
            _cleanup(path)

    def test_load_base64_empty_media_type_uses_wav(self):
        io = _AudioFileMediaIO()
        path = io.load_base64("", base64.b64encode(b"\x00" * 10).decode())
        try:
            assert path.endswith(".wav")
        finally:
            _cleanup(path)

    def test_load_base64_non_alnum_falls_back_to_wav(self):
        io = _AudioFileMediaIO()
        path = io.load_base64("audio/foo-bar", base64.b64encode(b"\x00" * 10).decode())
        try:
            assert path.endswith(".wav")
        finally:
            _cleanup(path)

    def test_load_base64_too_long_extension_falls_back_to_wav(self):
        io = _AudioFileMediaIO()
        path = io.load_base64("audio/verylongextension", base64.b64encode(b"\x00" * 10).decode())
        try:
            assert path.endswith(".wav")
        finally:
            _cleanup(path)

    def test_load_file_returns_path_string(self, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00" * 100)
        io = _AudioFileMediaIO()
        result = io.load_file(audio_file)
        assert result == str(audio_file)


# ---------------------------------------------------------------------------
# decode_audio_url with data URLs (exercises real MediaConnector + MediaIO)
# ---------------------------------------------------------------------------


class TestDecodeAudioUrlDataUrl:
    """Tests for base64 data URL decoding — no mocks, real code path."""

    def test_valid_mp3_data_url(self):
        audio_bytes = b"\xff\xfb\x90\x00" * 100
        b64 = base64.b64encode(audio_bytes).decode()
        url = f"data:audio/mp3;base64,{b64}"

        path = _run(decode_audio_url(url, _permissive_connector()))
        try:
            assert os.path.isfile(path)
            assert path.endswith(".mp3")
            with open(path, "rb") as f:
                assert f.read() == audio_bytes
        finally:
            _cleanup(path)

    def test_valid_wav_data_url(self):
        audio_bytes = b"RIFF" + b"\x00" * 100
        b64 = base64.b64encode(audio_bytes).decode()
        url = f"data:audio/wav;base64,{b64}"

        path = _run(decode_audio_url(url, _permissive_connector()))
        try:
            assert path.endswith(".wav")
            with open(path, "rb") as f:
                assert f.read() == audio_bytes
        finally:
            _cleanup(path)

    def test_invalid_base64_raises(self):
        url = "data:audio/mp3;base64,!!!not-valid-base64!!!"
        with pytest.raises(InvalidInputReferenceError, match="Incorrect padding"):
            _run(decode_audio_url(url, _permissive_connector()))

    def test_empty_audio_data_raises(self):
        url = "data:audio/mp3;base64,"
        with pytest.raises(InvalidInputReferenceError, match="audio data is empty"):
            _run(decode_audio_url(url, _permissive_connector()))


# ---------------------------------------------------------------------------
# decode_audio_url with file:// URIs (exercises load_file through connector)
# ---------------------------------------------------------------------------


class TestDecodeAudioUrlFileUri:
    """Tests using file:// URIs — exercises MediaConnector + load_file."""

    def test_file_uri_with_allowed_path(self, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\xff\xfb\x90\x00" * 50)
        file_uri = audio_file.as_uri()
        connector = MediaConnector(
            allowed_local_media_path=str(tmp_path),
            allowed_media_domains=None,
        )
        path = _run(decode_audio_url(file_uri, connector))
        assert path == str(audio_file)

    def test_file_uri_without_allowed_path_raises(self, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00" * 50)
        file_uri = audio_file.as_uri()
        connector = MediaConnector(
            allowed_local_media_path="",
            allowed_media_domains=None,
        )
        with pytest.raises(InvalidInputReferenceError):
            _run(decode_audio_url(file_uri, connector))

    def test_file_uri_outside_allowed_path_raises(self, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"\x00" * 50)
        file_uri = audio_file.as_uri()
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        connector = MediaConnector(
            allowed_local_media_path=str(other_dir),
            allowed_media_domains=None,
        )
        with pytest.raises(InvalidInputReferenceError):
            _run(decode_audio_url(file_uri, connector))


# ---------------------------------------------------------------------------
# decode_audio_url — invalid schemes and SSRF
# ---------------------------------------------------------------------------


class TestDecodeAudioUrlInvalid:
    """Tests for invalid URL schemes."""

    def test_local_file_path_raises(self):
        with pytest.raises(InvalidInputReferenceError, match="must be either a HTTP, data or file URL"):
            _run(decode_audio_url("/path/to/local/file.mp3", _permissive_connector()))

    def test_ftp_url_raises(self):
        with pytest.raises(InvalidInputReferenceError, match="must be either a HTTP, data or file URL"):
            _run(decode_audio_url("ftp://example.com/file.mp3", _permissive_connector()))

    def test_empty_string_raises(self):
        with pytest.raises(InvalidInputReferenceError, match="must be either a HTTP, data or file URL"):
            _run(decode_audio_url("", _permissive_connector()))


class TestDecodeAudioUrlSSRF:
    """Tests for SSRF protection via MediaConnector domain restrictions."""

    def test_restricted_domain_rejects_disallowed_url(self):
        connector = _restricted_connector(allowed_domains=["trusted.example.com"])
        with pytest.raises(InvalidInputReferenceError, match="allowed domains"):
            _run(
                decode_audio_url(
                    "https://evil.internal/admin/audio.mp3",
                    connector,
                )
            )

    def test_data_url_bypasses_domain_restriction(self):
        audio_bytes = b"\xff\xfb\x90\x00" * 50
        b64 = base64.b64encode(audio_bytes).decode()
        url = f"data:audio/wav;base64,{b64}"

        connector = _restricted_connector(allowed_domains=["trusted.example.com"])
        path = _run(decode_audio_url(url, connector))
        try:
            assert os.path.isfile(path)
        finally:
            _cleanup(path)
