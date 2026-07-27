# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for decode_audio_url helper."""

import asyncio
import base64
import os

import httpx
import pytest

from vllm_omni.entrypoints.openai.errors import InvalidInputReferenceError
from vllm_omni.entrypoints.openai.video_api_utils import decode_audio_url

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _run(coro):
    return asyncio.run(coro)


def _cleanup(path):
    if path and os.path.exists(path):
        os.unlink(path)


class TestDecodeAudioUrlDataUrl:
    """Tests for base64 data URL decoding."""

    def test_valid_mp3_data_url(self):
        audio_bytes = b"\xff\xfb\x90\x00" * 100
        b64 = base64.b64encode(audio_bytes).decode()
        url = f"data:audio/mp3;base64,{b64}"

        path = _run(decode_audio_url(url))
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

        path = _run(decode_audio_url(url))
        try:
            assert path.endswith(".wav")
            with open(path, "rb") as f:
                assert f.read() == audio_bytes
        finally:
            _cleanup(path)

    def test_valid_mpeg_data_url_uses_mp3_suffix(self):
        audio_bytes = b"\xff\xfb\x90\x00" * 50
        b64 = base64.b64encode(audio_bytes).decode()
        url = f"data:audio/mpeg;base64,{b64}"

        path = _run(decode_audio_url(url))
        try:
            assert path.endswith(".mp3")
        finally:
            _cleanup(path)

    def test_valid_ogg_data_url(self):
        audio_bytes = b"OggS" + b"\x00" * 100
        b64 = base64.b64encode(audio_bytes).decode()
        url = f"data:audio/ogg;base64,{b64}"

        path = _run(decode_audio_url(url))
        try:
            assert path.endswith(".ogg")
        finally:
            _cleanup(path)

    def test_invalid_base64_raises(self):
        url = "data:audio/mp3;base64,!!!not-valid-base64!!!"

        with pytest.raises(InvalidInputReferenceError, match="not valid base64"):
            _run(decode_audio_url(url))

    def test_empty_audio_data_raises(self):
        url = "data:audio/mp3;base64,"

        with pytest.raises(InvalidInputReferenceError, match="audio data is empty"):
            _run(decode_audio_url(url))


class TestDecodeAudioUrlSuffixSanitization:
    """Tests for MIME extension sanitization (path traversal prevention)."""

    def test_path_traversal_in_mime_is_neutralized(self):
        audio_bytes = b"\x00" * 100
        b64 = base64.b64encode(audio_bytes).decode()
        url = f"data:audio/../..;base64,{b64}"

        path = _run(decode_audio_url(url))
        try:
            # ".." contains non-alnum dots, so falls back to .wav
            assert path.endswith(".wav")
            assert os.path.dirname(path) == "/tmp"
        finally:
            _cleanup(path)

    def test_non_alnum_extension_falls_back_to_wav(self):
        audio_bytes = b"\x00" * 100
        b64 = base64.b64encode(audio_bytes).decode()
        url = f"data:audio/foo-bar;base64,{b64}"

        path = _run(decode_audio_url(url))
        try:
            assert path.endswith(".wav")
        finally:
            _cleanup(path)

    def test_too_long_extension_falls_back_to_wav(self):
        audio_bytes = b"\x00" * 100
        b64 = base64.b64encode(audio_bytes).decode()
        url = f"data:audio/verylongextension;base64,{b64}"

        path = _run(decode_audio_url(url))
        try:
            assert path.endswith(".wav")
        finally:
            _cleanup(path)

    def test_valid_short_alnum_extension_is_used(self):
        audio_bytes = b"\x00" * 100
        b64 = base64.b64encode(audio_bytes).decode()
        url = f"data:audio/flac;base64,{b64}"

        path = _run(decode_audio_url(url))
        try:
            assert path.endswith(".flac")
        finally:
            _cleanup(path)


class TestDecodeAudioUrlHttp:
    """Tests for HTTP URL decoding."""

    def test_valid_http_url(self, monkeypatch):
        audio_bytes = b"\xff\xfb\x90\x00" * 50
        fake_resp = httpx.Response(200, content=audio_bytes)

        async def _mock_get(self, url):
            return fake_resp

        monkeypatch.setattr(httpx.AsyncClient, "get", _mock_get)
        monkeypatch.setattr(httpx.Response, "raise_for_status", lambda self: None)

        path = _run(decode_audio_url("https://example.com/audio.mp3"))
        try:
            assert os.path.isfile(path)
            assert path.endswith(".wav")
            with open(path, "rb") as f:
                assert f.read() == audio_bytes
        finally:
            _cleanup(path)

    def test_http_error_raises(self, monkeypatch):
        async def _mock_get(self, url):
            raise httpx.HTTPStatusError("Not Found", request=httpx.Request("GET", url), response=httpx.Response(404))

        monkeypatch.setattr(httpx.AsyncClient, "get", _mock_get)

        with pytest.raises(InvalidInputReferenceError, match="failed to download audio"):
            _run(decode_audio_url("https://example.com/missing.mp3"))


class TestDecodeAudioUrlInvalid:
    """Tests for invalid URL schemes."""

    def test_local_file_path_raises(self):
        with pytest.raises(InvalidInputReferenceError, match="must be an http.*URL or data URL"):
            _run(decode_audio_url("/path/to/local/file.mp3"))

    def test_ftp_url_raises(self):
        with pytest.raises(InvalidInputReferenceError, match="must be an http.*URL or data URL"):
            _run(decode_audio_url("ftp://example.com/file.mp3"))

    def test_empty_string_raises(self):
        with pytest.raises(InvalidInputReferenceError, match="must be an http.*URL or data URL"):
            _run(decode_audio_url(""))
