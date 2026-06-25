# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for audiox materialize_media_source and _TempFileMediaIO."""

import base64
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_omni.transformers_utils.processors.audiox import (
    _TempFileMediaIO,
    materialize_media_source,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _cleanup(path):
    if path and os.path.exists(path):
        os.unlink(path)


def _fake_forward_context(allowed_domains=None, allowed_local_media_path=""):
    """Return a fake ForwardContext with model_config for SSRF testing."""
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                allowed_local_media_path=allowed_local_media_path,
                allowed_media_domains=allowed_domains,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# _TempFileMediaIO direct tests
# ---------------------------------------------------------------------------


class TestTempFileMediaIO:
    """Direct tests for _TempFileMediaIO methods."""

    def test_load_bytes_writes_temp_file(self):
        data = b"\x00\x01\x02" * 50
        io = _TempFileMediaIO()
        path = io.load_bytes(data)
        try:
            assert os.path.isfile(path)
            assert path.endswith(".bin")
            with open(path, "rb") as f:
                assert f.read() == data
        finally:
            _cleanup(path)

    def test_load_base64_writes_temp_file(self):
        data = b"\xff" * 100
        b64 = base64.b64encode(data).decode()
        io = _TempFileMediaIO()
        path = io.load_base64("video/mp4", b64)
        try:
            assert os.path.isfile(path)
            with open(path, "rb") as f:
                assert f.read() == data
        finally:
            _cleanup(path)

    def test_load_file_returns_path_string(self, tmp_path):
        media_file = tmp_path / "video.mp4"
        media_file.write_bytes(b"\x00" * 50)
        io = _TempFileMediaIO()
        result = io.load_file(media_file)
        assert result == str(media_file)


# ---------------------------------------------------------------------------
# materialize_media_source
# ---------------------------------------------------------------------------


class TestMaterializeMediaSource:
    """Tests for materialize_media_source — exercises real MediaConnector."""

    def test_data_url_with_no_vllm_config(self):
        data = b"\x00" * 50
        b64 = base64.b64encode(data).decode()
        url = f"data:audio/wav;base64,{b64}"
        ctx = SimpleNamespace(vllm_config=None)

        with patch(
            "vllm_omni.diffusion.forward_context.get_forward_context",
            return_value=ctx,
        ):
            path = materialize_media_source(url)
        try:
            assert os.path.isfile(path)
        finally:
            _cleanup(path)

    def test_data_url_materializes_to_temp_file(self):
        data = b"\x00\x01\x02" * 50
        b64 = base64.b64encode(data).decode()
        url = f"data:audio/wav;base64,{b64}"

        with patch(
            "vllm_omni.diffusion.forward_context.get_forward_context",
            return_value=_fake_forward_context(),
        ):
            path = materialize_media_source(url)
        try:
            assert os.path.isfile(path)
            with open(path, "rb") as f:
                assert f.read() == data
        finally:
            _cleanup(path)

    def test_local_path_with_allowed_path(self, tmp_path):
        local_file = tmp_path / "media.bin"
        local_file.write_bytes(b"\x00" * 50)
        ctx = _fake_forward_context(allowed_local_media_path=str(tmp_path))
        with patch(
            "vllm_omni.diffusion.forward_context.get_forward_context",
            return_value=ctx,
        ):
            path = materialize_media_source(str(local_file))
        assert path == str(local_file)

    def test_local_path_without_allowed_path_raises(self, tmp_path):
        local_file = tmp_path / "media.bin"
        local_file.write_bytes(b"\x00" * 50)
        ctx = _fake_forward_context(allowed_local_media_path="")
        with patch(
            "vllm_omni.diffusion.forward_context.get_forward_context",
            return_value=ctx,
        ):
            with pytest.raises(RuntimeError, match="allowed-local-media-path"):
                materialize_media_source(str(local_file))

    def test_http_url_rejected_by_domain_filter(self):
        ctx = _fake_forward_context(allowed_domains=["trusted.example.com"])
        with patch(
            "vllm_omni.diffusion.forward_context.get_forward_context",
            return_value=ctx,
        ):
            with pytest.raises(ValueError, match="allowed domains"):
                materialize_media_source("https://evil.internal/media.bin")

    def test_file_uri_with_allowed_path(self, tmp_path):
        media_file = tmp_path / "audio.wav"
        media_file.write_bytes(b"\x00" * 50)
        file_uri = media_file.as_uri()

        ctx = _fake_forward_context(allowed_local_media_path=str(tmp_path))
        with patch(
            "vllm_omni.diffusion.forward_context.get_forward_context",
            return_value=ctx,
        ):
            path = materialize_media_source(file_uri)
        assert path == str(media_file)

    def test_file_uri_without_allowed_path_raises(self, tmp_path):
        media_file = tmp_path / "audio.wav"
        media_file.write_bytes(b"\x00" * 50)
        file_uri = media_file.as_uri()

        ctx = _fake_forward_context(allowed_local_media_path="")
        with patch(
            "vllm_omni.diffusion.forward_context.get_forward_context",
            return_value=ctx,
        ):
            with pytest.raises(RuntimeError, match="allowed-local-media-path"):
                materialize_media_source(file_uri)
