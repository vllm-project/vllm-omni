# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import io

import numpy as np
import pytest

from vllm_omni.diffusion.utils import media_utils
from vllm_omni.diffusion.utils.media_utils import (
    default_audio_codec_for_format,
    default_video_codec_for_format,
    default_video_codec_options,
    media_type_for_format,
    resolve_encoder_settings,
)
from vllm_omni.entrypoints.openai.video_api_utils import _encode_video_bytes

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_container_defaults_are_consistent() -> None:
    assert default_video_codec_for_format("mp4") == "h264"
    assert default_audio_codec_for_format("mp4") == "aac"
    assert media_type_for_format("mp4") == "video/mp4"

    assert default_video_codec_for_format("webm") == "libvpx-vp9"
    assert default_audio_codec_for_format("webm") == "libopus"
    assert media_type_for_format("webm") == "video/webm"


def test_unknown_container_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported video output format 'avi'"):
        default_video_codec_for_format("avi")


def test_default_encoder_options_preserve_the_existing_http_policy() -> None:
    assert default_video_codec_options("h264") == {"preset": "ultrafast", "threads": "0"}
    assert default_video_codec_options("h264", low_latency=True) == {
        "preset": "ultrafast",
        "threads": "0",
        "tune": "zerolatency",
    }


def test_unavailable_encoder_falls_back_with_matching_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(media_utils, "_encoder_is_usable", lambda codec: False)

    codec, options = resolve_encoder_settings(
        "h264_nvenc",
        {"preset": "p1", "tune": "ull"},
        output_format="mp4",
    )

    assert codec == "h264"
    assert options == {"preset": "ultrafast", "threads": "0"}


def test_available_incompatible_encoder_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(media_utils, "_encoder_is_usable", lambda codec: True)

    with pytest.raises(ValueError, match="incompatible with 'webm'"):
        resolve_encoder_settings("h264", output_format="webm")


def test_unavailable_incompatible_encoder_is_not_silently_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(media_utils, "_encoder_is_usable", lambda codec: False)

    with pytest.raises(ValueError, match="incompatible with 'webm'"):
        resolve_encoder_settings("h264_nvenc", output_format="webm")


def test_incompatible_fallback_codec_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(media_utils, "_encoder_is_usable", lambda codec: False)

    with pytest.raises(ValueError, match="Fallback video codec 'h264' is incompatible with 'webm'"):
        resolve_encoder_settings("libvpx", fallback="h264", output_format="webm")


@pytest.mark.parametrize(
    ("output_format", "expected_video_codec", "expected_audio_codec"),
    [("mp4", "h264", "aac"), ("webm", "vp9", "opus")],
)
def test_encode_path_uses_container_compatible_video_and_audio_codecs(
    output_format: str,
    expected_video_codec: str,
    expected_audio_codec: str,
) -> None:
    av = pytest.importorskip("av")
    frames = np.zeros((6, 32, 48, 3), dtype=np.uint8)
    audio = np.zeros(8000, dtype=np.float32)

    encoded = _encode_video_bytes(
        frames,
        fps=8,
        audio=audio,
        audio_sample_rate=16000,
        output_format=output_format,
    )

    with av.open(io.BytesIO(encoded)) as container:
        video_stream = container.streams.video[0]
        audio_stream = container.streams.audio[0]
        assert video_stream.codec_context.name == expected_video_codec
        assert audio_stream.codec_context.name == expected_audio_codec
        assert video_stream.codec_context.width == 48
        assert video_stream.codec_context.height == 32
