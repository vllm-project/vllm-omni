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


@pytest.mark.parametrize("requested_codec", ["h264_nvenc", "hevc_nvenc"])
def test_encoder_policy_does_not_probe_the_resolving_process(mocker, requested_codec: str) -> None:
    context = mocker.patch.object(media_utils.av.codec, "CodecContext")
    context.create.side_effect = ValueError("encoder unavailable in API process")

    codec, options = resolve_encoder_settings(
        requested_codec,
        {"preset": "p1", "tune": "ull"},
        output_format="mp4",
    )

    assert codec == requested_codec
    assert options == {"preset": "p1", "tune": "ull"}
    context.create.assert_not_called()


@pytest.mark.parametrize("codec", ["h264", "h264_nvenc"])
def test_incompatible_encoder_is_rejected(codec: str) -> None:
    with pytest.raises(ValueError, match="incompatible with 'webm'"):
        resolve_encoder_settings(codec, output_format="webm")


def test_incompatible_fallback_codec_is_rejected() -> None:
    with pytest.raises(ValueError, match="Fallback video codec 'h264' is incompatible with 'webm'"):
        resolve_encoder_settings("libvpx", fallback="h264", output_format="webm")


@pytest.mark.parametrize("encode_path", ["array", "iterator", "chunked"])
def test_unavailable_encoder_failure_is_propagated_from_encoding_process(mocker, encode_path: str) -> None:
    codec, options = resolve_encoder_settings("h264_nvenc", {"preset": "p1", "tune": "ull"})
    container = mocker.MagicMock()
    container.__enter__.return_value = container
    container.add_stream.side_effect = ValueError("requested encoder cannot be opened")
    mocker.patch.object(media_utils.av, "open", return_value=container)
    frames = np.zeros((2, 32, 48, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="requested encoder cannot be opened"):
        if encode_path == "array":
            media_utils.mux_video_audio_bytes(frames, video_codec=codec, video_codec_options=options)
        elif encode_path == "iterator":
            media_utils.mux_av_video_audio_bytes(
                [], width=48, height=32, video_codec=codec, video_codec_options=options
            )
        else:
            encoder = media_utils.ChunkedMP4Encoder(
                width=48, height=32, fps=8, video_codec=codec, video_codec_options=options
            )
            encoder.finish()

    assert container.add_stream.call_args.args[0] == "h264_nvenc"
    container.add_stream.assert_called_once()


@pytest.mark.parametrize("enable_borrowed_frames", [False, True])
@pytest.mark.parametrize(
    ("output_format", "expected_video_codec", "expected_audio_codec"),
    [("mp4", "h264", "aac"), ("webm", "vp9", "opus")],
)
def test_encode_path_uses_container_compatible_video_and_audio_codecs(
    output_format: str,
    expected_video_codec: str,
    expected_audio_codec: str,
    enable_borrowed_frames: bool,
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
        enable_borrowed_frames=enable_borrowed_frames,
    )

    with av.open(io.BytesIO(encoded)) as container:
        video_stream = container.streams.video[0]
        audio_stream = container.streams.audio[0]
        assert video_stream.codec_context.name == expected_video_codec
        assert audio_stream.codec_context.name == expected_audio_codec
        assert video_stream.codec_context.width == 48
        assert video_stream.codec_context.height == 32


def test_borrowed_frame_encoder_preserves_explicit_codec_and_options(mocker) -> None:
    muxer = mocker.spy(media_utils, "mux_av_video_audio_bytes")
    frames = np.zeros((2, 32, 48, 3), dtype=np.uint8)
    options = {"deadline": "realtime", "cpu-used": "8"}
    encoded = _encode_video_bytes(
        frames,
        fps=8,
        output_format="webm",
        video_codec="libvpx",
        video_codec_options=options,
        enable_borrowed_frames=True,
    )

    assert muxer.call_args.kwargs["video_codec"] == "libvpx"
    assert muxer.call_args.kwargs["video_codec_options"] == options
    assert muxer.call_args.kwargs["output_format"] == "webm"
    with media_utils.av.open(io.BytesIO(encoded)) as container:
        assert container.streams.video[0].codec_context.name == "vp8"
        assert len(list(container.decode(video=0))) == 2


@pytest.mark.parametrize("mux_path", ["array", "iterator"])
def test_webm_opus_resamples_unsupported_44100_audio(mux_path: str) -> None:
    av = pytest.importorskip("av")
    frames = np.zeros((6, 32, 48, 3), dtype=np.uint8)
    audio = np.zeros(4410, dtype=np.float32)

    if mux_path == "array":
        encoded = media_utils.mux_video_audio_bytes(
            frames,
            audio,
            fps=8,
            audio_sample_rate=44100,
            output_format="webm",
        )
    else:
        encoded = media_utils.mux_av_video_audio_bytes(
            (av.VideoFrame.from_ndarray(frame, format="rgb24") for frame in frames),
            width=48,
            height=32,
            audio_waveform=audio,
            fps=8,
            audio_sample_rate=44100,
            output_format="webm",
        )

    with av.open(io.BytesIO(encoded)) as container:
        audio_stream = container.streams.audio[0]
        decoded_audio = list(container.decode(audio=0))

    assert audio_stream.codec_context.name == "opus"
    assert audio_stream.rate == 48000
    assert decoded_audio
    assert {frame.sample_rate for frame in decoded_audio} == {48000}
    assert sum(frame.samples for frame in decoded_audio) == 4800
