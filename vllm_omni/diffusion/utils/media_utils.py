# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Video/audio muxing utilities using PyAV (no ffmpeg binary dependency)."""

from __future__ import annotations

import functools
import io
from collections.abc import Iterable
from fractions import Fraction
from typing import Any, cast

import av
import numpy as np
from vllm.logger import init_logger

logger = init_logger(__name__)

DEFAULT_VIDEO_CODEC = "h264"
DEFAULT_OUTPUT_FORMAT = "mp4"

_FORMAT_DEFAULTS: dict[str, dict[str, str]] = {
    "mp4": {"video_codec": "h264", "audio_codec": "aac", "media_type": "video/mp4"},
    "webm": {"video_codec": "libvpx-vp9", "audio_codec": "libopus", "media_type": "video/webm"},
}

_FORMAT_VIDEO_CODECS: dict[str, frozenset[str]] = {
    "mp4": frozenset(
        {"h264", "libx264", "hevc", "libx265", "h264_nvenc", "hevc_nvenc", "av1", "libaom-av1", "libsvtav1"}
    ),
    "webm": frozenset({"vp8", "libvpx", "vp9", "libvpx-vp9", "av1", "libaom-av1", "libsvtav1"}),
}

_FAST_CODEC_OPTIONS: dict[str, dict[str, str]] = {
    "h264": {"preset": "ultrafast", "threads": "0"},
    "libx264": {"preset": "ultrafast", "threads": "0"},
    "hevc": {"preset": "ultrafast", "threads": "0"},
    "libx265": {"preset": "ultrafast", "threads": "0"},
    "h264_nvenc": {"preset": "p1", "tune": "ull"},
    "hevc_nvenc": {"preset": "p1", "tune": "ull"},
}

_LOW_LATENCY_OPTIONS: dict[str, dict[str, str]] = {
    "h264": {"tune": "zerolatency"},
    "libx264": {"tune": "zerolatency"},
    "hevc": {"tune": "zerolatency"},
    "libx265": {"tune": "zerolatency"},
}


def _format_defaults(output_format: str | None) -> dict[str, str]:
    resolved_format = output_format or DEFAULT_OUTPUT_FORMAT
    try:
        return _FORMAT_DEFAULTS[resolved_format]
    except KeyError:
        raise ValueError(
            f"Unsupported video output format {resolved_format!r}; expected one of {sorted(_FORMAT_DEFAULTS)}"
        ) from None


def default_video_codec_for_format(output_format: str | None) -> str:
    return _format_defaults(output_format)["video_codec"]


def default_audio_codec_for_format(output_format: str | None) -> str:
    return _format_defaults(output_format)["audio_codec"]


def media_type_for_format(output_format: str | None) -> str:
    return _format_defaults(output_format)["media_type"]


@functools.cache
def _encoder_is_usable(codec: str) -> bool:
    try:
        context = av.codec.CodecContext.create(codec, "w")
        context.width = 64
        context.height = 64
        context.pix_fmt = "yuv420p"
        context.open()
    except (ValueError, av.error.FFmpegError) as exc:
        logger.debug("Encoder %s is not usable on this host: %s", codec, exc)
        return False
    return True


def default_video_codec_options(codec: str, *, low_latency: bool = False) -> dict[str, str]:
    options = dict(_FAST_CODEC_OPTIONS.get(codec, {}))
    if low_latency:
        options.update(_LOW_LATENCY_OPTIONS.get(codec, {}))
    return options


def resolve_encoder_settings(
    codec: str | None,
    codec_options: dict[str, str] | None = None,
    *,
    low_latency: bool = False,
    fallback: str | None = None,
    output_format: str | None = None,
) -> tuple[str, dict[str, str]]:
    resolved_format = output_format or DEFAULT_OUTPUT_FORMAT
    _format_defaults(resolved_format)
    fallback_codec = fallback or default_video_codec_for_format(resolved_format)
    requested_codec = codec or fallback_codec

    compatible_codecs = _FORMAT_VIDEO_CODECS[resolved_format]
    if fallback_codec not in compatible_codecs:
        raise ValueError(
            f"Fallback video codec {fallback_codec!r} is incompatible with {resolved_format!r}; "
            f"expected one of {sorted(compatible_codecs)}"
        )
    if requested_codec not in compatible_codecs:
        raise ValueError(
            f"Video codec {requested_codec!r} is incompatible with {resolved_format!r}; "
            f"expected one of {sorted(compatible_codecs)}"
        )

    if requested_codec != fallback_codec and not _encoder_is_usable(requested_codec):
        logger.warning(
            "Video encoder %r cannot be opened on this host; falling back to %r.",
            requested_codec,
            fallback_codec,
        )
        requested_codec = fallback_codec
        codec_options = None
    if codec_options:
        return requested_codec, dict(codec_options)
    return requested_codec, default_video_codec_options(requested_codec, low_latency=low_latency)


class FragmentedMP4Muxer:
    """Incrementally mux video frames into one fragmented MP4 byte stream."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: float = 25.0,
        video_codec: str = "h264",
        crf: str = "18",
        video_codec_options: dict[str, str] | None = None,
    ) -> None:
        self._buf = io.BytesIO()
        self._closed = False
        self._container = av.open(
            self._buf,
            mode="w",
            format="mp4",
            options={"movflags": "+frag_every_frame+empty_moov+default_base_moof"},
        )

        self._stream: av.VideoStream = cast(
            av.VideoStream,
            self._container.add_stream(video_codec, rate=Fraction(fps).limit_denominator(10000)),
        )
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"

        options: dict[str, object] = {"crf": str(crf)}
        if video_codec_options:
            options.update(video_codec_options)
        self._stream.options = options

        try:
            self._stream.codec_context.max_b_frames = 0
        except AttributeError:
            pass

    def mux_video_frames(self, video_frames: np.ndarray) -> bytes:
        """Mux a batch of ``uint8`` RGB frames and return newly written MP4 bytes."""
        if self._closed:
            raise RuntimeError("Cannot mux frames after FragmentedMP4Muxer.close().")
        if video_frames.ndim != 4 or video_frames.shape[-1] != 3:
            raise ValueError("video_frames must have shape (T, H, W, 3).")
        if video_frames.dtype != np.uint8:
            raise ValueError("video_frames must be uint8.")
        if video_frames.shape[1] != self._stream.height or video_frames.shape[2] != self._stream.width:
            raise ValueError("All fragmented MP4 chunks in a session must use the same frame size.")

        for frame_data in video_frames:
            frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
            for packet in self._stream.encode(frame):
                self._container.mux(packet)
        return self._read_new_bytes()

    def close(self) -> bytes:
        """Flush delayed encoder packets, close the container, and return final bytes."""
        if self._closed:
            return b""
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()
        self._closed = True
        return self._read_new_bytes()

    def _read_new_bytes(self) -> bytes:
        """Return newly muxed bytes in the current video container,
        then clear the buffer to prepare for the next chunk."""
        chunk = self._buf.getvalue()
        self._buf.seek(0)
        self._buf.truncate()
        return chunk


def finalize_streaming_video_bytes(
    video_bytes: bytes,
    *,
    input_format: str,
    fps: float = 25.0,
    video_codec_options: dict[str, str] | None = None,
) -> bytes:
    """Convert streamed video bytes into a progressive MP4 for local playback."""
    if not video_bytes:
        return video_bytes

    normalized_format = input_format.lower()
    if normalized_format == "m4s":
        demux_format = "mp4"
    else:
        raise ValueError(f"Unsupported streaming video format: {input_format}")

    try:
        with cast(Any, av.open(io.BytesIO(video_bytes), format=demux_format)) as container:
            stream = container.streams.video[0]
            frame_arrays = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
    except Exception:
        return video_bytes

    if not frame_arrays:
        return video_bytes

    frames_u8 = np.ascontiguousarray(np.stack(frame_arrays, axis=0), dtype=np.uint8)
    return mux_video_audio_bytes(
        frames_u8,
        None,
        fps=float(fps),
        video_codec_options=video_codec_options,
    )


def mux_video_audio_bytes(
    video_frames: np.ndarray,
    audio_waveform: np.ndarray | None = None,
    *,
    fps: float = 25.0,
    audio_sample_rate: int = 44100,
    video_codec: str | None = None,
    audio_codec: str | None = None,
    crf: str = "18",
    video_codec_options: dict[str, str] | None = None,
    output_format: str | None = None,
) -> bytes:
    """Mux video frames and optional audio waveform into MP4 bytes.

    Args:
        video_frames: uint8 array of shape ``(T, H, W, 3)`` (RGB).
        audio_waveform: float32 array – mono ``(N,)`` or ``(N, C)`` / ``(C, N)``.
        fps: Video frame rate.
        audio_sample_rate: Audio sample rate in Hz.
        video_codec: Video codec name.
        audio_codec: Audio codec name.
        crf: Constant rate factor for the video encoder.

    Returns:
        Raw MP4 bytes ready to be written to disk or streamed.
    """
    container_format = output_format or DEFAULT_OUTPUT_FORMAT
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format=container_format)

    v_stream = cast(
        av.VideoStream,
        container.add_stream(
            video_codec or default_video_codec_for_format(container_format),
            rate=Fraction(fps).limit_denominator(10000),
        ),
    )
    v_stream.width = video_frames.shape[2]
    v_stream.height = video_frames.shape[1]
    v_stream.pix_fmt = "yuv420p"

    options: dict[str, object] = {"crf": str(crf)}
    if video_codec_options:
        options.update(video_codec_options)
    v_stream.options = options

    a_stream: av.AudioStream | None = None
    samples: np.ndarray | None = None
    layout: str | None = None
    if audio_waveform is not None:
        samples = audio_waveform.astype(np.float32)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        elif samples.ndim == 2 and samples.shape[0] > samples.shape[1]:
            samples = np.ascontiguousarray(samples.T)
        num_channels = samples.shape[0]
        layout = "stereo" if num_channels >= 2 else "mono"
        a_stream = cast(
            av.AudioStream,
            container.add_stream(
                audio_codec or default_audio_codec_for_format(container_format),
                rate=audio_sample_rate,
            ),
        )
        a_stream.layout = layout

    for frame_data in video_frames:
        frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
        for packet in v_stream.encode(frame):
            container.mux(packet)
    for packet in v_stream.encode():
        container.mux(packet)

    if a_stream is not None and audio_waveform is not None:
        if samples is None or layout is None:
            raise ValueError("Audio samples were not prepared for muxing.")
        audio_frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout=layout)
        audio_frame.sample_rate = audio_sample_rate
        # AAC has a one-frame encoder delay. Mark the input waveform as
        # starting at t=0 so the MP4 muxer writes the corresponding negative
        # priming timestamp instead of exposing the delay as leading silence.
        audio_frame.pts = 0
        audio_frame.time_base = Fraction(1, audio_sample_rate)
        for packet in a_stream.encode(audio_frame):
            container.mux(packet)
        for packet in a_stream.encode():
            container.mux(packet)

    container.close()
    return buf.getvalue()


def mux_av_video_audio_bytes(
    video_frames: Iterable[av.VideoFrame],
    width: int,
    height: int,
    audio_waveform: np.ndarray | None = None,
    *,
    fps: float = 25.0,
    audio_sample_rate: int | None = None,
    video_codec: str | None = None,
    audio_codec: str | None = None,
    crf: str = "18",
    video_codec_options: dict[str, str] | None = None,
    output_format: str | None = None,
) -> bytes:
    """Mux preconstructed video frames and optional audio into container bytes."""
    container_format = output_format or DEFAULT_OUTPUT_FORMAT
    buf = io.BytesIO()
    with cast(Any, av.open(buf, mode="w", format=container_format)) as container:
        v_stream = cast(
            av.VideoStream,
            container.add_stream(
                video_codec or default_video_codec_for_format(container_format),
                rate=Fraction(fps).limit_denominator(10000),
            ),
        )
        v_stream.width = width
        v_stream.height = height
        v_stream.pix_fmt = "yuv420p"

        options: dict[str, object] = {"crf": str(crf)}
        if video_codec_options:
            options.update(video_codec_options)
        v_stream.options = options

        a_stream: av.AudioStream | None = None
        samples: np.ndarray | None = None
        layout: str | None = None
        if audio_waveform is not None:
            effective_audio_sample_rate = 44100 if audio_sample_rate is None else audio_sample_rate
            samples = audio_waveform.astype(np.float32)
            if samples.ndim == 1:
                samples = samples.reshape(1, -1)
            elif samples.ndim == 2 and samples.shape[0] > samples.shape[1]:
                samples = np.ascontiguousarray(samples.T)
            num_channels = samples.shape[0]
            layout = "stereo" if num_channels >= 2 else "mono"
            a_stream = cast(
                av.AudioStream,
                container.add_stream(
                    audio_codec or default_audio_codec_for_format(container_format),
                    rate=effective_audio_sample_rate,
                ),
            )
            a_stream.layout = layout

        for frame in video_frames:
            for packet in v_stream.encode(frame):
                container.mux(packet)
        for packet in v_stream.encode():
            container.mux(packet)

        if a_stream is not None and audio_waveform is not None:
            if samples is None or layout is None:
                raise ValueError("Audio samples were not prepared for muxing.")
            audio_frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout=layout)
            audio_frame.sample_rate = effective_audio_sample_rate
            # AAC has a one-frame encoder delay. Mark the input waveform as
            # starting at t=0 so the MP4 muxer writes the corresponding negative
            # priming timestamp instead of exposing the delay as leading silence.
            audio_frame.pts = 0
            audio_frame.time_base = Fraction(1, effective_audio_sample_rate)
            for packet in a_stream.encode(audio_frame):
                container.mux(packet)
            for packet in a_stream.encode():
                container.mux(packet)

    return buf.getvalue()
