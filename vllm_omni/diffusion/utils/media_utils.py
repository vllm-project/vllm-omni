# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Video/audio muxing utilities.

CPU encoding uses PyAV (no ffmpeg binary dependency); the *_nvenc codecs are
encoded on GPU through torchcodec's NVENC backend.
"""

from __future__ import annotations

import io
from collections.abc import Iterable
from fractions import Fraction
from typing import Any, cast

import av
import numpy as np

# Video codecs handled by torchcodec's CUDA (NVENC) encoder instead of PyAV.
NVENC_VIDEO_CODECS = frozenset(("h264_nvenc", "hevc_nvenc", "av1_nvenc"))


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
    video_codec: str = "h264",
    audio_codec: str = "aac",
    crf: str = "18",
    video_codec_options: dict[str, str] | None = None,
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
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp4")

    v_stream = cast(av.VideoStream, container.add_stream(video_codec, rate=Fraction(fps).limit_denominator(10000)))
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
        a_stream = cast(av.AudioStream, container.add_stream(audio_codec, rate=audio_sample_rate))
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
    video_codec: str = "h264",
    audio_codec: str = "aac",
    crf: str = "18",
    video_codec_options: dict[str, str] | None = None,
) -> bytes:
    """Mux preconstructed video frames and optional audio into MP4 bytes."""
    buf = io.BytesIO()
    with cast(Any, av.open(buf, mode="w", format="mp4")) as container:
        v_stream = cast(
            av.VideoStream,
            container.add_stream(video_codec, rate=Fraction(fps).limit_denominator(10000)),
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
            a_stream = cast(av.AudioStream, container.add_stream(audio_codec, rate=effective_audio_sample_rate))
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


def mux_video_audio_bytes_nvenc(
    video_frames: np.ndarray,
    audio_waveform: np.ndarray | None = None,
    *,
    fps: float = 25.0,
    audio_sample_rate: int = 44100,
    video_codec: str = "h264_nvenc",
    video_codec_options: dict[str, str] | None = None,
) -> bytes:
    """Encode uint8 frames on GPU through torchcodec's NVENC backend.

    ``video_frames`` must have shape ``(T, H, W, 3)`` (RGB); the RGB-to-NV12
    conversion and the H.264/HEVC/AV1 encode both run on the GPU.
    ``video_codec_options`` accepts the ``preset``/``crf`` keys handled by
    torchcodec directly; any other key is forwarded to the FFmpeg encoder as an
    extra option (e.g. ``{"rc": "vbr", "cq": "23"}`` for h264_nvenc).

    Raises:
        RuntimeError: If no CUDA device or no NVENC runtime is available.
    """
    import torch
    from torchcodec.encoders import Encoder

    if not torch.cuda.is_available():
        raise RuntimeError("NVENC video encoding requires a CUDA device.")
    if video_frames.ndim != 4 or video_frames.shape[-1] != 3 or video_frames.dtype != np.uint8:
        raise ValueError("NVENC video encoding expects contiguous uint8 frames shaped (T, H, W, 3).")

    _, height, width, _ = video_frames.shape
    frames = (
        torch.from_numpy(np.ascontiguousarray(video_frames))
        .permute(0, 3, 1, 2)
        .contiguous()
        .to("cuda", non_blocking=True)
    )

    options = dict(video_codec_options or {})
    preset = options.pop("preset", None)
    crf = options.pop("crf", None)

    encoder = Encoder()
    video_stream = encoder.add_video(
        height=height,
        width=width,
        frame_rate=float(fps),
        device="cuda",
        codec=video_codec,
        preset=preset,
        crf=float(crf) if crf is not None else None,
        extra_options=options or None,
    )

    audio_stream = None
    samples: np.ndarray | None = None
    if audio_waveform is not None:
        samples = audio_waveform.astype(np.float32)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        elif samples.ndim == 2 and samples.shape[0] > samples.shape[1]:
            samples = np.ascontiguousarray(samples.T)
        audio_stream = encoder.add_audio(
            sample_rate=int(audio_sample_rate),
            num_channels=int(samples.shape[0]),
        )

    buf = io.BytesIO()
    with encoder.open_file_like(buf, format="mp4"):
        video_stream.add_frames(frames)
        if audio_stream is not None and samples is not None:
            audio_stream.add_samples(torch.from_numpy(samples))
    return buf.getvalue()
