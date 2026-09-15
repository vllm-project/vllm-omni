# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Small media helpers used by the duplex runner and judge.

Video and audio decoding is delegated to PyAV (``av``), which bundles a pinned
FFmpeg version, so results do not depend on whatever ffmpeg binary the host
provides (see #7364: host ffmpeg 4.4.2 could hang forever on some HEVC clips).
"""

from __future__ import annotations

import io
import logging
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from vllm_omni.experimental.fullduplex.client import PCM16_BYTES_PER_SAMPLE, PCM16_SAMPLE_RATE

logger = logging.getLogger(__name__)


def materialize_media(value: Any, output_dir: str | Path, stem: str, suffix: str) -> Path:
    """Resolve common Hugging Face media values to a local file."""
    if isinstance(value, str | Path):
        return Path(value)
    path = value.get("path") if isinstance(value, dict) else getattr(value, "path", None)
    if path:
        return Path(path)
    payload = value.get("bytes") if isinstance(value, dict) else None
    if payload is None and isinstance(value, bytes | bytearray):
        payload = value
    if payload is None:
        raise ValueError(f"cannot materialize media value for {stem!r}")
    destination = Path(output_dir) / f"{stem}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(bytes(payload))
    return destination


def video_duration(path: str | Path) -> float:
    """Return the video duration in seconds using PyAV (no ffprobe subprocess)."""
    import av

    try:
        with av.open(str(path)) as container:
            # Container-level duration is the same source ffprobe's format=duration reads.
            if container.duration and container.duration > 0:
                return max(0.0, container.duration / av.time_base)
            if container.streams.video:
                stream = container.streams.video[0]
                if stream.duration is not None and stream.time_base is not None:
                    return max(0.0, float(stream.duration * stream.time_base))
            # Last resort: count decodable frames.
            if container.streams.video:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                frame_count = sum(1 for _ in container.decode(stream))
                if stream.average_rate and frame_count > 0:
                    return frame_count / float(stream.average_rate)
            return 0.0
    except OSError:
        return 0.0


def extract_jpeg(path: str | Path, *, timestamp: float, quality: int = 3) -> bytes:
    """Extract a single JPEG frame at ``timestamp`` using PyAV.

    ``quality`` follows the ffmpeg ``-q:v`` convention (lower = better, 1..31)
    and is mapped to the PIL JPEG quality scale (higher = better).

    After ``seek`` the function continues decoding until a frame with PTS >=
    target_pts is found, matching the ``ffmpeg -ss`` semantic of returning the
    first decodable frame *at or after* the requested timestamp.

    Raises:
        ValueError: If the file contains no video stream or no frame could be decoded.
    """
    import av

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"no video stream in {path}")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = stream.time_base
        if time_base is None:
            time_base = 1 / 90000
        target_pts = int(max(0.0, timestamp) / time_base)
        container.seek(target_pts, any_frame=False, backward=True, stream=stream)
        closest_frame = None
        for frame in container.decode(stream):
            closest_frame = frame
            if frame.pts is not None and frame.pts >= target_pts:
                break
        if closest_frame is not None:
            image = closest_frame.to_image()
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=_ffmpeg_q_to_pil_quality(quality))
            return buf.getvalue()
    raise ValueError(f"no decodable video frames in {path} at timestamp {timestamp}")


def iter_jpegs(
    path: str | Path, *, fps: float = 1.0, duration: float | None = None, quality: int = 3
) -> Iterator[tuple[float, bytes]]:
    end = duration if duration is not None else video_duration(path)
    step = 1.0 / fps
    timestamp = 0.0
    while timestamp < end:
        try:
            frame = extract_jpeg(path, timestamp=timestamp, quality=quality)
        except (OSError, ValueError) as exc:
            logger.warning("Frame extraction failed at %.3fs for %s: %s", timestamp, path, exc)
            break
        if frame:
            yield timestamp, frame
        timestamp += step


def read_audio_pcm16(path: str | Path) -> bytes:
    source = Path(path)
    if source.suffix.lower() == ".wav":
        try:
            with wave.open(str(source), "rb") as wav_file:
                if (
                    wav_file.getnchannels() == 1
                    and wav_file.getsampwidth() == 2
                    and wav_file.getframerate() == PCM16_SAMPLE_RATE
                    and wav_file.getcomptype() == "NONE"
                ):
                    return wav_file.readframes(wav_file.getnframes())
        except (wave.Error, OSError):
            pass
    # Non-WAV audio: decode and resample to mono s16 PCM via PyAV.
    import av
    import numpy as np

    try:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=PCM16_SAMPLE_RATE)
        chunks: list[Any] = []
        with av.open(str(source)) as container:
            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray())
            # Flush the resampler to drain buffered tail samples (P3).
            for resampled in resampler.resample(None):
                chunks.append(resampled.to_ndarray())
        if not chunks:
            return b""
        return np.concatenate([chunk.reshape(-1) for chunk in chunks]).astype(np.int16).tobytes()
    except (OSError, ValueError):
        return b""


def iter_av_units(
    audio_pcm16: bytes, frames: Iterator[tuple[float, bytes]], *, unit_ms: int = 1000
) -> Iterator[tuple[bytes, bytes | None]]:
    unit_bytes = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * unit_ms // 1000
    frame_iter = iter(frames)
    next_frame = next(frame_iter, None)
    for offset in range(0, len(audio_pcm16), unit_bytes):
        timestamp = offset / (PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE)
        frame = None
        if next_frame is not None and next_frame[0] <= timestamp + unit_ms / 1000:
            frame = next_frame[1]
            next_frame = next(frame_iter, None)
        yield audio_pcm16[offset : offset + unit_bytes], frame


def _ffmpeg_q_to_pil_quality(quality: int) -> int:
    """Map ffmpeg ``-q:v`` (1..31, lower = better) to PIL JPEG quality (1..100, higher = better)."""
    ffmpeg_q = max(1, min(31, int(quality)))
    return max(1, min(100, round(95 - (ffmpeg_q - 1) * (93 / 30))))
