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
import math
import os
import tempfile
import wave
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

from vllm_omni.experimental.fullduplex.client import PCM16_BYTES_PER_SAMPLE, PCM16_SAMPLE_RATE

logger = logging.getLogger(__name__)


def materialize_media(value: Any, output_dir: str | Path, stem: str, suffix: str) -> Path:
    """Resolve common Hugging Face media values to a local file.

    ``str``/``Path`` values are returned as-is. Inline byte payloads (a
    ``bytes``/``bytearray`` or a Hugging Face ``Audio``/``Video``/``Image``
    feature mapping ``{"bytes": ..., "path": ...}``) are written to a real file
    under ``output_dir`` and that existing path is returned. For such mappings
    the ``path`` key is the original artifact name inside the dataset, which is
    usually *not* present on the local disk, so it is only used to pick a
    content extension and never treated as an existing file. A mapping or
    object carrying only a ``path`` (no inline ``bytes``) is returned as an
    already-materialized file reference.
    """
    if isinstance(value, str | Path):
        return Path(value)
    path = value.get("path") if isinstance(value, dict) else getattr(value, "path", None)
    payload = value.get("bytes") if isinstance(value, dict) else getattr(value, "bytes", None)
    if payload is None and isinstance(value, bytes | bytearray):
        payload = value
    if payload is not None:
        return _write_media_bytes(payload, output_dir, stem, _media_extension(path, suffix))
    if path:
        return Path(path)
    raise ValueError(f"cannot materialize media value for {stem!r}")


def _media_extension(path: str | Path | None, fallback: str) -> str:
    """Pick a safe lower-case file extension from ``path`` when available."""
    extension = Path(str(path)).suffix.lower() if path else ""
    if extension.startswith(".") and len(extension) <= 8 and extension[1:].isalnum():
        return extension
    return fallback


def _write_media_bytes(payload: bytes | bytearray, output_dir: str | Path, stem: str, extension: str) -> Path:
    data = bytes(payload)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{stem}{extension}"
    if destination.exists() and destination.stat().st_size == len(data) and destination.read_bytes() == data:
        return destination
    fd, temporary = tempfile.mkstemp(dir=root, prefix=f".{destination.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(temporary, destination)
    except OSError:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return destination


class _CodecContext(Protocol):
    @property
    def name(self) -> str | None: ...


class _VideoStream(Protocol):
    @property
    def codec_context(self) -> _CodecContext: ...


def _video_thread_type(stream: _VideoStream) -> str:
    """HEVC frame threads deadlock after seek; other codecs keep AUTO."""
    codec = getattr(getattr(stream, "codec_context", None), "name", None)
    return "SLICE" if codec == "hevc" else "AUTO"


def video_duration(path: str | Path) -> float:
    """Return the video duration in seconds using PyAV (no ffprobe subprocess).

    Raises:
        OSError: If the file cannot be opened (missing file, permission denied, etc.).
        ValueError: If the file is corrupt or cannot be parsed (e.g.
            :class:`av.error.InvalidDataError`).
    """
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
                stream.thread_type = _video_thread_type(stream)
                frame_count = sum(1 for _ in container.decode(stream))
                if stream.average_rate and frame_count > 0:
                    return frame_count / float(stream.average_rate)
            return 0.0
    except (OSError, ValueError) as exc:
        logger.warning("Failed to read video duration from %s: %s", path, exc)
        raise


def extract_jpeg(path: str | Path, *, timestamp: float, quality: int = 3) -> bytes:
    """Extract a single JPEG frame at ``timestamp`` using PyAV.

    ``quality`` follows the ffmpeg ``-q:v`` convention (lower = better, 1..31)
    and is mapped to the PIL JPEG quality scale (higher = better).

    ``timestamp`` is seconds after the video stream's first presentation time,
    the same origin ``read_soundtrack_pcm16`` uses. After ``seek`` the function
    continues decoding until a frame with PTS >= target_pts is found, matching
    the ``ffmpeg -ss`` semantic of returning the first decodable frame *at or
    after* the requested timestamp.

    Raises:
        ValueError: If the file contains no video stream or no frame could be decoded.
    """
    import av

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"no video stream in {path}")
        stream = container.streams.video[0]
        stream.thread_type = _video_thread_type(stream)
        time_base = stream.time_base
        if time_base is None:
            time_base = Fraction(1, 90000)
        # Session time 0 is the first presented frame, not absolute PTS 0.
        # Audio placement uses this same origin.
        origin = _presentation_origin(container)
        target_pts = int((origin + Fraction(max(0.0, timestamp))) / Fraction(time_base))
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
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(end) or end < 0:
        raise ValueError("fps must be finite and positive; video duration must be finite and nonnegative")
    index = 0
    timestamp = 0.0
    while timestamp < end:
        try:
            frame = extract_jpeg(path, timestamp=timestamp, quality=quality)
        except (OSError, ValueError) as exc:
            logger.warning("Frame extraction failed at %.3fs for %s: %s", timestamp, path, exc)
            break
        if frame:
            yield timestamp, frame
        # Repeated addition can turn the excluded endpoint into an extra frame
        # (e.g. 4 seconds at 1.5 fps ends at 3.9999999999999996).
        index += 1
        timestamp = index / fps


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
    except (OSError, ValueError) as exc:
        logger.warning("Failed to decode audio in %s: %s", source, exc)
        raise


def _presentation_origin(container: Any) -> Fraction:
    """Return the timeline zero used by frame extraction, in seconds.

    Frame requests are seconds after the video stream's first presentation
    time, not absolute PTS. Audio has to use that same origin: a container
    whose audio and video both start late is not silence, while audio that
    starts later than the video is.
    """
    if container.streams.video:
        stream = container.streams.video[0]
        if stream.start_time is not None and stream.time_base is not None:
            return Fraction(int(stream.start_time)) * Fraction(stream.time_base)
    start = getattr(container, "start_time", None)
    if start:
        import av

        return Fraction(int(start), int(av.time_base))
    return Fraction(0)


def _pcm_sample_index(frame: Any, origin: Fraction) -> int | None:
    """Map a decoded frame onto the 16 kHz timeline, or None when it has no PTS."""
    if frame.pts is None or frame.time_base is None:
        return None
    delta = Fraction(int(frame.pts)) * Fraction(frame.time_base) - origin
    return int(round(delta * PCM16_SAMPLE_RATE))


def read_soundtrack_pcm16(path: str | Path) -> bytes:
    """Decode a clip's own audio track to 16 kHz mono PCM16.

    The returned buffer starts at the video timeline's zero. A track with no
    audio stream is empty. A track whose presentation time is later than the
    video start is left-padded with silence; one that leads the video is trimmed.
    Sample placement follows each decoded frame's PTS, so a leading offset or an
    edit list that shifts the start stays aligned with the question audio and the
    extracted frames.
    """
    import av
    import numpy as np

    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                return b""
            origin = _presentation_origin(container)
            resampler = av.AudioResampler(format="s16", layout="mono", rate=PCM16_SAMPLE_RATE)
            pieces: list[tuple[int, Any]] = []
            cursor = 0

            def _take(frame: Any) -> None:
                nonlocal cursor
                samples = np.asarray(frame.to_ndarray(), dtype=np.int16).reshape(-1).copy()
                if samples.size == 0:
                    return
                index = _pcm_sample_index(frame, origin)
                if index is None:
                    index = cursor
                if index < 0:
                    samples = samples[-index:]
                    index = 0
                if samples.size == 0:
                    return
                pieces.append((index, samples))
                cursor = index + int(samples.size)

            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    _take(resampled)
            for resampled in resampler.resample(None):
                _take(resampled)
    except (OSError, ValueError) as exc:
        logger.warning("Failed to read the audio track from %s: %s", path, exc)
        raise
    if not pieces:
        return b""
    end = max((index + int(samples.size) for index, samples in pieces))
    mixed = np.zeros(end, dtype=np.int16)
    for index, samples in pieces:
        mixed[index : index + int(samples.size)] = samples
    return mixed.tobytes()


def mix_pcm16(*tracks: bytes) -> bytes:
    """Sum PCM16 tracks sample by sample, padding to the longest and clipping to int16."""
    import numpy as np

    arrays = [np.frombuffer(track[: len(track) // 2 * 2], dtype=np.int16) for track in tracks if track]
    if not arrays:
        return b""
    mixed = np.zeros(max(len(array) for array in arrays), dtype=np.int32)
    for array in arrays:
        mixed[: len(array)] += array
    return np.clip(mixed, -32768, 32767).astype(np.int16).tobytes()


def iter_av_units(
    audio_pcm16: bytes, frames: Iterator[tuple[float, bytes]], *, unit_ms: int = 1000
) -> Iterator[tuple[bytes, bytes | list[bytes] | None]]:
    """Attach every frame in each audio interval [start, end), including the tail.

    Audio must cover the video timeline. Frames that cannot fit the interval or
    the wire's two-image limit fail explicitly instead of being lost or delayed.
    """
    if unit_ms <= 0 or len(audio_pcm16) % PCM16_BYTES_PER_SAMPLE:
        raise ValueError("unit_ms must be positive and PCM16 must contain whole samples")
    unit_bytes = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE * unit_ms // 1000
    bytes_per_second = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE
    frame_iter = iter(frames)
    next_frame = next(frame_iter, None)
    previous_timestamp = -1.0
    for offset in range(0, len(audio_pcm16), unit_bytes):
        end = min(offset + unit_bytes, len(audio_pcm16))
        selected: list[bytes] = []
        while next_frame is not None:
            timestamp, frame = next_frame
            if not math.isfinite(timestamp) or timestamp < 0 or timestamp < previous_timestamp:
                raise ValueError("frame timestamps must be finite, nonnegative and ordered")
            if timestamp >= end / bytes_per_second:
                break
            selected.append(frame)
            if len(selected) > 2:
                raise ValueError("Realtime permits at most two frames per audio unit; reduce fps or unit_ms")
            previous_timestamp = timestamp
            next_frame = next(frame_iter, None)
        yield audio_pcm16[offset:end], selected[0] if len(selected) == 1 else selected or None
    if next_frame is not None:
        raise ValueError("frame timestamp is outside the audio timeline")


def _ffmpeg_q_to_pil_quality(quality: int) -> int:
    """Map ffmpeg ``-q:v`` (1..31, lower = better) to PIL JPEG quality (1..100, higher = better)."""
    ffmpeg_q = max(1, min(31, int(quality)))
    return max(1, min(100, round(95 - (ffmpeg_q - 1) * (93 / 30))))
