# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for omni_duplex_eval_media — PyAV-backed media helpers."""

from __future__ import annotations

import io
from pathlib import Path

import av
import numpy as np
import pytest

from vllm_omni.benchmarks.duplex.omni_duplex_eval_media import (
    _ffmpeg_q_to_pil_quality,
    extract_jpeg,
    read_audio_pcm16,
    video_duration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Helpers: synthetic test media generated with PyAV (no external files)
# ---------------------------------------------------------------------------


def _make_synthetic_mp4(duration: float = 1.0, fps: int = 30) -> bytes:
    """Return an MP4 byte blob with a solid-color pattern.

    Uses the built-in ``mpeg4`` encoder — available everywhere PyAV's bundled
    FFmpeg is present — so no external codec dependency.
    """
    buf = io.BytesIO()
    total_frames = int(duration * fps)
    with av.open(buf, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        for i in range(total_frames):
            frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), (i * 8) % 256, dtype=np.uint8), format="rgb24")
            for pkt in stream.encode(frame):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
    return buf.getvalue()


def _make_synthetic_wav(duration: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Return a 16-bit mono PCM WAV byte blob (440 Hz sine wave)."""
    import wave as _wave

    n = int(sample_rate * duration)
    samples = (np.sin(2 * np.pi * 440 * np.arange(n) / sample_rate) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with _wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def _make_synthetic_ogg(duration: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Return an OGG Vorbis byte blob (440 Hz sine wave).

    Uses ``libopus`` as a fallback if ``libvorbis`` is unavailable in the
    bundled FFmpeg (e.g. PyAV wheels from an older build).
    """
    buf = io.BytesIO()
    n = int(sample_rate * duration)
    samples = (np.sin(2 * np.pi * 440 * np.arange(n) / sample_rate) * 0.3).astype(np.float32)
    with av.open(buf, mode="w", format="ogg") as container:
        codec_name = "libopus" if "libvorbis" not in av.codecs_available else "libvorbis"
        stream = container.add_stream(codec_name, rate=sample_rate, layout="mono")
        frame = av.AudioFrame.from_ndarray(samples[np.newaxis, :], format="fltp", layout="mono")
        frame.sample_rate = sample_rate
        for pkt in stream.encode(frame):
            container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# video_duration
# ---------------------------------------------------------------------------


class TestVideoDuration:
    @staticmethod
    def test_returns_duration(tmp_path: Path) -> None:
        mp4 = _make_synthetic_mp4(duration=2.0, fps=30)
        path = tmp_path / "test.mp4"
        path.write_bytes(mp4)
        dur = video_duration(path)
        assert abs(dur - 2.0) < 0.05, f"expected ~2.0s, got {dur}"

    @staticmethod
    def test_zero_for_invalid_path(tmp_path: Path) -> None:
        dur = video_duration(tmp_path / "nonexistent.mp4")
        assert dur == 0.0


# ---------------------------------------------------------------------------
# extract_jpeg
# ---------------------------------------------------------------------------


class TestExtractJpeg:
    @staticmethod
    def test_returns_valid_jpeg(tmp_path: Path) -> None:
        mp4 = _make_synthetic_mp4(duration=1.0, fps=30)
        path = tmp_path / "test.mp4"
        path.write_bytes(mp4)
        jpeg = extract_jpeg(path, timestamp=0.5, quality=3)
        assert jpeg.startswith(b"\xff\xd8"), "not a valid JPEG header"
        assert len(jpeg) > 100

    @staticmethod
    def test_quality_maps_to_pil_scale(tmp_path: Path) -> None:
        """Higher ffmpeg quality (lower q value) produces a larger JPEG."""
        mp4 = _make_synthetic_mp4(duration=1.0, fps=30)
        path = tmp_path / "test.mp4"
        path.write_bytes(mp4)
        high = extract_jpeg(path, timestamp=0.5, quality=1)
        low = extract_jpeg(path, timestamp=0.5, quality=31)
        assert len(high) > len(low), "quality=1 should produce larger JPEG than quality=31"

    @staticmethod
    def test_negative_timestamp_clamped(tmp_path: Path) -> None:
        """Negative timestamps are clamped to 0."""
        mp4 = _make_synthetic_mp4(duration=1.0, fps=30)
        path = tmp_path / "test.mp4"
        path.write_bytes(mp4)
        jpeg = extract_jpeg(path, timestamp=-1.0, quality=3)
        assert jpeg.startswith(b"\xff\xd8")

    @staticmethod
    def test_raises_value_error_for_no_video(tmp_path: Path) -> None:
        wav = _make_synthetic_wav()
        path = tmp_path / "test.wav"
        path.write_bytes(wav)
        with pytest.raises(ValueError):
            extract_jpeg(path, timestamp=0.0)

    @staticmethod
    def test_long_gop_returns_different_frames_for_different_timestamps(tmp_path: Path) -> None:
        """Long GOP: different timestamps must return different frame content (P2)."""
        # Create a long-GOP video: 3s @ 10fps = 30 frames, gop_size=30 (1 keyframe)
        buf = io.BytesIO()
        fps = 10
        duration = 3.0
        total_frames = int(duration * fps)
        with av.open(buf, mode="w", format="mp4") as container:
            stream = container.add_stream("mpeg4", rate=fps)
            stream.width = 64
            stream.height = 64
            stream.pix_fmt = "yuv420p"
            stream.gop_size = 30  # 3 seconds per keyframe
            for i in range(total_frames):
                frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), (i * 64) % 256, dtype=np.uint8), format="rgb24")
                for pkt in stream.encode(frame):
                    container.mux(pkt)
            for pkt in stream.encode():
                container.mux(pkt)
        mp4 = buf.getvalue()

        path = tmp_path / "long_gop.mp4"
        path.write_bytes(mp4)

        # Extract frames at different timestamps
        frame0 = extract_jpeg(path, timestamp=0.0, quality=3)
        frame1 = extract_jpeg(path, timestamp=1.0, quality=3)
        frame2 = extract_jpeg(path, timestamp=2.0, quality=3)

        # Verify all are valid JPEGs
        assert frame0.startswith(b"\xff\xd8")
        assert frame1.startswith(b"\xff\xd8")
        assert frame2.startswith(b"\xff\xd8")

        # Verify content differs: decode JPEGs and compare mean brightness
        from PIL import Image

        img0 = np.asarray(Image.open(io.BytesIO(frame0)))
        img1 = np.asarray(Image.open(io.BytesIO(frame1)))
        img2 = np.asarray(Image.open(io.BytesIO(frame2)))
        means = [img.mean() for img in (img0, img1, img2)]
        assert len(set(round(m, 1) for m in means)) >= 2, (
            f"timestamps 0/1/2s should give different frames, got means {means}"
        )


# ---------------------------------------------------------------------------
# read_audio_pcm16
# ---------------------------------------------------------------------------


class TestReadAudioPcm16:
    @staticmethod
    def test_wav_fast_path(tmp_path: Path) -> None:
        """WAV with matching params uses the fast wave module path."""
        wav = _make_synthetic_wav(duration=1.0, sample_rate=16000)
        path = tmp_path / "test.wav"
        path.write_bytes(wav)
        pcm = read_audio_pcm16(path)
        assert len(pcm) == 16000 * 2, f"expected 32000 bytes, got {len(pcm)}"
        assert isinstance(pcm, bytes)

    @staticmethod
    def test_wav_different_rate_resamples(tmp_path: Path) -> None:
        """A 48 kHz WAV is resampled down to 16 kHz — exact sample count after flush."""
        wav = _make_synthetic_wav(duration=1.0, sample_rate=48000)
        path = tmp_path / "test.wav"
        path.write_bytes(wav)
        pcm = read_audio_pcm16(path)
        assert len(pcm) > 0
        assert len(pcm) % 2 == 0
        # 1s @ 48kHz → 1s @ 16kHz = 16000 samples = 32000 bytes (±1 sample)
        assert abs(len(pcm) - 32000) <= 4, f"expected ~32000 bytes, got {len(pcm)}"

    @staticmethod
    def test_non_wav_ogg(tmp_path: Path) -> None:
        """Non-WAV audio (OGG) is decoded and resampled via PyAV."""
        ogg = _make_synthetic_ogg(duration=1.0, sample_rate=16000)
        path = tmp_path / "test.ogg"
        path.write_bytes(ogg)
        pcm = read_audio_pcm16(path)
        assert len(pcm) > 0
        assert len(pcm) % 2 == 0

    @staticmethod
    def test_non_wav_ogg_resample_accuracy(tmp_path: Path) -> None:
        """OGG resample should produce exact sample count after flush."""
        ogg = _make_synthetic_ogg(duration=1.5, sample_rate=48000)
        path = tmp_path / "test.ogg"
        path.write_bytes(ogg)
        pcm = read_audio_pcm16(path)
        # 1.5s @ 48kHz → 1.5s @ 16kHz = 24000 samples = 48000 bytes (±1 sample)
        expected = int(1.5 * 16000) * 2
        assert abs(len(pcm) - expected) <= 4, f"expected ~{expected} bytes, got {len(pcm)}"

    @staticmethod
    def test_empty_bytes_for_broken_file(tmp_path: Path) -> None:
        path = tmp_path / "broken.bin"
        path.write_bytes(b"not a media file")
        pcm = read_audio_pcm16(path)
        assert pcm == b""


# ---------------------------------------------------------------------------
# _ffmpeg_q_to_pil_quality
# ---------------------------------------------------------------------------


class TestFfmpegQToPilQuality:
    @staticmethod
    def test_q1_maps_to_best() -> None:
        assert _ffmpeg_q_to_pil_quality(1) >= 90

    @staticmethod
    def test_q31_maps_to_lowest() -> None:
        assert _ffmpeg_q_to_pil_quality(31) <= 10

    @staticmethod
    def test_monotonic() -> None:
        """Higher ffmpeg q → lower PIL quality (monotonically non-increasing)."""
        prev = 100
        for q in range(1, 32):
            p = _ffmpeg_q_to_pil_quality(q)
            assert p <= prev, f"quality.increase at q={q}: {prev} → {p}"
            prev = p
