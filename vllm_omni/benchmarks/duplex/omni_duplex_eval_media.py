# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Small ffmpeg-backed media helpers used by the duplex runner and judge."""

from __future__ import annotations

import subprocess
import wave
from collections.abc import Iterator
from pathlib import Path

from vllm_omni.experimental.fullduplex.client import PCM16_BYTES_PER_SAMPLE, PCM16_SAMPLE_RATE


def _run_ffmpeg(args: list[str]) -> bytes:
    result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *args], capture_output=True, check=True)
    return result.stdout


def video_duration(path: str | Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return max(0.0, float(result.stdout.strip()))


def extract_jpeg(path: str | Path, *, timestamp: float, quality: int = 3) -> bytes:
    return _run_ffmpeg(
        [
            "-ss",
            f"{max(0.0, timestamp):.3f}",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-q:v",
            str(quality),
            "-f",
            "image2",
            "pipe:1",
        ]
    )


def iter_jpegs(
    path: str | Path, *, fps: float = 1.0, duration: float | None = None, quality: int = 3
) -> Iterator[tuple[float, bytes]]:
    end = duration if duration is not None else video_duration(path)
    step = 1.0 / fps
    timestamp = 0.0
    while timestamp < end:
        try:
            frame = extract_jpeg(path, timestamp=timestamp, quality=quality)
        except (OSError, subprocess.CalledProcessError):
            break
        if frame:
            yield timestamp, frame
        timestamp += step


def read_audio_pcm16(path: str | Path) -> bytes:
    source = Path(path)
    if source.suffix.lower() == ".wav":
        with wave.open(str(source), "rb") as wav_file:
            if (
                wav_file.getnchannels() != 1
                or wav_file.getsampwidth() != 2
                or wav_file.getframerate() != PCM16_SAMPLE_RATE
            ):
                raise ValueError("question_audio WAV must be mono PCM16 at 16 kHz")
            return wav_file.readframes(wav_file.getnframes())
    return _run_ffmpeg(["-i", str(source), "-f", "s16le", "-ac", "1", "-ar", str(PCM16_SAMPLE_RATE), "pipe:1"])


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
