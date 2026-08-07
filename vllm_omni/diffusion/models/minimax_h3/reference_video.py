# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 Ref2VA reference-video preparation."""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm_omni.errors import OmniClientError

MINIMAX_H3_FPS = 24.0
MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS = 2.0
MINIMAX_H3_QWEN_TEMPORAL_PATCH = 2
MINIMAX_H3_BASE_SHORT_EDGE = 768
MINIMAX_H3_MAX_PIXELS = 768 * 1344
MINIMAX_H3_CANVAS_MULTIPLE = 32
MINIMAX_H3_MIN_REFERENCE_DIMENSION = 256
MINIMAX_H3_MAX_REFERENCE_DIMENSION = 5760
MINIMAX_H3_MIN_REFERENCE_FPS = 23.976
MINIMAX_H3_MAX_REFERENCE_FPS = 60.0
MINIMAX_H3_MIN_REFERENCE_DURATION = 2.0
MINIMAX_H3_MAX_REFERENCE_DURATION = 15.0
MINIMAX_H3_MAX_REFERENCE_TOTAL_DURATION = 15.0
# ffprobe reports container durations at a finer precision than the official
# example's rounded values. Trim only this small probe-rounding excess.
MINIMAX_H3_REFERENCE_DURATION_EPSILON = 1e-2
MINIMAX_H3_MAX_VIDEO_BYTES = 50 * 1024 * 1024
MINIMAX_H3_MAX_AUDIO_BYTES = 15 * 1024 * 1024
MINIMAX_H3_VIDEO_FORMATS = frozenset({"mp4", "mov"})
MINIMAX_H3_VIDEO_CODECS = frozenset({"h264", "h265", "hevc"})
MINIMAX_H3_AUDIO_FORMATS = frozenset({"wav", "mp3"})
MINIMAX_H3_AUDIO_CODECS = frozenset({"aac", "mp3"})


def _nearest_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def _probe_video(path: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            (
                "stream=codec_type,codec_name,width,height,r_frame_rate,nb_read_frames,nb_frames,duration,"
                "sample_aspect_ratio:format=format_name,size,duration"
            ),
            "-of",
            "json",
            path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    probe = json.loads(result.stdout)
    streams = probe.get("streams") or []
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    if not video_streams:
        raise OmniClientError(f"media has no video stream: {path}")
    stream = video_streams[0]
    try:
        numerator, denominator = str(stream["r_frame_rate"]).split("/", 1)
        fps = float(numerator) / float(denominator)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise OmniClientError(f"cannot determine video FPS: {path}") from exc
    if not math.isfinite(fps) or fps <= 0:
        raise OmniClientError(f"cannot determine video FPS: {path}")
    raw_count = stream.get("nb_read_frames") or stream.get("nb_frames")
    if raw_count in (None, "", "N/A"):
        raise OmniClientError(f"cannot determine video frame count: {path}")
    raw_duration = stream.get("duration")
    if raw_duration in (None, "", "N/A"):
        raw_duration = probe.get("format", {}).get("duration")
    duration = float(raw_duration) if raw_duration not in (None, "", "N/A") else int(raw_count) / fps
    format_info = probe.get("format") or {}
    format_names = tuple(
        name.strip().lower() for name in str(format_info.get("format_name", "")).split(",") if name.strip()
    )
    raw_size = format_info.get("size")
    file_size = int(raw_size) if raw_size not in (None, "", "N/A") else os.path.getsize(path)
    audio_codecs = tuple(
        str(stream.get("codec_name", "")).lower()
        for stream in streams
        if stream.get("codec_type") == "audio" and stream.get("codec_name")
    )
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "frame_count": int(raw_count),
        "duration": duration,
        "format_names": format_names,
        "video_codec": str(stream.get("codec_name", "")).lower(),
        "audio_codecs": audio_codecs,
        "file_size": file_size,
    }


def _validate_reference_video_metadata(meta: dict[str, Any], *, index: int, source: str) -> None:
    width = int(meta["width"])
    height = int(meta["height"])
    if (
        min(width, height) < MINIMAX_H3_MIN_REFERENCE_DIMENSION
        or max(width, height) > MINIMAX_H3_MAX_REFERENCE_DIMENSION
    ):
        raise OmniClientError(f"reference video {index} dimensions must be in [256, 5760] pixels, got {width}x{height}")
    ratio = width / height
    if not 0.4 <= ratio <= 2.5:
        raise OmniClientError(f"reference video {index} aspect ratio must be in [0.4, 2.5], got {width}x{height}")

    fps = float(meta["fps"])
    if not MINIMAX_H3_MIN_REFERENCE_FPS <= fps <= MINIMAX_H3_MAX_REFERENCE_FPS:
        raise OmniClientError(f"reference video {index} FPS must be in [23.976, 60], got {fps:.3f}")
    duration = float(meta["duration"])
    if (
        not math.isfinite(duration)
        or not MINIMAX_H3_MIN_REFERENCE_DURATION <= duration <= MINIMAX_H3_MAX_REFERENCE_DURATION
    ):
        raise OmniClientError(f"reference video {index} duration must be in [2, 15] seconds, got {duration:.3f}")

    raw_file_size = meta.get("file_size")
    file_size = int(raw_file_size) if raw_file_size is not None else os.path.getsize(source)
    if file_size > MINIMAX_H3_MAX_VIDEO_BYTES:
        raise OmniClientError(f"reference video {index} exceeds the 50 MiB size limit")
    format_names = set(meta.get("format_names", ()))
    if not format_names.intersection(MINIMAX_H3_VIDEO_FORMATS):
        formats = ", ".join(sorted(format_names)) or "unknown"
        raise OmniClientError(f"reference video {index} must use an MP4 or MOV container, got {formats}")
    video_codec = str(meta.get("video_codec", "")).lower()
    if video_codec not in MINIMAX_H3_VIDEO_CODECS:
        raise OmniClientError(f"reference video {index} must use H.264 or H.265 video, got {video_codec or 'unknown'}")
    audio_codecs = tuple(str(codec).lower() for codec in meta.get("audio_codecs", ()))
    invalid_audio = [codec for codec in audio_codecs if codec not in MINIMAX_H3_AUDIO_CODECS]
    if invalid_audio:
        raise OmniClientError(
            f"reference video {index} must use AAC or MP3 audio when audio is present, got {invalid_audio[0]}"
        )


def _audio_items(values: Any) -> list[Any]:
    if isinstance(values, (str, os.PathLike)):
        return [values]
    if isinstance(values, tuple) and len(values) == 2 and isinstance(values[1], (int, np.integer)):
        return [values]
    if isinstance(values, (list, tuple)):
        return list(values)
    return [values]


def _probe_audio(path: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name,duration:format=format_name,size,duration",
            "-of",
            "json",
            path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    probe = json.loads(result.stdout)
    streams = [stream for stream in probe.get("streams", []) if stream.get("codec_type") == "audio"]
    if not streams:
        raise OmniClientError(f"media has no audio stream: {path}")
    stream = streams[0]
    raw_duration = stream.get("duration")
    if raw_duration in (None, "", "N/A"):
        raw_duration = (probe.get("format") or {}).get("duration")
    if raw_duration in (None, "", "N/A"):
        raise OmniClientError(f"cannot determine audio duration: {path}")
    format_info = probe.get("format") or {}
    format_names = tuple(
        name.strip().lower() for name in str(format_info.get("format_name", "")).split(",") if name.strip()
    )
    raw_size = format_info.get("size")
    file_size = int(raw_size) if raw_size not in (None, "", "N/A") else os.path.getsize(path)
    return {
        "duration": float(raw_duration),
        "format_names": format_names,
        "codec": str(stream.get("codec_name", "")).lower(),
        "file_size": file_size,
    }


def validate_reference_audio_files(values: Any) -> None:
    """Validate path-backed standalone audio references against the H3 spec."""
    total_duration = 0.0
    for index, value in enumerate(_audio_items(values)):
        if not isinstance(value, (str, os.PathLike)):
            continue
        source = str(value)
        meta = _probe_audio(source)
        file_size = int(meta["file_size"])
        if file_size > MINIMAX_H3_MAX_AUDIO_BYTES:
            raise OmniClientError(f"reference audio {index} exceeds the 15 MiB size limit")
        if not set(meta["format_names"]).intersection(MINIMAX_H3_AUDIO_FORMATS):
            formats = ", ".join(meta["format_names"]) or "unknown"
            raise OmniClientError(f"reference audio {index} must be WAV or MP3, got {formats}")
        codec = str(meta["codec"]).lower()
        if codec not in MINIMAX_H3_AUDIO_CODECS and not codec.startswith("pcm_"):
            raise OmniClientError(f"reference audio {index} must use MP3 or PCM audio, got {codec or 'unknown'}")
        duration = float(meta["duration"])
        if (
            not math.isfinite(duration)
            or not MINIMAX_H3_MIN_REFERENCE_DURATION <= duration <= MINIMAX_H3_MAX_REFERENCE_DURATION
        ):
            raise OmniClientError(f"reference audio {index} duration must be in [2, 15] seconds, got {duration:.3f}")
        total_duration += duration
    if total_duration > MINIMAX_H3_MAX_REFERENCE_TOTAL_DURATION:
        raise OmniClientError("reference audio must be at most 15 seconds in total")


def validate_reference_audio_waveforms(values: list[tuple[torch.Tensor, int]]) -> None:
    """Validate in-memory standalone audio references using sample counts."""
    total_duration = 0.0
    for index, (waveform, sample_rate) in enumerate(values):
        if int(sample_rate) <= 0:
            raise OmniClientError(f"reference audio {index} has an invalid sample rate")
        samples = int(waveform.shape[-1])
        duration = samples / int(sample_rate)
        if not MINIMAX_H3_MIN_REFERENCE_DURATION <= duration <= MINIMAX_H3_MAX_REFERENCE_DURATION:
            raise OmniClientError(f"reference audio {index} duration must be in [2, 15] seconds, got {duration:.3f}")
        total_duration += duration
    if total_duration > MINIMAX_H3_MAX_REFERENCE_TOTAL_DURATION:
        raise OmniClientError("reference audio must be at most 15 seconds in total")


def _reference_video_shape(width: int, height: int) -> tuple[int, int]:
    if (
        min(width, height) < MINIMAX_H3_MIN_REFERENCE_DIMENSION
        or max(width, height) > MINIMAX_H3_MAX_REFERENCE_DIMENSION
    ):
        raise OmniClientError(f"reference video dimensions must be in [256, 5760] pixels, got {width}x{height}")
    ratio = float(width) / float(height)
    if not 0.4 <= ratio <= 2.5:
        raise OmniClientError(f"reference video aspect ratio must be in [0.4, 2.5], got {width}x{height}")
    if ratio >= 1.0:
        target_width = MINIMAX_H3_BASE_SHORT_EDGE * ratio
        target_height = float(MINIMAX_H3_BASE_SHORT_EDGE)
    else:
        target_width = float(MINIMAX_H3_BASE_SHORT_EDGE)
        target_height = MINIMAX_H3_BASE_SHORT_EDGE / ratio
    area = target_width * target_height
    if area > MINIMAX_H3_MAX_PIXELS:
        scale = math.sqrt(MINIMAX_H3_MAX_PIXELS / area)
        target_width *= scale
        target_height *= scale
    return (
        _nearest_multiple(target_width, MINIMAX_H3_CANVAS_MULTIPLE),
        _nearest_multiple(target_height, MINIMAX_H3_CANVAS_MULTIPLE),
    )


def _transcode_reference_video(
    source: str,
    *,
    target_width: int,
    target_height: int,
    workdir: str,
    start_time_seconds: float = 0.0,
    duration_seconds: float | None = None,
) -> str:
    output = str(Path(workdir) / "prepared.mp4")
    duration_args = ["-t", f"{float(duration_seconds):.6f}"] if duration_seconds is not None else []
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{float(start_time_seconds):.6f}",
            "-i",
            source,
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            (f"fps={MINIMAX_H3_FPS:g},scale={target_width}:{target_height}:flags=lanczos,setsar=1"),
            *duration_args,
            "-metadata:s:v:0",
            "rotate=0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            output,
        ],
        check=True,
    )
    return output


def prepare_reference_videos(
    values: Any,
    *,
    target_frame_count: int,
    workdir: str,
    start_time_seconds: Any = None,
) -> list[dict[str, Any]]:
    # Keep this argument for compatibility with existing pipeline callers;
    # reference inputs are now bounded by their source/segment durations, not
    # clipped to the generated frame count.
    del target_frame_count
    if isinstance(values, (str, os.PathLike)):
        values = [values]
    if not isinstance(values, (list, tuple)) or not values:
        raise OmniClientError("MiniMax H3 Ref2VA video input must be a path or a non-empty list of paths")
    if isinstance(start_time_seconds, (list, tuple)):
        start_times = list(start_time_seconds)
    else:
        start_times = [start_time_seconds] * len(values)
    if len(start_times) != len(values):
        raise OmniClientError("start_time_seconds must contain one value per reference video")

    prepared: list[dict[str, Any]] = []
    total_duration = 0.0
    for index, value in enumerate(values):
        item_start = start_times[index]
        if isinstance(value, dict):
            item_start = value.get("start_time_seconds", item_start)
            value = value.get("path", value.get("video_path", value.get("video_url")))
        if not isinstance(value, (str, os.PathLike)):
            raise OmniClientError(
                f"MiniMax H3 Ref2VA video references require file paths, got item {index}: {type(value)!r}"
            )
        source = str(value)
        meta = _probe_video(source)
        _validate_reference_video_metadata(meta, index=index, source=source)
        start = 0.0 if item_start is None else float(item_start)
        if not math.isfinite(start) or start < 0 or start >= float(meta["duration"]):
            raise OmniClientError(f"reference video {index} start_time_seconds is outside the source duration")
        duration = float(meta["duration"]) - start
        if duration < MINIMAX_H3_MIN_REFERENCE_DURATION:
            raise OmniClientError(
                f"reference video {index} must provide at least 2 seconds after start_time_seconds, got {duration:.3f}"
            )
        remaining_duration = MINIMAX_H3_MAX_REFERENCE_TOTAL_DURATION - total_duration
        if duration > remaining_duration:
            if duration - remaining_duration > MINIMAX_H3_REFERENCE_DURATION_EPSILON:
                raise OmniClientError("reference videos must be at most 15 seconds in total")
            duration = remaining_duration
            if duration < MINIMAX_H3_MIN_REFERENCE_DURATION:
                raise OmniClientError("reference videos must be at most 15 seconds in total")
        total_duration += duration
        width, height = _reference_video_shape(meta["width"], meta["height"])
        item_workdir = Path(workdir) / f"video_{index}"
        item_workdir.mkdir(parents=True)
        prepared_path = _transcode_reference_video(
            source,
            target_width=width,
            target_height=height,
            workdir=str(item_workdir),
            start_time_seconds=start,
            duration_seconds=duration,
        )
        prepared.append(
            {
                "original_path": source,
                "prepared_path": prepared_path,
                "input_has_audio": bool(meta.get("audio_codecs")),
                "width": width,
                "height": height,
                "start_time_seconds": start,
                "duration_seconds": duration,
            }
        )
    return prepared


def load_video_frames(path: str) -> np.ndarray:
    try:
        import decord
    except ImportError:
        import av

        with av.open(path) as container:
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        if not frames:
            raise OmniClientError(f"video has no frames: {path}")
        return np.stack(frames)

    reader = decord.VideoReader(path)
    if len(reader) <= 0:
        raise OmniClientError(f"video has no frames: {path}")
    frames = reader.get_batch(list(range(len(reader))))
    return frames.asnumpy() if hasattr(frames, "asnumpy") else np.asarray(frames)


def sample_reference_video_frames(
    prepared_path: str,
    *,
    workdir: str,
) -> dict[str, Any]:
    from PIL import Image

    meta = _probe_video(prepared_path)
    ratio = MINIMAX_H3_FPS / MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS
    indices: list[int] = []
    cursor = 0.0
    while True:
        frame_index = int(round(cursor))
        if frame_index >= meta["frame_count"]:
            break
        if not indices or frame_index > indices[-1]:
            indices.append(frame_index)
        cursor += ratio
    if not indices:
        raise OmniClientError(f"no frames sampled from {prepared_path}")

    frame_dir = Path(workdir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for output_index, source_index in enumerate(indices, start=1):
        output = frame_dir / f"frame_{output_index:06d}.png"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                prepared_path,
                "-vf",
                f"select=eq(n\\,{source_index})",
                "-vsync",
                "vfr",
                "-frames:v",
                "1",
                str(output),
            ],
            check=True,
        )
        frames.append(np.asarray(Image.open(output).convert("RGB")))

    timestamps = [index / MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS for index in range(len(indices))]
    timestamps += [timestamps[-1]] * ((-len(timestamps)) % MINIMAX_H3_QWEN_TEMPORAL_PATCH)
    block_timestamps = [
        (timestamps[index] + timestamps[index + MINIMAX_H3_QWEN_TEMPORAL_PATCH - 1]) / 2
        for index in range(
            0,
            len(timestamps),
            MINIMAX_H3_QWEN_TEMPORAL_PATCH,
        )
    ]
    return {
        "frames": frames,
        "block_timestamps": block_timestamps,
    }


def _soundfile_to_waveform(path: str) -> tuple[torch.Tensor, int]:
    import soundfile as sf

    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    # soundfile is (samples, channels); torchaudio.load is (channels, samples).
    waveform = torch.from_numpy(data).t().contiguous()
    return waveform, int(sample_rate)


def load_audio_file(path: str) -> tuple[torch.Tensor, int]:
    """Load an audio file as ``(waveform[C, T] float32, sample_rate)``.

    torchaudio 2.6+ routes ``load`` through TorchCodec, whose wheels do not load
    on aarch64 with a CPU-only torch (they link CUDA libraries that are absent).
    Fall back to soundfile; if the container is not libsndfile-readable (e.g.
    mp3/m4a/mp4), demux to wav with ffmpeg first so any ffmpeg-supported input
    works at its native sample rate.
    """
    try:
        import torchaudio

        return torchaudio.load(path)
    except (ImportError, RuntimeError, OSError):
        pass
    try:
        return _soundfile_to_waveform(path)
    except Exception:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="minimax_h3_audio_") as tmpdir:
            wav = str(Path(tmpdir) / "audio.wav")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    path,
                    "-vn",
                    "-f",
                    "wav",
                    wav,
                ],
                check=True,
            )
            return _soundfile_to_waveform(wav)


def load_video_audio(
    path: str,
    *,
    start_time_seconds: float = 0.0,
    duration_seconds: float | None = None,
) -> tuple[torch.Tensor, int]:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="minimax_h3_video_audio_") as tmpdir:
        output = str(Path(tmpdir) / "audio.wav")
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{float(start_time_seconds):.6f}",
            "-i",
            path,
            "-vn",
            "-ac",
            "2",
            "-ar",
            "44100",
            "-f",
            "wav",
        ]
        if duration_seconds is not None:
            command.extend(["-t", f"{float(duration_seconds):.6f}"])
        command.append(output)
        subprocess.run(command, check=True)
        return load_audio_file(output)


__all__ = [
    "load_audio_file",
    "load_video_audio",
    "load_video_frames",
    "prepare_reference_videos",
    "sample_reference_video_frames",
    "validate_reference_audio_files",
    "validate_reference_audio_waveforms",
]
