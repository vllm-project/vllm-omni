# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU-only admission and decoding for trusted, file-backed H3 timeline guides.

Descriptors are ordered dictionaries with a strict integer ``frame_index`` and
one or more ``image``, ``video``, ``audio`` paths. They are internal descriptors,
not the public upload manifest; callers must never accept client-supplied paths.
"""

from __future__ import annotations

import json
import math
import os
import selectors
import stat
import subprocess
import time
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

GUIDES_EXTRA_KEY = "_minimax_h3_timeline_guides"
_CONFIG_KEY = "minimax_h3_timeline_guides"
_FPS = 24
_AUDIO_RATE = 32000


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or (positive and value <= 0):
        raise ValueError(f"{name} must be a {'positive ' if positive else ''}strict integer")
    return value


@dataclass(frozen=True)
class TimelineGuideLimits:
    max_entries: int = 8
    max_unique_files: int = 16
    max_image_bytes: int = 30 * 1024 * 1024
    max_video_bytes: int = 50 * 1024 * 1024
    max_audio_bytes: int = 15 * 1024 * 1024
    max_total_upload_bytes: int = 128 * 1024 * 1024
    max_guide_rows: int = 65536
    max_packed_rows: int = 262144
    max_source_pixels: int = 16777216
    max_decoded_visual_pixels: int = 268435456
    max_decoded_audio_samples: int = 8388608
    subprocess_timeout_seconds: float = 60.0
    max_outstanding_requests: int = 4

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "subprocess_timeout_seconds":
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{field.name} must be finite and positive")
            else:
                _integer(value, field.name, positive=True)

    @classmethod
    def from_config(cls, model_config: Mapping[str, Any] | None) -> TimelineGuideLimits:
        """Read only the server's model_config block, never sampling extras."""
        if model_config is None:
            return cls()
        if not isinstance(model_config, Mapping):
            raise ValueError("model_config must be a mapping")
        block = model_config.get(_CONFIG_KEY, {})
        if not isinstance(block, Mapping):
            raise ValueError(f"{_CONFIG_KEY} must be a mapping")
        unknown = set(block) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"unknown timeline guide limits: {sorted(unknown, key=str)}")
        return cls(**block)


def normalize_visual_frame_count(M: int) -> int:  # noqa: N803 - native guide notation
    """Native Add Guide cadence, rounding down rather than output alignment."""
    _integer(M, "source visual frame count", positive=True)
    return 1 if M < 5 else 5 + 17 * ((M - 5) // 17)


def resolve_guide_start(index: int, output_frames: int, visual_frames: int = 1) -> int:
    """Resolve a pixel-frame index against the actual, already-aligned output."""
    _integer(index, "frame_index")
    _integer(output_frames, "output_frames", positive=True)
    _integer(visual_frames, "visual_frames", positive=True)
    start = output_frames + index if index < 0 else index
    if start < 0 or start + visual_frames > output_frames:
        raise ValueError("timeline guide must fit within the output; trim the source or change frame_index")
    return start


def guide_audio_limit(output_frames: int, start: int) -> int:
    """Available per-channel audio latent positions (not PCM scalar samples)."""
    _integer(start, "start")
    if start < 0:
        raise ValueError("start must be resolved and nonnegative")
    resolve_guide_start(start, output_frames)
    # Integer arithmetic preserves floor at fractional 5/3-frame origins.
    target_audio_t = (5 * output_frames + 1) // 3  # round(N * 40 / 24), without float precision loss
    available = (3 * target_audio_t - 5 * start) // 3
    if available < 1:
        raise ValueError("audio guide must leave at least one audio latent position")
    return available


def _file_size(path: str, modality: str, limits: TimelineGuideLimits) -> int:
    if not isinstance(path, str) or not path or "\x00" in path or "://" in path:
        raise ValueError(f"guide {modality} must be a trusted local file path")
    try:
        info = Path(path).stat()
    except OSError as exc:
        raise ValueError(f"cannot access guide {modality} file") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_size == 0:
        raise ValueError(f"guide {modality} must be a nonempty regular file")
    if info.st_size > getattr(limits, f"max_{modality}_bytes"):
        raise ValueError(f"guide {modality} exceeds upload byte limit; trim the source")
    return info.st_size


def validate_guide_descriptors(descriptors: Any, limits: TimelineGuideLimits) -> list[dict[str, Any]]:
    """Validate trusted paths and upload budgets, preserving every occurrence.

    This does not probe/encode sources or resolve output-dependent placement.
    Unique resolved paths count once for upload bytes; decoding counts per use.
    """
    if not isinstance(descriptors, list):
        raise ValueError("timeline guide descriptors must be a list")
    if len(descriptors) > limits.max_entries:
        raise ValueError("too many timeline guide entries")
    result = []
    files: dict[str, int] = {}
    for descriptor in descriptors:
        if not isinstance(descriptor, dict) or set(descriptor) - {"frame_index", "image", "video", "audio"}:
            raise ValueError("invalid timeline guide descriptor fields")
        _integer(descriptor.get("frame_index"), "frame_index")
        sources = set(descriptor) - {"frame_index"}
        if not sources or {"image", "video"} <= sources:
            raise ValueError("guide requires a source and cannot combine image with video")
        for modality in sources:
            path = descriptor[modality]
            size = _file_size(path, modality, limits)
            files[str(Path(path).resolve())] = size
        result.append(dict(descriptor))
    if len(files) > limits.max_unique_files:
        raise ValueError("too many unique timeline guide files")
    if sum(files.values()) > limits.max_total_upload_bytes:
        raise ValueError("timeline guides exceed aggregate upload byte limit")
    return result


@dataclass
class TimelineGuideBudget:
    """Per-request accounting. Charge each occurrence, including reused files.

    Video work is max(source frames, 24-FPS frames) * max(source pixels, canvas
    pixels), including frames discarded by FPS/cadence normalization. Decoders
    enforce both frame counts before returning; callers check packed row budgets.
    """

    limits: TimelineGuideLimits
    decoded_visual_pixels: int = 0
    decoded_audio_samples: int = 0
    guide_rows: int = 0

    def add_visual(self, pixel_frames: int) -> None:
        _integer(pixel_frames, "pixel_frames", positive=True)
        total = self.decoded_visual_pixels + pixel_frames
        if total > self.limits.max_decoded_visual_pixels:
            raise ValueError("timeline guides exceed aggregate decoded visual pixel budget; trim the sources")
        self.decoded_visual_pixels = total

    def add_audio(self, scalar_samples: int) -> None:
        _integer(scalar_samples, "scalar_samples", positive=True)
        total = self.decoded_audio_samples + scalar_samples
        if total > self.limits.max_decoded_audio_samples:
            raise ValueError("timeline guides exceed aggregate decoded audio sample budget; trim the sources")
        self.decoded_audio_samples = total

    def add_guide_rows(self, rows: int) -> None:
        _integer(rows, "guide rows", positive=True)
        if self.guide_rows + rows > self.limits.max_guide_rows:
            raise ValueError("timeline guides exceed guide row budget")
        self.guide_rows += rows

    def check_packed_rows(self, rows: int) -> None:
        _integer(rows, "packed rows", positive=True)
        if rows > self.limits.max_packed_rows:
            raise ValueError("timeline guides exceed entire packed request row budget")


def _run_bounded(
    command: list[str],
    limits: TimelineGuideLimits,
    max_bytes: int,
    *,
    on_stderr: Callable[[bytes], None] | None = None,
    max_stderr_bytes: int = 65536,
) -> bytes:
    """Drain both pipes with finite memory and wall time, killing/reaping on error."""
    try:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        raise ValueError(f"cannot start guide media tool {command[0]}") from exc
    stdout = bytearray()
    stderr = bytearray()
    stderr_received = 0
    deadline = time.monotonic() + limits.subprocess_timeout_seconds
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ, stdout)
            selector.register(process.stderr, selectors.EVENT_READ, stderr)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ValueError("timeline guide probe/decode timed out; trim the source")
                for key, _ in selector.select(remaining):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    buffer = key.data
                    if buffer is stderr:
                        stderr_received += len(chunk)
                        if stderr_received > max_stderr_bytes:
                            raise ValueError("timeline guide probe/decode output exceeds budget; trim the source")
                        if on_stderr is not None:
                            on_stderr(chunk)
                            continue
                    cap = max_bytes if buffer is stdout else max_stderr_bytes
                    if len(buffer) + len(chunk) > cap:
                        raise ValueError("timeline guide probe/decode output exceeds budget; trim the source")
                    buffer.extend(chunk)
            process.wait(timeout=max(0.001, deadline - time.monotonic()))
        if process.returncode:
            raise ValueError("timeline guide media probe/decode failed; check the media or trim the source")
        return bytes(stdout)
    except subprocess.TimeoutExpired as exc:
        raise ValueError("timeline guide probe/decode timed out; trim the source") from exc
    except OSError as exc:
        raise ValueError("timeline guide probe/decode I/O failed") from exc
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()
        process.stderr.close()


def _input_options(
    path: str, limits: TimelineGuideLimits, *, modality: str, decoder_pixels: int | None = None
) -> list[str]:
    # Reject playlist/network demuxers, and treat dash-prefixed paths as files.
    # max_pixels also covers codec scratch alignment, not just visible pixels.
    # Before dimensions are known, 128 * (P + 127) bounds 128-aligned W*H for
    # every positive W,H with W*H <= P. Visible admission is checked separately.
    video_options = []
    if modality == "video":
        if decoder_pixels is None:
            decoder_pixels = 128 * (limits.max_source_pixels + 127)
        # FFmpeg exposes this scratch-buffer limit as a signed 32-bit AVOption.
        video_options = ["-max_pixels", str(min(decoder_pixels, 2**31 - 1))]
    return [
        "-nofind_stream_info",
        "-max_alloc",
        str(max(5000000, limits.max_source_pixels * 16, limits.max_decoded_audio_samples * 8)),
        "-protocol_whitelist",
        "file,pipe",
        "-format_whitelist",
        "mov,matroska,avi,wav,mp3,flac",
        "-probesize",
        "5000000",
        "-analyzeduration",
        "5000000",
        *video_options,
        "-max_samples",
        # Permit a bounded codec block even when tests/admission allow fewer PCM samples.
        str(max(65536, limits.max_decoded_audio_samples)),
        "-threads",
        "1",
        "-i",
        str(Path(path).resolve()),
    ]


def _probe(path: str, modality: str, limits: TimelineGuideLimits) -> dict[str, Any]:
    _file_size(path, modality, limits)
    command = [
        "ffprobe",
        "-v",
        "error",
        *_input_options(path, limits, modality=modality),
        "-select_streams",
        "v:0" if modality == "video" else "a:0",
        "-show_entries",
        "stream=codec_type,width,height,channels,duration:format=duration",
        "-of",
        "json",
    ]
    try:
        info = json.loads(_run_bounded(command, limits, 65536))
        stream = info["streams"][0]
        if stream["codec_type"] != modality:
            raise ValueError("wrong media stream")
        if modality == "video":
            _pixels(int(stream["width"]), int(stream["height"]), limits)
        return stream
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"guide has no valid {modality} stream") from exc


def _pixels(width: int, height: int, limits: TimelineGuideLimits) -> int:
    _integer(width, "width", positive=True)
    _integer(height, "height", positive=True)
    pixels = width * height
    if pixels > limits.max_source_pixels:
        raise ValueError("guide source/canvas dimensions exceed pixel limit; resize the source")
    return pixels


def decode_guide_image(
    path: str, width: int, height: int, limits: TimelineGuideLimits, *, budget: TimelineGuideBudget | None = None
) -> Image.Image:
    """Decode a native raster still, orient it, and center-crop to the RGB canvas.

    Delegate-backed formats such as EPS are deliberately excluded: their external
    tools would bypass the subprocess timeout and output-memory bounds.
    """
    _file_size(path, "image", limits)
    canvas_pixels = _pixels(width, height, limits)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path, formats=("JPEG", "PNG", "WEBP", "BMP", "TIFF", "GIF")) as source:
                work = max(_pixels(*source.size, limits), canvas_pixels)
                if work > limits.max_decoded_visual_pixels:
                    raise ValueError("guide image exceeds decoded visual pixel budget")
                if budget is not None:
                    budget.add_visual(work)
                # Do not scan all frames merely to reject an animated source.
                try:
                    source.seek(1)
                except EOFError:
                    source.seek(0)
                else:
                    raise ValueError("guide image must be a still image; use a video guide for clips")
                image = ImageOps.exif_transpose(source).convert("RGB")
                return ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
    except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("invalid or oversized guide image") from exc


def decode_guide_video(
    path: str,
    width: int,
    height: int,
    output_frames: int,
    limits: TimelineGuideLimits,
    *,
    budget: TimelineGuideBudget | None = None,
) -> list[Image.Image]:
    """PTS-based 24 FPS decode, center crop, then native downward normalization.

    The bound is the whole output, not the remaining space after guide placement.
    A sentinel frame detects oversized sources even without duration metadata.
    """
    _integer(output_frames, "output_frames", positive=True)
    canvas_pixels = _pixels(width, height, limits)
    meta = _probe(path, "video", limits)
    try:
        source_width, source_height = int(meta["width"]), int(meta["height"])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("cannot determine guide video dimensions") from exc
    source_pixels = _pixels(source_width, source_height, limits)
    raw_cap = 4 if output_frames < 5 else normalize_visual_frame_count(output_frames) + 16
    remaining_pixels = limits.max_decoded_visual_pixels
    if budget is not None:
        remaining_pixels = min(remaining_pixels, budget.limits.max_decoded_visual_pixels - budget.decoded_visual_pixels)
    source_cap = remaining_pixels // max(source_pixels, canvas_pixels)
    raw_cap = min(raw_cap, source_cap)
    if raw_cap < 1:
        raise ValueError("guide video exceeds decoded visual pixel budget")
    duration = meta.get("duration")
    if duration not in (None, "N/A", ""):
        try:
            seconds = float(duration)
        except (ValueError, TypeError) as exc:
            raise ValueError("invalid guide video duration") from exc
        if not math.isfinite(seconds) or seconds <= 0 or seconds * _FPS > raw_cap + 1e-5:
            raise ValueError("guide video exceeds duration/frame budget; trim the source")
    # Crop before scaling so extreme aspect ratios never allocate a huge intermediate.
    if source_width * height > source_height * width:
        crop_width, crop_height = max(1, source_height * width // height), source_height
    else:
        crop_width, crop_height = source_width, max(1, source_width * height // width)
    # Decoder buffers need codec alignment; reject actual size changes in the
    # crop filter, while allowing bounded scratch padding around the probed frame.
    decoder_pixels = ((source_width + 127) // 128 * 128) * ((source_height + 127) // 128 * 128)
    filters = (
        # Trim bounds source decode even if the caller has not yet drained stderr.
        f"trim=end_frame={source_cap + 1},"
        f"crop=w='if(eq(iw,{source_width})*eq(ih,{source_height}),{crop_width},0)':h={crop_height}:exact=1,"
        "metadata=mode=add:key=minimax_h3_frame:value=1,"
        "metadata=mode=print:key=minimax_h3_frame:file='pipe\\:2':direct=1,"
        "setpts=PTS-STARTPTS,fps=24:eof_action=pass,"
        f"scale={width}:{height}:flags=lanczos,setsar=1"
    )
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-xerror",
        "-err_detect",
        "explode",
        "-noautorotate",
        *_input_options(path, limits, modality="video", decoder_pixels=decoder_pixels),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        filters,
        "-filter_threads",
        "1",
        "-frames:v",
        str(raw_cap + 1),
        "-threads",
        "1",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    frame_bytes = canvas_pixels * 3
    source_frames = 0
    pending = b""

    def count_source_frames(chunk: bytes) -> None:
        nonlocal source_frames, pending
        lines = (pending + chunk).split(b"\n")
        pending = lines.pop()
        if len(pending) > 65536:
            raise ValueError("timeline guide decode metadata exceeds budget")
        for line in lines:
            if line.startswith(b"frame:"):
                source_frames += 1
                if source_frames > source_cap:
                    raise ValueError("guide video exceeds source-frame pixel budget; trim the source")

    raw = _run_bounded(
        command,
        limits,
        (raw_cap + 1) * frame_bytes,
        on_stderr=count_source_frames,
        max_stderr_bytes=65536 + (source_cap + 1) * 256,
    )
    count, remainder = divmod(len(raw), frame_bytes)
    if remainder or count > raw_cap:
        raise ValueError("guide video exceeds frame/pixel budget or is incomplete; trim the source")
    normalized = normalize_visual_frame_count(count)
    if source_frames == 0:
        raise ValueError("guide video decode did not report source frames")
    if budget is not None:
        budget.add_visual(max(count, source_frames) * max(source_pixels, canvas_pixels))
    return [
        Image.frombytes("RGB", (width, height), raw[i * frame_bytes : (i + 1) * frame_bytes]) for i in range(normalized)
    ]


def decode_guide_audio(
    path: str, limits: TimelineGuideLimits, *, budget: TimelineGuideBudget | None = None
) -> tuple[np.ndarray, int]:
    """Decode WAV/MP3/FLAC (or explicit container audio) to stereo 32 kHz PCM.

    Mono is duplicated and multichannel audio is downmixed by ffmpeg. There is no
    reference-audio minimum duration, implicit video soundtrack, or output crop.
    """
    _probe(path, "audio", limits)
    # Stereo scalar samples are the admission unit, independent of source layout.
    remaining_samples = limits.max_decoded_audio_samples
    if budget is not None:
        remaining_samples = min(
            remaining_samples, budget.limits.max_decoded_audio_samples - budget.decoded_audio_samples
        )
    sample_frames = remaining_samples // 2
    if sample_frames < 1:
        raise ValueError("guide audio exceeds decoded sample budget")
    # Container duration can include codec delay/padding (notably MP3). Only
    # bounded, resampled PCM counts decide admission; metadata is not authoritative.
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-xerror",
        "-err_detect",
        "explode",
        *_input_options(path, limits, modality="audio"),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "2",
        "-ar",
        str(_AUDIO_RATE),
        "-af",
        "asetpts=N/SR/TB",
        "-filter_threads",
        "1",
        "-t",
        # ffmpeg duration syntax rejects scientific notation for tiny budgets.
        f"{(sample_frames + 1) / _AUDIO_RATE:.8f}",
        "-threads",
        "1",
        "-f",
        "f32le",
        "pipe:1",
    ]
    raw = _run_bounded(command, limits, (sample_frames + 1) * 2 * 4)
    if not raw or len(raw) % 8:
        raise ValueError("guide audio is empty or incomplete")
    if len(raw) // 4 > remaining_samples:
        raise ValueError("guide audio exceeds decoded sample budget; trim the source")
    if budget is not None:
        budget.add_audio(len(raw) // 4)
    audio = np.frombuffer(raw, dtype="<f4").reshape(-1, 2).T.copy()
    if not np.isfinite(audio).all():
        raise ValueError("guide audio contains nonfinite samples")
    return audio, _AUDIO_RATE
