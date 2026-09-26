# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Decode restoration inputs before scheduler admission."""

import functools
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch

from vllm_omni.diffusion import envs
from vllm_omni.errors import OmniClientError

# Bound per-frame decode bookkeeping; the padded clip-pixel budget is tighter
# for normal video resolutions.
MAX_FRAME_PIXELS = 848 * 480
MAX_CLIP_PIXELS = 5 * MAX_FRAME_PIXELS


def max_frames() -> int:
    """Decoder-work frame cap, shared by every serving profile."""
    return envs.VLLM_OMNI_SEEDVR2_MAX_FRAMES


# The sharded clip budget validated on the smallest qualified device (32 GB, SP4):
# five 2560x1472 frames.
CALIBRATED_SHARDED_CLIP_PIXELS = 5 * 2560 * 1472
# The largest clip run through the forward at SP4 on a 268 GiB B300 (513 frames
# at 1536x2688, 55.5 GiB peak activations). Past it the DiT's fp32 RoPE
# intermediates and allocator fragmentation outgrow the model below, so the
# scaled budget stops here. It also keeps one channel of a clip under 2**31
# elements.
VALIDATED_SHARDED_CLIP_PIXELS = 513 * 1536 * 2688
# Peak activation bytes per rank of one sharded forward at SP4, the smallest
# sharded degree (more ranks need less). Measured on B300 from 5 to 1,921 frames
# at 768x1344, 2560x1472 and 1536x2688, weights excluded:
#   max(ENCODER * frame_pixels * min(frames, 9) + CLIP * clip_pixels,
#       DECODED * clip_pixels)
# The VAE encoder's first temporal chunk (up to nine frames) peaks for short
# clips; the whole-clip fp16 input and gathered decode peak for long ones.
ENCODER_CHUNK_FRAMES = 9
ENCODER_BYTES_PER_PIXEL = 1140
CLIP_BYTES_PER_PIXEL = 8.5
DECODED_BYTES_PER_PIXEL = 28
WEIGHT_BYTES = 7 * 1024**3
# Room for the CUDA context and communicator buffers, then for allocator
# fragmentation, which reached 37% of the allocated activations near OOM.
USABLE_MEMORY_FRACTION = 0.85
FRAGMENTATION = 1.37


@functools.cache
def _device_memory() -> int | None:
    """Smallest total memory among the visible accelerators, if it can be read."""
    from vllm_omni.platforms import current_omni_platform

    try:
        count = torch.accelerator.device_count()
        return min(int(current_omni_platform.get_device_total_memory(index)) for index in range(count)) or None
    except (AttributeError, NotImplementedError, RuntimeError, ValueError):
        return None


def scaled_clip_pixels(frame_pixels: int, device_memory: int | None) -> int:
    """Largest padded clip the memory model admits on ``device_memory``.

    The result never falls below the calibrated budget or rises above the
    validated one. The worst case for a clip budget is a clip of full-size
    frames, so the model is evaluated at ``frame_pixels``; smaller frames only
    need less.
    """
    if device_memory is None:
        return CALIBRATED_SHARDED_CLIP_PIXELS
    budget = (device_memory * USABLE_MEMORY_FRACTION - WEIGHT_BYTES) / FRAGMENTATION
    chunk = ENCODER_CHUNK_FRAMES * frame_pixels
    clip = budget / (ENCODER_BYTES_PER_PIXEL + CLIP_BYTES_PER_PIXEL)
    if clip > chunk:
        clip = min(
            (budget - ENCODER_BYTES_PER_PIXEL * chunk) / CLIP_BYTES_PER_PIXEL,
            budget / DECODED_BYTES_PER_PIXEL,
        )
    return min(max(CALIBRATED_SHARDED_CLIP_PIXELS, int(clip)), VALIDATED_SHARDED_CLIP_PIXELS)


def sharded_budget() -> tuple[int, int]:
    """Per-frame and padded-clip budgets for the VAE-sharded serving profile.

    An unset clip budget scales with the smallest visible device's memory.
    """
    frame_pixels = envs.VLLM_OMNI_SEEDVR2_SHARDED_FRAME_PIXELS
    clip_pixels = envs.VLLM_OMNI_SEEDVR2_SHARDED_CLIP_PIXELS
    if clip_pixels is None:
        clip_pixels = scaled_clip_pixels(frame_pixels, _device_memory())
    return frame_pixels, clip_pixels


def validate_clip_size(frame_count: int, height: int, width: int, frame_pixels: int, clip_pixels: int) -> None:
    padded_frames = frame_count + (1 - frame_count) % 4
    if frame_count > max_frames() or height * width > frame_pixels or padded_frames * height * width > clip_pixels:
        raise OmniClientError("SeedVR2 clip exceeds the configured frame or pixel budget")


@dataclass(frozen=True)
class SourceVideo:
    fps: float
    pts: tuple[int, ...]
    time_base: Fraction
    audio: torch.Tensor | None
    audio_sample_rate: int | None


def read_video(
    path: str | Path, frame_pixels: int = MAX_FRAME_PIXELS, clip_pixels: int = MAX_CLIP_PIXELS
) -> tuple[torch.Tensor, SourceVideo]:
    try:
        return _read_video(path, frame_pixels, clip_pixels)
    except av.FFmpegError as exc:
        raise OmniClientError(f"Cannot decode SeedVR2 input video: {exc.strerror}") from exc


def _read_video(path: str | Path, frame_pixels: int, clip_pixels: int) -> tuple[torch.Tensor, SourceVideo]:
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise OmniClientError("SeedVR2 input contains no video stream")
        stream = container.streams.video[0]
        rate = stream.average_rate
        if rate is None or rate <= 0:
            raise OmniClientError("SeedVR2 input must declare a positive frame rate")
        height, width = stream.codec_context.height, stream.codec_context.width
        if height > 0 and width > 0:
            validate_clip_size(max(1, stream.frames), height, width, frame_pixels, clip_pixels)
        if stream.duration is not None and stream.time_base is not None:
            if stream.duration * stream.time_base * rate > max_frames():
                raise OmniClientError("SeedVR2 input exceeds the maximum duration")
        frames = []
        pts = []
        for frame in container.decode(stream):
            validate_clip_size(len(frames) + 1, frame.height, frame.width, frame_pixels, clip_pixels)
            if frame.pts is None:
                raise OmniClientError("SeedVR2 input video is missing frame timestamps")
            frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
            pts.append(frame.pts)
        if not frames:
            raise OmniClientError("SeedVR2 input video contains no decoded frames")
        time_base = Fraction(stream.time_base)
        if any(b <= a for a, b in zip(pts, pts[1:])):
            raise OmniClientError("SeedVR2 input timestamps must increase")
        if any(
            abs((pts[index] - pts[0]) * time_base - Fraction(index, 1) / rate) > time_base for index in range(len(pts))
        ):
            raise OmniClientError("SeedVR2 currently supports constant-frame-rate video only")
    audio = None
    sample_rate = None
    with av.open(str(path)) as container:
        if container.streams.audio:
            stream = container.streams.audio[0]
            sample_rate = stream.codec_context.sample_rate
            resampler = av.AudioResampler(format="fltp", layout=stream.codec_context.layout, rate=sample_rate)
            channels = len(stream.codec_context.layout.channels)
            if channels not in (1, 2):
                raise OmniClientError("SeedVR2 audio preservation currently supports mono or stereo")
            sample_count = round(len(frames) / rate * sample_rate)
            waveform = np.zeros((channels, sample_count), dtype=np.float32)

            def resampled_frames() -> Iterator[av.AudioFrame]:
                for frame in container.decode(stream):
                    yield from resampler.resample(frame)
                yield from resampler.resample(None)

            for frame in resampled_frames():
                if frame.pts is None:
                    raise OmniClientError("SeedVR2 audio is missing timestamps")
                offset = round((frame.pts * frame.time_base - pts[0] * time_base) * sample_rate)
                if offset >= sample_count:
                    break
                chunk = frame.to_ndarray()
                start, stop = max(0, offset), min(sample_count, offset + chunk.shape[1])
                if stop > start:
                    waveform[:, start:stop] = chunk[:, start - offset : stop - offset]
            audio = torch.from_numpy(waveform).unsqueeze(0)
    # Frames stay uint8 THWC: the device converts them, so the host never builds
    # (or ships to every rank) a float copy four times the size.
    return torch.stack(frames), SourceVideo(
        float(rate),
        tuple(pts),
        time_base,
        audio,
        sample_rate,
    )
