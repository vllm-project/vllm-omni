# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SeedVR2 restoration pipeline: input admission, one Euler step and colour transfer."""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from pathlib import Path
from typing import ClassVar

import av
import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F
from vllm.utils.torch_utils import set_torch_threads_for_runtime

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.seedvr2 import config as envs
from vllm_omni.diffusion.models.seedvr2.config import COLOR_CORRECTION_METHODS, DEFAULT_COLOR_CORRECTION_METHOD
from vllm_omni.diffusion.models.seedvr2.nadit import SEEDVR2_3B_CONFIG, SeedVR2NaDiT, validate_seedvr2_parallel_config
from vllm_omni.diffusion.models.seedvr2.vae import SeedVR2VAE
from vllm_omni.diffusion.offloader.config import offload_enabled
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

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


# Decomposition depth: the fifth band already isolates the illumination tint.
WAVELET_LEVELS = 5
# Restoration luminance is the reason to run the model, so chroma is matched
# fully while luminance keeps this much of the restored value.
DEFAULT_LUMINANCE_WEIGHT = 0.8

# CIE 15 lightness transfer, sRGB (IEC 61966-2-1) primaries, D65 white point.
_CIELAB_DELTA = 6.0 / 29.0
_D65_WHITE = (0.95047, 1.0, 1.08883)
_SRGB_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)
_XYZ_TO_SRGB = (
    (3.2404542, -1.5371385, -0.4985314),
    (-0.9692660, 1.8760108, 0.0415560),
    (0.0556434, -0.2040259, 1.0572252),
)


def _binomial_lowpass(image: Tensor, radius: int) -> Tensor:
    """One a-trous low-pass step with the separable 1-2-1 binomial kernel.

    The dilated taps run as shifted weighted sums rather than a dilated
    conv2d. On NPU workers the dilated conv2d routes through lazy TBE
    compilation, whose helper ``multiprocessing.Manager`` cannot start inside
    a spawned worker and aborts the process; the shift-add form is exactly
    equivalent for the fixed 1-2-1 kernel and compiles everywhere.
    """
    height, width = image.shape[-2:]
    padded = F.pad(image, (radius,) * 4, mode="replicate")

    def axis_lowpass(x: Tensor, size: int, dim: int) -> Tensor:
        low, mid, high = (x.narrow(dim, offset, size) for offset in (0, radius, 2 * radius))
        return (low + 2 * mid + high) / 4.0

    return axis_lowpass(axis_lowpass(padded, height, -2), width, -1)


def _lowest_band(image: Tensor, levels: int = WAVELET_LEVELS) -> Tensor:
    """Residual band after ``levels`` doubling-radius low-pass steps.

    Summing the per-level high-frequency residuals telescopes to
    ``image - lowest_band(image)``, so only this band has to be materialized.
    """
    # Keep a dilated tap well inside the frame so it samples real signal rather
    # than replicated edges; this bound matches the reference decomposition.
    limit = max(1, min(image.shape[-2:]) // 8)
    for level in range(levels):
        image = _binomial_lowpass(image, min(1 << level, limit))
    return image


def _match_histogram(source: Tensor, reference: Tensor) -> Tensor:
    """Give ``source`` the value distribution of ``reference``, rank for rank.

    Both channels hold the same pixel count, so ranks map one to one and the
    match is exact without quantile interpolation.
    """
    order = source.flatten().argsort()
    matched = torch.empty_like(order, dtype=reference.dtype)
    matched[order] = reference.flatten().sort().values
    return matched.view_as(source)


def _srgb_to_lab(rgb: Tensor) -> Tensor:
    linear = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055).pow(2.4), rgb / 12.92)
    matrix = rgb.new_tensor(_SRGB_TO_XYZ)
    xyz = torch.einsum("ij,njhw->nihw", matrix, linear)
    xyz = xyz / xyz.new_tensor(_D65_WHITE).view(1, 3, 1, 1)
    scaled = torch.where(
        xyz > _CIELAB_DELTA**3,
        xyz.clamp_min(0.0).pow(1.0 / 3.0),
        xyz / (3.0 * _CIELAB_DELTA**2) + 4.0 / 29.0,
    )
    x, y, z = scaled[:, 0], scaled[:, 1], scaled[:, 2]
    return torch.stack([116.0 * y - 16.0, 500.0 * (x - y), 200.0 * (y - z)], dim=1)


def _lab_to_srgb(lab: Tensor) -> Tensor:
    lightness, chroma_a, chroma_b = lab[:, 0], lab[:, 1], lab[:, 2]
    y = (lightness + 16.0) / 116.0
    scaled = torch.stack([y + chroma_a / 500.0, y, y - chroma_b / 200.0], dim=1)
    xyz = torch.where(
        scaled > _CIELAB_DELTA,
        scaled.pow(3.0),
        3.0 * _CIELAB_DELTA**2 * (scaled - 4.0 / 29.0),
    )
    xyz = xyz * xyz.new_tensor(_D65_WHITE).view(1, 3, 1, 1)
    linear = torch.einsum("ij,njhw->nihw", lab.new_tensor(_XYZ_TO_SRGB), xyz)
    rgb = torch.where(
        linear > 0.0031308,
        1.055 * linear.clamp_min(0.0).pow(1.0 / 2.4) - 0.055,
        12.92 * linear,
    )
    return rgb.clamp_(0.0, 1.0)


def _wavelet_transfer(restored: Tensor, reference: Tensor) -> Tensor:
    return (restored - _lowest_band(restored) + _lowest_band(reference)).clamp_(0.0, 1.0)


def _channel_stats(image: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    dims = (2, 3)
    deviation = (image.var(dims, correction=0, keepdim=True) + eps).sqrt()
    return image.mean(dims, keepdim=True), deviation


def _adain_transfer(restored: Tensor, reference: Tensor, eps: float = 1e-5) -> Tensor:
    restored_mean, restored_std = _channel_stats(restored, eps)
    reference_mean, reference_std = _channel_stats(reference, eps)
    normalized = (restored - restored_mean) / restored_std
    return (normalized * reference_std + reference_mean).clamp_(0.0, 1.0)


def _lab_transfer(restored: Tensor, reference: Tensor, luminance_weight: float) -> Tensor:
    restored_lab = _srgb_to_lab(_wavelet_transfer(restored, reference))
    reference_lab = _srgb_to_lab(reference)
    lightness = restored_lab[:, 0]
    if luminance_weight < 1.0:
        matched = _match_histogram(lightness, reference_lab[:, 0])
        lightness = lightness * luminance_weight + matched * (1.0 - luminance_weight)
    corrected = torch.stack(
        [
            lightness,
            _match_histogram(restored_lab[:, 1], reference_lab[:, 1]),
            _match_histogram(restored_lab[:, 2], reference_lab[:, 2]),
        ],
        dim=1,
    )
    return _lab_to_srgb(corrected)


def correct_video_color(
    video: Tensor,
    reference: Tensor,
    method: str = DEFAULT_COLOR_CORRECTION_METHOD,
    luminance_weight: float = DEFAULT_LUMINANCE_WEIGHT,
) -> Tensor:
    """Transfer ``reference`` colour onto ``video``; both are ``[B,C,T,H,W]`` in ``[0,1]``.

    Frames are processed one at a time because histogram matching sorts every
    pixel of a channel. Inputs are never modified.
    """
    if method not in COLOR_CORRECTION_METHODS:
        raise ValueError(f"SeedVR2 color_correction_method must be one of {list(COLOR_CORRECTION_METHODS)}: {method!r}")
    if method == "none":
        return video
    if video.shape != reference.shape:
        raise ValueError(f"SeedVR2 color correction needs matching shapes, got {video.shape} and {reference.shape}")
    if not 0.0 <= luminance_weight <= 1.0:
        raise ValueError(f"SeedVR2 luminance_weight must be within [0, 1]: {luminance_weight!r}")

    # Colour math is scale sensitive, so it runs in fp32 regardless of the model dtype.
    frames = video.movedim(2, 0).flatten(0, 1).float()
    references = reference.movedim(2, 0).flatten(0, 1).float()
    corrected = torch.empty_like(frames)
    for index in range(frames.shape[0]):
        frame, guide = frames[index : index + 1], references[index : index + 1]
        if method == "wavelet":
            corrected[index : index + 1] = _wavelet_transfer(frame, guide)
        elif method == "adain":
            corrected[index : index + 1] = _adain_transfer(frame, guide)
        else:
            corrected[index : index + 1] = _lab_transfer(frame, guide, luminance_weight)
    corrected = corrected.unflatten(0, (video.shape[2], video.shape[0])).movedim(0, 2)
    return corrected.to(dtype=video.dtype)


def validate_seedvr2_config(config: OmniDiffusionConfig) -> None:
    # Read the operator-tuned budgets here so a bad value is a startup error
    # rather than a traceback from whichever module imports them first.
    max_frames()
    sharded_budget()
    if config.dtype != torch.float16:
        raise ValueError("SeedVR2 3B requires dtype=float16")
    if not config.enforce_eager:
        raise ValueError("SeedVR2 requires enforce_eager=True; compiled execution is not supported")
    if config.cache_backend != "none" or config.quantization_config is not None:
        raise ValueError("SeedVR2 requires unquantized weights and cache_backend=none for its single Euler step")
    if config.vae_use_slicing:
        raise ValueError("SeedVR2 whole-clip VAE does not support batch slicing")
    if offload_enabled(config):
        raise ValueError("SeedVR2 native whole-clip execution does not support CPU offload")
    parallel = config.parallel_config
    validate_seedvr2_parallel_config(parallel)
    if parallel.data_parallel_size is not None and parallel.data_parallel_size > 1:
        raise ValueError("SeedVR2 does not support data_parallel_size > 1")
    if parallel.ulysses_degree > 20:
        raise ValueError("SeedVR2 3B Ulysses requires at least one of its 20 heads per rank")
    degrees = {
        "cfg_parallel_size": parallel.cfg_parallel_size,
        "tensor_parallel_size": parallel.tensor_parallel_size,
        "pipeline_parallel_size": parallel.pipeline_parallel_size,
        "text_encoder_tp_size": parallel.text_encoder_tp_size,
        "ring_degree": parallel.ring_degree,
        "allgather_degree": parallel.allgather_degree,
    }
    for name, degree in degrees.items():
        if degree != 1:
            raise ValueError(f"SeedVR2 requires {name}=1; use ulysses_degree for its model-owned SP")
    if parallel.vae_patch_parallel_size > 1 and parallel.vae_patch_parallel_size != parallel.ulysses_degree:
        raise ValueError("SeedVR2 requires vae_patch_parallel_size to match ulysses_degree")
    if parallel.vae_parallel_mode == "spatial_shard_width":
        raise ValueError("SeedVR2 VAE supports spatial_shard_height or tile mode")
    if parallel.use_hsdp or parallel.enable_expert_parallel:
        raise ValueError("SeedVR2 does not support HSDP or expert parallelism")
    if config.lora_path is not None:
        raise ValueError("SeedVR2 does not support LoRA adapters")


def _check_frames(frames: torch.Tensor, height: int, width: int) -> None:
    """Validate host frames: uint8 ``[T,H,W,3]`` or float ``[T,3,H,W]`` in ``[0,1]``."""
    rgb_axis = 3 if frames.dtype == torch.uint8 else 1
    if frames.ndim != 4 or frames.shape[rgb_axis] != 3 or min(frames.shape) < 1:
        raise OmniClientError("SeedVR2 requires nonempty RGB frames with shape [T,3,H,W]")
    if min(height, width) < 16 or height % 16 or width % 16:
        raise OmniClientError("SeedVR2 output dimensions must be positive multiples of 16")
    if frames.dtype == torch.uint8:
        return
    if not frames.is_floating_point() or not torch.isfinite(frames).all():
        raise OmniClientError("SeedVR2 frames must be finite floating-point RGB values")
    if torch.any(frames < 0) or torch.any(frames > 1):
        raise OmniClientError("SeedVR2 RGB values must be in [0,1]")


def prepare_video(frames: torch.Tensor, height: int, width: int, device: torch.device) -> torch.Tensor:
    """Resize RGB frames to the model grid on ``device`` and repeat the final frame.

    ``frames`` is uint8 ``[T,H,W,3]`` or float ``[T,3,H,W]`` in ``[0,1]``. Frames
    are converted one at a time, so the device never holds a float copy of the
    whole clip, and returns fp16 ``[1,3,T',H,W]`` in ``[-1,1]`` with ``T' = 4n+1``.
    """
    count = frames.shape[0]
    sample = torch.empty((1, 3, count + (1 - count) % 4, height, width), device=device, dtype=torch.float16)
    for index in range(count):
        frame = frames[index].to(device, non_blocking=True)
        frame = frame.permute(2, 0, 1).float() / 255 if frame.dtype == torch.uint8 else frame.float()
        if frame.shape[-2:] != (height, width):
            frame = F.interpolate(
                frame.unsqueeze(0), size=(height, width), mode="bicubic", align_corners=False, antialias=True
            ).squeeze(0)
        sample[0, :, index] = frame.clamp(0, 1) * 2 - 1
    sample[0, :, count:] = sample[0, :, count - 1 : count]
    return sample


@dataclass(frozen=True)
class SeedVR2Input:
    # uint8 [T,H,W,3] for decoded video and PIL frames, float [T,3,H,W] otherwise;
    # resizing and normalization run on the device in ``prepare_video``.
    frames: torch.Tensor
    frame_count: int
    source: SourceVideo | None = None


def _admission_budget(config: OmniDiffusionConfig) -> tuple[int, int]:
    """Pick the budget for the serving profile; more ranks never admit less."""
    parallel = config.parallel_config
    sharded = parallel.ulysses_degree == parallel.vae_patch_parallel_size
    if config.vae_use_tiling and sharded and parallel.ulysses_degree >= 4:
        return sharded_budget()
    return MAX_FRAME_PIXELS, MAX_CLIP_PIXELS


def prepare_request(
    request: OmniDiffusionRequest,
    *,
    frame_pixels: int = MAX_FRAME_PIXELS,
    clip_pixels: int = MAX_CLIP_PIXELS,
) -> OmniDiffusionRequest:
    params = request.sampling_params
    if request.is_dummy_run():
        # The engine's generic warmup has no source video. Only internal warmup
        # requests may synthesize conditioning; real requests stay strict.
        request.prompt = {"prompt": "", "multi_modal_data": {"video": torch.zeros(1, 3, params.height, params.width)}}
        params.num_frames = 1
        params.num_inference_steps = 1
        params.guidance_scale = 1.0
    if params.num_inference_steps not in (None, 1) or params.guidance_scale != 1.0:
        raise OmniClientError("SeedVR2 whole-clip restoration requires one Euler step and guidance_scale=1")
    method = params.extra_args.get("color_correction_method")
    if method is not None and method not in COLOR_CORRECTION_METHODS:
        raise OmniClientError(f"SeedVR2 color_correction_method must be one of {list(COLOR_CORRECTION_METHODS)}")
    prompt = request.prompt
    if not isinstance(prompt, dict) or "multi_modal_data" not in prompt:
        raise OmniClientError("SeedVR2 requires multi_modal_data.video")
    if params.height is None or params.width is None:
        raise OmniClientError("SeedVR2 requires explicit output height and width")
    if min(params.height, params.width) < 16 or params.height % 16 or params.width % 16:
        raise OmniClientError("SeedVR2 output dimensions must be positive multiples of 16")
    validate_clip_size(1, params.height, params.width, frame_pixels, clip_pixels)
    frames = prompt["multi_modal_data"].get("video")
    source = None
    if isinstance(frames, list) and len(frames) == 1 and isinstance(frames[0], (str, Path)):
        frames = frames[0]
    if isinstance(frames, (str, Path)):
        frames, source = read_video(frames, frame_pixels, clip_pixels)
        if params.fps is not None and abs(params.fps - source.fps) > 1e-6:
            raise OmniClientError("SeedVR2 preserves source FPS; omit fps or match the input frame rate")
    if isinstance(frames, list) and frames and all(isinstance(frame, Image.Image) for frame in frames):
        for frame in frames:
            validate_clip_size(len(frames), frame.height, frame.width, frame_pixels, clip_pixels)
        frames = torch.stack([torch.from_numpy(np.array(frame.convert("RGB"))) for frame in frames])
    if not isinstance(frames, torch.Tensor):
        raise OmniClientError("SeedVR2 video must be a TCHW RGB tensor or a list of PIL frames")
    if frames.ndim == 4:
        frame_height, frame_width = frames.shape[1:3] if frames.dtype == torch.uint8 else frames.shape[2:]
        validate_clip_size(frames.shape[0], frame_height, frame_width, frame_pixels, clip_pixels)
        validate_clip_size(frames.shape[0], params.height, params.width, frame_pixels, clip_pixels)
    _check_frames(frames, params.height, params.width)
    if (prompt.get("prompt") or "").strip():
        raise OmniClientError("SeedVR2 uses fixed checkpoint conditioning; prompt text is unsupported")
    if params.num_outputs_per_prompt != 1:
        raise OmniClientError("SeedVR2 produces one restored video per request")
    request.prepared_layout = SeedVR2Input(frames, frames.shape[0], source)
    return request


def get_seedvr2_pre_process_func(
    od_config: OmniDiffusionConfig,
) -> Callable[[OmniDiffusionRequest], OmniDiffusionRequest]:
    # This factory runs in the serving process. Keep SeedVR2's CPU scheduling
    # policy local, and respect an explicit OMP_NUM_THREADS override.
    set_torch_threads_for_runtime()
    frame_pixels, clip_pixels = _admission_budget(od_config)
    return partial(prepare_request, frame_pixels=frame_pixels, clip_pixels=clip_pixels)


def _seedvr2_post_process(output: dict[str, object]) -> dict[str, object]:
    payload = output["payload"]
    assert isinstance(payload, dict)
    video = payload["video"]
    assert isinstance(video, torch.Tensor) and video.dtype == torch.uint8
    # The device already produced uint8 [B,T,H,W,3], the layout encoders take.
    return {"payload": {**payload, "video": video.cpu().numpy()}, "metadata": output["metadata"]}


def get_seedvr2_post_process_func(_od_config: OmniDiffusionConfig) -> Callable[[dict[str, object]], dict[str, object]]:
    return _seedvr2_post_process


def finish_video(decoded: torch.Tensor, sample: torch.Tensor, frame_count: int, method: str) -> torch.Tensor:
    """Colour-correct and quantize decoded ``[B,3,T,H,W]`` frames to uint8 ``[B,T,H,W,3]``.

    One frame at a time: whole-clip float copies would dominate device memory on
    long clips, and a per-frame slice of one would need 64-bit indexing once the
    clip passes 2**31 pixels.
    """
    batch_size, _, _, height, width = decoded.shape
    video = torch.empty((batch_size, frame_count, height, width, 3), device=decoded.device, dtype=torch.uint8)
    for index in range(frame_count):
        frame = slice(index, index + 1)
        # Restoration shifts global colour, so the resized input carries the
        # reference colour back onto the restored detail.
        corrected = correct_video_color(
            ((decoded[:, :, frame].float() + 1) / 2).clamp(0, 1),
            ((sample[:, :, frame].float() + 1) / 2).clamp(0, 1),
            method=method,
        )
        # Quantize on the device, matching the encoders' rint(clip(x) * 255), so
        # the host copy and the IPC payload are a quarter of the float size.
        video[:, index] = (corrected[:, :, 0].clamp(0, 1) * 255).round_().permute(0, 2, 3, 1)
    return video


def sample_noise(condition: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    # Reference randn_like preserves latent strides; RNG values follow storage order.
    return torch.empty_like(condition).normal_(generator=generator)


class SeedVR2Pipeline(nn.Module):
    supports_request_batch = True
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _encoder_modules: ClassVar[list[str]] = []

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        validate_seedvr2_config(od_config)
        set_torch_threads_for_runtime()
        self.device = get_local_device()
        self.od_config = od_config
        self.frame_pixels, self.clip_pixels = _admission_budget(od_config)
        self.transformer = SeedVR2NaDiT(**SEEDVR2_3B_CONFIG, use_varlen_kernel=False)
        self.vae = SeedVR2VAE()
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder=None,
                revision=od_config.revision,
                prefix=component + ".",
                fall_back_to_pt=False,
                allow_patterns_overrides=[filename],
            )
            for component, filename in (
                ("transformer", "seedvr2_ema_3b_fp16.safetensors"),
                ("vae", "ema_vae_fp16.safetensors"),
            )
        ]
        model_path = Path(od_config.model)
        if not model_path.is_dir():
            model_path = Path(
                download_weights_from_hf_specific(od_config.model, None, ["pos_emb.pt"], revision=od_config.revision)
            )
        text = torch.load(model_path / "pos_emb.pt", map_location="cpu", weights_only=True)
        if not isinstance(text, torch.Tensor) or text.ndim != 2 or text.shape[1] != 5120 or text.shape[0] == 0:
            raise ValueError("SeedVR2 pos_emb.pt must contain a tensor of shape [L,5120]")
        self.register_buffer("text", text.to(device=self.device, dtype=od_config.dtype), persistent=False)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        targets = self.state_dict()
        loaded: set[str] = set()
        for name, weight in weights:
            if name.endswith(".rope.rope.freqs"):
                name = name.removesuffix(".rope.rope.freqs") + ".rope.freqs"
            if name not in targets or name in loaded:
                raise ValueError(f"Unexpected or duplicate SeedVR2 checkpoint key: {name}")
            if targets[name].shape != weight.shape:
                raise ValueError(f"SeedVR2 checkpoint shape mismatch for {name}")
            targets[name].copy_(weight)
            loaded.add(name)
        missing = targets.keys() - loaded
        if missing:
            raise ValueError(f"Missing SeedVR2 checkpoint keys: {sorted(missing)}")
        return loaded

    @torch.inference_mode()
    def forward(self, batch: DiffusionRequestBatch) -> list[DiffusionOutput]:
        outputs = []
        for request in batch.requests:
            params = request.sampling_params
            if not isinstance(request.prepared_layout, SeedVR2Input):
                prepare_request(request, frame_pixels=self.frame_pixels, clip_pixels=self.clip_pixels)
            prepared = request.prepared_layout
            sample = prepare_video(prepared.frames, params.height, params.width, self.device)
            generator = params.generator
            if generator is None:
                generator = torch.Generator(device=self.device).manual_seed(params.seed)
            if not isinstance(generator, torch.Generator):
                raise ValueError("SeedVR2 accepts one generator per request")
            latent = self.vae.encode(sample).sample(generator=generator)
            condition = latent.permute(0, 2, 3, 4, 1).squeeze(0) * 0.9152
            noise = sample_noise(condition, generator)
            video = torch.cat((noise, condition, torch.ones_like(condition[..., :1])), dim=-1).reshape(-1, 33)
            shape = torch.tensor([condition.shape[:3]], device=self.device, dtype=torch.long)
            text_shape = torch.tensor([[self.text.shape[0]]], device=self.device, dtype=torch.long)
            timestep = torch.tensor([1000.0], device=self.device, dtype=torch.float16)
            runtime = self.transformer.build_runtime(
                self.transformer.token_grid_for(shape),
                text_len=self.text.shape[0],
                parallel_config=self.od_config.parallel_config,
            )
            velocity = self.transformer(video, self.text, shape, text_shape, timestep, runtime).vid_sample
            # Reference Euler returns fp32, then VAE casts to fp16 before scaling.
            restored = noise - velocity.reshape_as(noise)
            decoded = self.vae.decode((restored / 0.9152).permute(3, 0, 1, 2).unsqueeze(0))
            method = params.extra_args.get("color_correction_method") or DEFAULT_COLOR_CORRECTION_METHOD
            video = finish_video(decoded, sample, prepared.frame_count, method)
            payload: dict[str, object] = {"video": video}
            fps = prepared.source.fps if prepared.source is not None else params.fps
            metadata: dict[str, object] = {"video": {"fps": fps}} if fps is not None else {}
            if prepared.source is not None and prepared.source.audio is not None:
                payload["audio"] = prepared.source.audio
                metadata["audio"] = {"sample_rate": prepared.source.audio_sample_rate}
            outputs.append(DiffusionOutput(output={"payload": payload, "metadata": metadata}))
        return outputs
