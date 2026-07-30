# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass(frozen=True)
class LingBotRefinerConfig:
    """Immutable startup configuration for the optional LingBot Refiner."""

    enabled: bool = False
    model_dir: str | None = None
    transformer_subfolder: str = "refiner"
    revision: str | None = None
    default_run: bool = True
    offload_vae_during_denoise: bool = True


@dataclass(frozen=True)
class LingBotRefinerInputs:
    """Canonical Base-to-Refiner handoff tensors and sampling metadata."""

    latents: torch.Tensor
    clean_prefix: torch.Tensor | None
    num_frames: int
    source_fps: float
    sample_fps: int


def normalize_lingbot_refiner_config(
    model_config: Mapping[str, Any] | None,
    *,
    base_model: str,
    base_revision: str | None,
) -> LingBotRefinerConfig:
    """Parse ``model_config.lingbot_refiner`` without probing model storage."""

    model_config = model_config or {}
    raw = model_config.get("lingbot_refiner")
    if raw is None:
        return LingBotRefinerConfig()
    if not isinstance(raw, Mapping):
        raise ValueError("LingBot `model_config.lingbot_refiner` must be a mapping.")

    supported = {
        "enabled",
        "model_dir",
        "transformer_subfolder",
        "revision",
        "default_run",
        "offload_vae_during_denoise",
    }
    unknown = sorted(set(raw) - supported)
    if unknown:
        raise ValueError(f"Unsupported LingBot Refiner startup options: {unknown}.")

    enabled = raw.get("enabled", False)
    default_run = raw.get("default_run", True)
    offload_vae = raw.get("offload_vae_during_denoise", True)
    for name, value in (
        ("enabled", enabled),
        ("default_run", default_run),
        ("offload_vae_during_denoise", offload_vae),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"LingBot Refiner `{name}` must be a boolean, got {value!r}.")

    model_dir = raw.get("model_dir")
    if model_dir is None:
        model_dir = base_model
    if not isinstance(model_dir, str) or not model_dir:
        raise ValueError(f"LingBot Refiner `model_dir` must be a non-empty string, got {model_dir!r}.")

    transformer_subfolder = raw.get("transformer_subfolder", "refiner")
    if not isinstance(transformer_subfolder, str) or not transformer_subfolder:
        raise ValueError(
            "LingBot Refiner `transformer_subfolder` must be a non-empty string, "
            f"got {transformer_subfolder!r}."
        )

    revision = raw.get("revision")
    if revision is None:
        revision = base_revision
    if revision is not None and (not isinstance(revision, str) or not revision):
        raise ValueError(f"LingBot Refiner `revision` must be null or a non-empty string, got {revision!r}.")

    return LingBotRefinerConfig(
        enabled=enabled,
        model_dir=model_dir,
        transformer_subfolder=transformer_subfolder,
        revision=revision,
        default_run=default_run,
        offload_vae_during_denoise=offload_vae,
    )


def validate_refiner_sigmas(
    sigmas: Sequence[float] | np.ndarray,
    t_thresh: float | None = None,
) -> np.ndarray:
    arr = np.asarray(list(sigmas), dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Refiner sigma schedule must be a non-empty 1D list")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Refiner sigma schedule contains non-finite values")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(f"Refiner sigma schedule values must be in [0, 1], got {arr.tolist()}")
    if arr.size > 1 and not np.all(np.diff(arr) < 0.0):
        raise ValueError(f"Refiner sigma schedule must be strictly descending, got {arr.tolist()}")
    if t_thresh is not None and abs(float(arr[0]) - float(t_thresh)) > 1e-6:
        raise ValueError(f"Refiner sigma schedule must start at t_thresh={float(t_thresh)}, got {float(arr[0])}")
    return arr


def compute_refiner_sigmas(
    *,
    sigma_max: float,
    sigma_min: float,
    num_inference_steps: int,
    shift: float,
    t_thresh: float,
    tail_steps: int = 0,
) -> np.ndarray:
    t_value = float(t_thresh)
    if not (0.0 < t_value <= 1.0):
        raise ValueError(f"Refiner t_thresh must lie in (0, 1], got {t_value}")
    steps = int(num_inference_steps)
    if steps < 1:
        raise ValueError(f"Refiner num_inference_steps must be >= 1, got {steps}")
    tail = int(tail_steps)
    if tail < 0:
        raise ValueError(f"refiner_sigma_tail_steps must be >= 0, got {tail}")

    base = np.linspace(float(sigma_max), float(sigma_min), steps + 1).copy()[:-1]
    shift_value = float(shift)
    shifted = shift_value * base / (1.0 + (shift_value - 1.0) * base)
    eps = 1e-6
    sigmas = shifted[shifted <= t_value + eps]
    if sigmas.size == 0 or abs(float(sigmas[0]) - t_value) > eps:
        sigmas = np.concatenate([[t_value], sigmas])
    if tail > 0:
        start = float(sigmas[-1])
        stop = min(float(sigma_min), start)
        extra = np.linspace(start, stop, tail + 2, dtype=np.float64)[1:-1]
        sigmas = np.concatenate([sigmas, extra])
    return validate_refiner_sigmas(sigmas, t_value).astype(np.float32)


def compute_refiner_frame_budget(
    num_source_frames: int,
    source_fps: float,
    *,
    sample_fps: int = 24,
    vae_temporal_factor: int = 4,
    max_frames: int | None = None,
) -> int:
    if num_source_frames <= 0:
        raise ValueError(f"Refiner source video must contain at least one frame, got {num_source_frames}.")
    if not math.isfinite(float(source_fps)) or source_fps <= 0:
        raise ValueError(f"Refiner source_fps must be positive and finite, got {source_fps!r}.")
    if sample_fps <= 0:
        raise ValueError(f"Refiner sample_fps must be positive, got {sample_fps}.")
    if vae_temporal_factor <= 0:
        raise ValueError(f"Refiner VAE temporal factor must be positive, got {vae_temporal_factor}.")
    if max_frames is not None and (
        max_frames <= 0 or (max_frames != 1 and (max_frames - 1) % vae_temporal_factor != 0)
    ):
        raise ValueError(
            "Refiner max_frames must be 1 or aligned to the VAE temporal factor, "
            f"got {max_frames}."
        )

    if source_fps > sample_fps:
        raw = int(num_source_frames / source_fps * sample_fps)
    else:
        raw = int(num_source_frames)
    sample_frames = ((raw - 1) // vae_temporal_factor) * vae_temporal_factor + 1
    sample_frames = max(sample_frames, 1)
    if max_frames is not None:
        sample_frames = min(sample_frames, int(max_frames))
    return int(sample_frames)


def compute_refiner_frame_indices(
    num_source_frames: int,
    sample_frames: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    if num_source_frames <= 0 or sample_frames <= 0:
        raise ValueError(
            "Refiner frame counts must be positive, "
            f"got source={num_source_frames}, sample={sample_frames}."
        )
    if num_source_frames >= sample_frames:
        indices = np.linspace(0, num_source_frames - 1, sample_frames, dtype=np.int64)
    else:
        indices = np.concatenate(
            [
                np.arange(num_source_frames, dtype=np.int64),
                np.full(sample_frames - num_source_frames, num_source_frames - 1, dtype=np.int64),
            ]
        )
    return torch.as_tensor(indices, device=device, dtype=torch.long)


def resize_refiner_video(video: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    if video.ndim != 5:
        raise ValueError(f"Refiner video must have shape [B,C,T,H,W], got {tuple(video.shape)}.")
    batch, channels, frames, source_height, source_width = video.shape
    flat = video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, source_height, source_width)
    resized = F.interpolate(flat, size=(height, width), mode="bicubic", align_corners=False)
    return (
        resized.clamp(0.0, 1.0)
        .reshape(batch, frames, channels, height, width)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )


def prepare_refiner_latent(
    x_up: torch.Tensor,
    noise: torch.Tensor,
    t_thresh: float | torch.Tensor,
) -> torch.Tensor:
    if x_up.shape != noise.shape:
        raise ValueError(f"Refiner x_up/noise shapes differ: {tuple(x_up.shape)} vs {tuple(noise.shape)}.")
    if not torch.is_tensor(t_thresh):
        t_thresh = torch.tensor(float(t_thresh), device=x_up.device, dtype=x_up.dtype)
    while t_thresh.ndim < x_up.ndim:
        t_thresh = t_thresh.view(*t_thresh.shape, *([1] * (x_up.ndim - t_thresh.ndim)))
    return (1.0 - t_thresh) * x_up + t_thresh * noise


def align_refiner_first_frame(
    image: Image.Image,
    *,
    target_height: int,
    target_width: int,
    source_height: int,
    source_width: int,
) -> torch.Tensor:
    """Center-crop the original TI2V image to the Base video's geometry."""

    image = image.convert("RGB")
    image_width, image_height = image.size
    source_aspect = float(source_width) / float(source_height)
    image_aspect = float(image_width) / float(image_height)
    if image_aspect > source_aspect:
        crop_height = image_height
        crop_width = max(1, int(round(crop_height * source_aspect)))
        left = int(round((image_width - crop_width) / 2.0))
        top = 0
    else:
        crop_width = image_width
        crop_height = max(1, int(round(crop_width / source_aspect)))
        left = 0
        top = int(round((image_height - crop_height) / 2.0))
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    cropped = cropped.resize((target_width, target_height), resample=Image.Resampling.BICUBIC)
    array = np.asarray(cropped, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).contiguous()
