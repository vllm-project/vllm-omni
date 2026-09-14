# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Wan TI2V first-frame conditioning shared by FULL and disaggregated stages.

The wire tensor is float32, normalized BCTHW with B=T=1, before output
fan-out. It is not the denoiser's output ``latents``. Versioned metadata is
also emitted for TI2V requests *without* an image, making dropped conditioning
detectable rather than silently turning an image request into T2V.
"""

from typing import Any

import PIL.Image
import torch


def validate_wan_batch_settings(settings: list[dict[str, Any]], *, stage: str) -> None:
    """Compare only effective settings actually shared by this execution path.

    The request scheduler already groups compatible requests. This is a runtime
    boundary guard for direct callers, not a replacement for its broader key.
    Values must be resolved scalars/tuples, never request-local tensors or RNGs.
    """
    if len(settings) < 2:
        return
    first = settings[0]
    for index, current in enumerate(settings[1:], start=1):
        for name, value in first.items():
            if current[name] != value:
                raise ValueError(
                    f"Wan {stage} batch requires matching effective {name}; "
                    f"request {index} has {current[name]!r}, request 0 has {value!r}."
                )


def effective_wan_dimensions(
    sampling: Any, spatial_scale: int, temporal_scale: int, patch_size: tuple
) -> tuple[int, int, int]:
    # Preserve Wan22Pipeline's rounding (including the historical width modulus).
    modulus = spatial_scale * patch_size[1]
    height = ((sampling.height or 480) // modulus) * modulus
    width = ((sampling.width or 832) // modulus) * modulus
    frames = sampling.num_frames or 81
    if frames % temporal_scale != 1:
        frames = frames // temporal_scale * temporal_scale + 1
    return height, width, max(frames, 1)


def conditioning_metadata(
    height: int,
    width: int,
    num_frames: int,
    spatial_scale: int,
    temporal_scale: int,
    has_image: bool,
) -> dict[str, Any]:
    return {
        "version": 1,
        "normalization": "wan_latents_mean_std",
        "layout": "BCTHW",
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "spatial_scale": spatial_scale,
        "temporal_scale": temporal_scale,
        "has_image": has_image,
    }


def prepare_wan_image_tensor(images: list, height: int, width: int, spatial_scale: int) -> torch.Tensor:
    """Keep PIL resize/preprocess and already-preprocessed tensor semantics."""
    from diffusers.video_processor import VideoProcessor

    processor = VideoProcessor(vae_scale_factor=spatial_scale)
    tensors = []
    for image in images:
        if isinstance(image, str):
            image = PIL.Image.open(image)
        if isinstance(image, PIL.Image.Image):
            image = image.resize((width, height), PIL.Image.Resampling.LANCZOS)
            tensor = processor.preprocess(image, height=height, width=width)
        elif isinstance(image, torch.Tensor):
            tensor = image.unsqueeze(0) if image.ndim == 3 else image
        else:
            raise ValueError("Wan image condition must be a PIL image, path or tensor.")
        if tensor.ndim != 4 or tensor.shape[-2:] != (height, width):
            raise ValueError(f"Wan image condition expects BCHW images at {height}x{width}, got {tensor.shape}.")
        tensors.append(tensor)
    return torch.cat(tensors, dim=0)


def encode_wan_image_condition(vae: Any, image_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Deterministic posterior mode: never read or advance request generators."""
    if vae is None:
        raise RuntimeError("Wan image conditioning requires an encoder VAE on the encode or full stage.")
    encoded = vae.encode(image_tensor.unsqueeze(2).to(device=device, dtype=vae.dtype))
    if hasattr(encoded, "latent_dist"):
        latent_condition = encoded.latent_dist.mode()
    elif hasattr(encoded, "latents"):
        latent_condition = encoded.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")
    mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(latent_condition)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(latent_condition)
    return ((latent_condition - mean) * inv_std).to(torch.float32)


def validate_wan_conditioning(
    latent_condition: Any,
    metadata: Any,
    *,
    expected: dict[str, Any],
    channels: int,
) -> bool:
    """Validate a single request before any G-stage RNG consumption or compute."""
    if not isinstance(metadata, dict):
        raise ValueError("Wan TI2V generation requires 'wan_conditioning_metadata' from the encode stage.")
    if type(metadata.get("has_image")) is not bool:
        raise ValueError("wan_conditioning_metadata.has_image must be a bool.")
    for key, value in expected.items():
        if key == "has_image":
            continue
        if type(metadata.get(key)) is not type(value) or metadata[key] != value:
            raise ValueError(f"Invalid wan_conditioning_metadata.{key}: expected {value!r}, got {metadata.get(key)!r}.")
    if not metadata["has_image"]:
        if latent_condition is not None:
            raise ValueError("wan_image_condition supplied with has_image=False.")
        return False
    shape = (
        1,
        channels,
        1,
        expected["height"] // expected["spatial_scale"],
        expected["width"] // expected["spatial_scale"],
    )
    if not isinstance(latent_condition, torch.Tensor) or tuple(latent_condition.shape) != shape:
        raise ValueError(f"wan_image_condition must be a normalized first-frame tensor with shape {shape}.")
    if latent_condition.dtype != torch.float32 or not torch.isfinite(latent_condition).all():
        raise ValueError("wan_image_condition must contain finite normalized float32 latents.")
    return True


def wan_first_frame_mask(latents: torch.Tensor) -> torch.Tensor:
    mask = torch.ones((latents.shape[0], 1, *latents.shape[2:]), dtype=torch.float32, device=latents.device)
    mask[:, :, 0] = 0
    return mask
