# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
MAX_IMAGE_RATIO = 200
SPATIAL_MERGE_SIZE = 2


@dataclass(frozen=True)
class LingBotImageCondition:
    vlm_image: Image.Image
    clean_latent: torch.Tensor


def _round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> tuple[int, int]:
    max_pixels = max_pixels if max_pixels is not None else IMAGE_MAX_TOKEN_NUM * factor**2
    min_pixels = min_pixels if min_pixels is not None else IMAGE_MIN_TOKEN_NUM * factor**2
    if max_pixels < min_pixels:
        raise ValueError("max_pixels must be greater than or equal to min_pixels.")
    if min(height, width) <= 0:
        raise ValueError(f"LingBot image height and width must be positive, got {height}x{width}.")
    if max(height, width) / min(height, width) > MAX_IMAGE_RATIO:
        raise ValueError(f"LingBot image absolute aspect ratio must be smaller than {MAX_IMAGE_RATIO}.")

    resized_height = max(factor, _round_by_factor(height, factor))
    resized_width = max(factor, _round_by_factor(width, factor))
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = _floor_by_factor(height / beta, factor)
        resized_width = _floor_by_factor(width / beta, factor)
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = _ceil_by_factor(height * beta, factor)
        resized_width = _ceil_by_factor(width * beta, factor)
    return resized_height, resized_width


def geometry_align_image(
    image: Image.Image,
    height: int,
    width: int,
) -> torch.Tensor:
    if not isinstance(image, Image.Image):
        raise TypeError(f"LingBot input image must be a PIL image, got {type(image)!r}.")
    if height <= 0 or width <= 0:
        raise ValueError(f"LingBot target height and width must be positive, got {height}x{width}.")

    raw = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).unsqueeze(0).contiguous()
    old_height, old_width = raw.shape[-2:]
    scale = max(height / old_height, width / old_width)
    new_height = max(math.ceil(old_height * scale), height)
    new_width = max(math.ceil(old_width * scale), width)
    resized = F.interpolate(
        raw,
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    )
    top = int(round((new_height - height) / 2.0))
    left = int(round((new_width - width) / 2.0))
    cropped = resized[:, :, top : top + height, left : left + width]
    return (cropped.float() / 255.0).unsqueeze(2)


def pixel_tensor_to_pil(pixel: torch.Tensor) -> Image.Image:
    if pixel.ndim != 5 or pixel.shape[0] != 1 or pixel.shape[1] != 3:
        raise ValueError(f"LingBot aligned image tensor must have shape [1,3,T,H,W], got {tuple(pixel.shape)}.")
    frame = pixel[0, :, 0].detach().cpu().clamp(0, 1)
    array = frame.permute(1, 2, 0).mul(255).byte().numpy()
    return Image.fromarray(array, mode="RGB")


def vlm_image_from_aligned(
    pixel: torch.Tensor,
    *,
    vision_patch_size: int,
) -> Image.Image:
    image = pixel_tensor_to_pil(pixel)
    patch_factor = int(vision_patch_size) * SPATIAL_MERGE_SIZE
    if patch_factor <= 0:
        raise ValueError(f"LingBot vision patch factor must be positive, got {patch_factor}.")
    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        factor=patch_factor,
    )
    return image.resize((resized_width, resized_height))


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.no_grad()
def encode_clean_image_latent(
    pixel: torch.Tensor,
    *,
    vae: torch.nn.Module,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    vae_device = _module_device(vae)
    normalized = (pixel.to(device=vae_device, dtype=torch.float32) - 0.5) / 0.5
    with torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
        enabled=vae_device.type == "cuda",
    ):
        encoded = vae.encode(normalized)
        # Match the official LingBot TI2V path exactly: sample the posterior
        # with the request generator before initial noise latents are created.
        # Using mode() would change both the clean prefix and the RNG stream.
        clean_latent = encoded.latent_dist.sample(generator)

    mean = torch.tensor(
        vae.config.latents_mean,
        device=clean_latent.device,
        dtype=torch.float32,
    ).view(1, -1, 1, 1, 1)
    std_inv = (
        1.0
        / torch.tensor(
            vae.config.latents_std,
            device=clean_latent.device,
            dtype=torch.float32,
        )
    ).view(1, -1, 1, 1, 1)
    return (clean_latent.float() - mean) * std_inv


def prepare_ti2v_image_condition(
    image: Image.Image,
    *,
    height: int,
    width: int,
    vae: torch.nn.Module,
    vision_patch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> LingBotImageCondition:
    pixel = geometry_align_image(image, height, width)
    vlm_image = vlm_image_from_aligned(
        pixel,
        vision_patch_size=vision_patch_size,
    )
    clean_latent = encode_clean_image_latent(
        pixel,
        vae=vae,
        generator=generator,
    ).to(device=device, dtype=torch.float32)
    return LingBotImageCondition(
        vlm_image=vlm_image,
        clean_latent=clean_latent,
    )


def apply_clean_prefix(
    latents: torch.Tensor,
    clean_latent: torch.Tensor,
) -> torch.Tensor:
    if latents.ndim != 5 or clean_latent.ndim != 5:
        raise ValueError(
            "LingBot latent and clean prefix must be 5D [B,C,T,H,W] tensors, "
            f"got {tuple(latents.shape)} and {tuple(clean_latent.shape)}."
        )
    if latents.shape[:2] != clean_latent.shape[:2] or latents.shape[3:] != clean_latent.shape[3:]:
        raise ValueError(
            "LingBot clean prefix batch/channel/spatial shape must match latents, "
            f"got {tuple(clean_latent.shape)} for {tuple(latents.shape)}."
        )
    prefix_length = int(clean_latent.shape[2])
    if prefix_length > latents.shape[2]:
        raise ValueError(f"LingBot clean prefix length {prefix_length} exceeds latent length {latents.shape[2]}.")
    latents[:, :, :prefix_length] = clean_latent.to(
        device=latents.device,
        dtype=latents.dtype,
    )
    return latents
