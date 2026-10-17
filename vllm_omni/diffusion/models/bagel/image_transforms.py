# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

VAE_MAX_SIZE = 1024
VAE_MIN_SIZE = 512
VIT_MAX_SIZE = 980
VIT_MIN_SIZE = 224
MAX_PIXELS = 14 * 14 * 9 * 1024


def to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA" or image.info.get("transparency") is not None:
        image = image.convert("RGBA")
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    return image.convert("RGB")


def _make_divisible(value: float, stride: int) -> int:
    return max(stride, int(round(value / stride) * stride))


def _apply_scale(width: int, height: int, scale: float, stride: int) -> tuple[int, int]:
    return _make_divisible(round(width * scale), stride), _make_divisible(round(height * scale), stride)


def resized_size(width: int, height: int, *, max_size: int, min_size: int, stride: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got {width}x{height}.")
    scale = min(max_size / max(width, height), 1.0)
    scale = max(scale, min_size / min(width, height))
    width, height = _apply_scale(width, height, scale, stride)
    if width * height > MAX_PIXELS:
        width, height = _apply_scale(width, height, MAX_PIXELS / (width * height), stride)
    if max(width, height) > max_size:
        width, height = _apply_scale(width, height, max_size / max(width, height), stride)
    return width, height


def vae_size(width: int, height: int, *, max_size: int = VAE_MAX_SIZE, stride: int = 16) -> tuple[int, int]:
    return resized_size(width, height, max_size=max_size, min_size=min(VAE_MIN_SIZE, max_size), stride=stride)


def vit_size(width: int, height: int, *, max_size: int = VIT_MAX_SIZE, stride: int = 14) -> tuple[int, int]:
    return resized_size(width, height, max_size=max_size, min_size=min(VIT_MIN_SIZE, max_size), stride=stride)


def resize_for_vae(image: Image.Image, *, max_size: int = VAE_MAX_SIZE, stride: int = 16) -> Image.Image:
    image = to_rgb(image)
    return image.resize(vae_size(*image.size, max_size=max_size, stride=stride), Image.BICUBIC)


def resize_for_vit(image: Image.Image, *, max_size: int = VIT_MAX_SIZE, stride: int = 14) -> Image.Image:
    return image.resize(vit_size(*image.size, max_size=max_size, stride=stride), Image.BICUBIC)


def to_tensor(image: Image.Image) -> torch.Tensor:
    pixels = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True)).permute(2, 0, 1).contiguous()
    return pixels.to(torch.float32).div(255).sub(0.5).div(0.5)
