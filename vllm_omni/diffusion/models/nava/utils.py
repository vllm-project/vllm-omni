# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from PIL import Image


def as_bool(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
        raise ValueError(f"Expected boolean-like value, got {value!r}.")
    return bool(value)


def resolve_num_frames(value: Any, extra: dict[str, Any], default: int) -> int:
    if value is not None and int(value) > 1:
        return int(value)
    extra_value = extra.get("num_frames", extra.get("frames"))
    return int(extra_value) if extra_value is not None else int(default)


def resize_center_crop(image: Image.Image, target_h: int, target_w: int) -> Image.Image:
    src_w, src_h = image.size
    scale = max(target_w / src_w, target_h / src_h)
    resize_w = int(round(src_w * scale))
    resize_h = int(round(src_h * scale))
    resized = image.resize((resize_w, resize_h), Image.Resampling.LANCZOS)
    left = (resize_w - target_w) // 2
    top = (resize_h - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def image_to_tensor(image: Any, height: int, width: int) -> torch.Tensor:
    if isinstance(image, list):
        if len(image) != 1:
            raise ValueError("NAVA image conditioning accepts exactly one first-frame image.")
        image = image[0]

    if isinstance(image, str | os.PathLike):
        with Image.open(image) as loaded:
            image = loaded.convert("RGB")

    if isinstance(image, Image.Image):
        image = resize_center_crop(image.convert("RGB"), height, width)
        array = np.array(image, copy=True)
        tensor = torch.from_numpy(array).float().permute(2, 0, 1).unsqueeze(0)
        return tensor / 255.0 * 2.0 - 1.0

    if isinstance(image, torch.Tensor):
        tensor = image.detach().float()
        if tensor.ndim == 3 and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 4 and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(0, 3, 1, 2)
        if tensor.ndim != 4:
            raise ValueError(f"Expected image tensor with 3 or 4 dims, got shape {tuple(tensor.shape)}.")
        if tensor.numel() and tensor.min().item() >= 0.0:
            tensor = tensor / 255.0 if tensor.max().item() > 2.0 else tensor
            tensor = tensor * 2.0 - 1.0
        return tensor

    raise TypeError(f"Unsupported NAVA image input type: {type(image)!r}")
