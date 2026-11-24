# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Inference-only inputs for the LIBERO10 native nonstream reproduction."""

from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F


def prepare_libero_policy_inputs(image: np.ndarray, prompt: str, image_size: int) -> tuple[np.ndarray, str, list[int]]:
    """Match native uint8 bicubic resize, bottom/right padding and JSON caption.

    The checkpoint's 256 resolution tier is intentionally fixed here. The four
    frame-size values also tell the backend to remove VAE padding before sampling.
    """
    if image_size != 256:
        raise ValueError("The native LIBERO10 reproduction profile requires image_size=256.")
    height, width = image.shape[:2]
    if height != image_size:
        image = np.asarray(
            Image.fromarray(image).resize(
                (round(width * image_size / height), image_size), resample=Image.Resampling.BILINEAR
            )
        ).copy()
    height, width = image.shape[:2]
    candidates = ((256, 256, "1,1"), (320, 256, "4,3"), (256, 320, "3,4"), (320, 192, "16,9"), (192, 320, "9,16"))
    target_w, target_h, aspect = min(candidates, key=lambda size: abs(height / width - size[1] / size[0]))
    scale = min(target_w / width, target_h / height, 1.0)
    resized_h, resized_w = int(scale * height + 0.5), int(scale * width + 0.5)
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1)
    if (resized_h, resized_w) != (height, width):
        tensor = F.resize(tensor, [resized_h, resized_w], interpolation=F.InterpolationMode.BICUBIC, antialias=True)
    bottom, right = target_h - resized_h, target_w - resized_w
    if bottom or right:
        mode = "edge" if right >= resized_w or bottom >= resized_h else "reflect"
        tensor = F.pad(tensor, [0, 0, right, bottom], padding_mode=mode)
    description = prompt.strip()
    if not description.endswith((".", "!", "?")):
        description += "."
    # Native uses concat_view even when an evaluator submits a single camera.
    caption = {
        "cinematography": {"framing": "This video contains concatenated views from multiple camera perspectives."},
        "actions": [{"time": "0:00-0:01", "description": description, "idle_frame": "0 out of 16."}],
        "duration": "0s",
        "fps": 20.0,
        "resolution": {"H": target_h, "W": target_w},
        "aspect_ratio": aspect,
    }
    return tensor.permute(1, 2, 0).numpy(), json.dumps(caption), [target_h, target_w, resized_h, resized_w]
