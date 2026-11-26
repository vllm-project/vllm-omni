# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pixel operations shared by Pi-family image processors."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def resize_with_pad(
    images: torch.Tensor,
    target_height: int,
    target_width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize ``(B, C, H, W)`` images with ``-1`` padding.

    This is the OpenPI ``resize_with_pad_torch`` pixel operation used by both
    Pi0 and Pi0.5 after their variant-specific input-domain conversion.
    """
    if images.ndim != 4:
        raise ValueError(f"Expected 4-D (B,C,H,W), got {images.ndim}-D")
    _, _, cur_h, cur_w = images.shape
    ratio = max(cur_w / target_width, cur_h / target_height)
    rh, rw = int(cur_h / ratio), int(cur_w / ratio)
    align_corners = False if mode == "bilinear" else None
    resized = F.interpolate(images, size=(rh, rw), mode=mode, align_corners=align_corners)
    resized = resized.clamp(-1.0, 1.0)
    ph, rem_h = divmod(target_height - rh, 2)
    pw, rem_w = divmod(target_width - rw, 2)
    return F.pad(resized, (pw, pw + rem_w, ph, ph + rem_h), value=-1.0)


def pil_image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL input to ``(1, 3, H, W)`` float32 in ``[-1, 1]``."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.array(image, dtype=np.float32) / 255.0 * 2.0 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def make_empty_image(image_size: int) -> torch.Tensor:
    """Create the ``-1``-filled tensor used for a missing camera slot."""
    return torch.full((1, 3, image_size, image_size), -1.0)
