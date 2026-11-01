# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Client-side serialization of MiniMax-H3 latent-edit masks.

A video mask is sent in frame space — a 2D spatial ``[H, W]`` (applied to every
frame) or a 3D ``[T, H, W]`` (one slice per frame) — and the server resolves it
to the latent grid. An audio mask is a scalar applied to all time steps.
"""

import json
import math

import torch
import torch.nn.functional as F

# The server rejects mask JSON above 8 MiB, so the spatial axes are
# area-downsampled by the VAE spatial stride before upload. This depends only on
# the stride, not on the H3 shape lattice, which the server still owns.
_VAE_SPATIAL_STRIDE = 16
_MASK_DECIMALS = 4


def scalar_mask_to_json(value: float) -> str:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"mask value must be in [0, 1], got {value}")
    return str(value)


def video_mask_to_json(mask: torch.Tensor) -> str:
    """Serialize a frame-space video mask to JSON for the ``video_noise_mask`` field.

    ``mask`` is a 2D spatial mask or a 3D frame-space mask. Its spatial axes are
    area-downsampled by the VAE stride; the server resizes the result to the
    latent grid, so the client neither floors the canvas nor maps frames to
    latents.
    """
    if mask.ndim not in (2, 3):
        raise ValueError(f"expected a 2D or 3D mask tensor, got {mask.ndim}D")
    frames = mask.unsqueeze(0) if mask.ndim == 2 else mask
    size = tuple(max(1, math.ceil(dim / _VAE_SPATIAL_STRIDE)) for dim in frames.shape[1:])
    frames = F.interpolate(frames.unsqueeze(1).float(), size=size, mode="area").squeeze(1)
    if mask.ndim == 2:
        frames = frames.squeeze(0)
    # Round in float64 so tolist() emits short decimals rather than float32 noise.
    return json.dumps(frames.double().round(decimals=_MASK_DECIMALS).tolist(), separators=(",", ":"))


# The temporal-mask node snaps its preserve boundary to the H3 frame lattice;
# this mirrors the server shape planner.
def _align_frame_count(frame_count: int) -> int:
    if frame_count <= 0:
        return 1
    current = int(frame_count)
    while current % 17 != 5:
        current += 1
    return current
