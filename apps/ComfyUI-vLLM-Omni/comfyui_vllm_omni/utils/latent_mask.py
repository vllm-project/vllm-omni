# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Client-side serialization of MiniMax-H3 latent-edit masks.

A video mask is sent in its raw form — a 2D spatial ``[H, W]`` (applied to
every frame) or a 3D frame-space ``[T, H, W]`` (one slice per source frame) —
and the server resolves it to the latent grid. An audio mask is either a scalar
(uniform, applied to all time steps) or a raw temporal ``[T]`` / channel-major
``[C, T]`` mask, which the server resamples to its audio latent length.
"""

import json

import torch


def scalar_mask_to_json(value: float) -> str:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"mask value must be in [0, 1], got {value}")
    return str(value)


def video_mask_to_json(mask: torch.Tensor) -> str:
    """Serialize a raw video mask to JSON for the ``video_noise_mask`` field.

    ``mask`` is a 2D spatial mask or a 3D frame-space mask. The server resizes
    it to the latent grid, so the client neither floors the canvas nor maps
    frames to latents.
    """
    if mask.ndim not in (2, 3):
        raise ValueError(f"expected a 2D or 3D mask tensor, got {mask.ndim}D")
    return json.dumps(mask.tolist(), separators=(",", ":"))


def audio_mask_to_json(mask: torch.Tensor) -> str:
    """Serialize a raw temporal audio mask to JSON for the ``audio_noise_mask``
    field.

    ``mask`` is a 1D temporal ``[T]`` mask (one value per time step) or a 2D
    channel-major ``[C, T]`` mask. The server resamples the time axis to its
    audio latent length and broadcasts a single channel to the stereo pair.
    """
    if mask.ndim not in (1, 2):
        raise ValueError(f"expected a 1D or 2D mask tensor, got {mask.ndim}D")
    return json.dumps(mask.tolist(), separators=(",", ":"))


# The temporal-mask node computes its preserve/regenerate boundary on the H3
# latent lattice for preview purposes; these mirror the server shape planner.
def _align_frame_count(frame_count: int) -> int:
    if frame_count <= 0:
        return 1
    current = int(frame_count)
    while current % 17 != 5:
        current += 1
    return current


def _video_latent_t(frame_count: int) -> int:
    if frame_count <= 5:
        return 2
    return ((int(frame_count) - 5) // 17) * 5 + 2
