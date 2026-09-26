# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent sensor geometries and one flat state for the shared scheduler."""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F


def pack_state(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    # Camera-only state needs no concatenation. The result may share storage
    # with its input, just like the views returned by unpack_state.
    if len(tensors) == 1:
        return tensors[0].flatten(1)
    return torch.cat([tensor.flatten(1) for tensor in tensors], dim=1)


def unpack_state(state: torch.Tensor, shapes: Sequence[tuple[int, ...]]) -> tuple[torch.Tensor, ...]:
    sizes = [math.prod(shape) for shape in shapes]
    if state.shape[1] != sum(sizes):
        raise ValueError("Packed camera/LiDAR state does not match the declared geometries.")
    return tuple(part.reshape(state.shape[0], *shape) for part, shape in zip(state.split(sizes, dim=1), shapes))


def patchify_sensor(latent: torch.Tensor, patch: int) -> torch.Tensor:
    batch, channels, time, height, width = latent.shape
    hp, wp = math.ceil(height / patch), math.ceil(width / patch)
    latent = F.pad(latent, (0, wp * patch - width, 0, hp * patch - height))
    return (
        latent.reshape(batch, channels, time, hp, patch, wp, patch)
        .permute(0, 2, 3, 5, 4, 6, 1)
        .reshape(batch, time * hp * wp, patch * patch * channels)
    )


def unpatchify_sensor(tokens: torch.Tensor, shape: tuple[int, ...], patch: int) -> torch.Tensor:
    channels, time, height, width = shape
    hp, wp = math.ceil(height / patch), math.ceil(width / patch)
    return (
        tokens.reshape(tokens.shape[0], time, hp, wp, patch, patch, channels)
        .permute(0, 6, 1, 2, 4, 3, 5)
        .reshape(tokens.shape[0], channels, time, hp * patch, wp * patch)[..., :height, :width]
    )


def packed_position_ids(
    items,
    *,
    text_origin: int,
    base_fps: float,
    camera_compression: int,
    lidar_compression: int = 1,
    enable_fps_modulation: bool = True,
    align_views: bool = True,
) -> tuple[torch.Tensor, int | float]:
    """Place sensor streams at a shared origin and return the next sample cursor.

    The endpoint matches reference packing for a subsequent sample or modality.
    The current pipeline packs one sample and no following modality, so its
    transformer consumes only the position IDs; the cursor is inert there.
    """
    from .transformer_cosmos3 import compute_mrope_position_ids_vision

    positions = []
    end = text_origin
    for item in items:
        compression = lidar_compression if item.is_lidar else camera_compression
        positions_i, endpoint = compute_mrope_position_ids_vision(
            *item.token_shape,
            temporal_offset=text_origin,
            fps=compression / item.seconds_per_frame,
            base_fps=base_fps,
            temporal_compression_factor=compression,
            base_temporal_compression_factor=camera_compression,
            enable_fps_modulation=enable_fps_modulation,
            temporal_position_period=item.token_shape[0] // item.num_views if align_views else None,
        )
        positions.append(positions_i)
        end = max(end, endpoint)
    return torch.cat(positions, dim=1), end
