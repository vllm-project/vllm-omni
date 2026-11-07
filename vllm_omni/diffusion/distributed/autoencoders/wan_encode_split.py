# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""How the distributed (tile-parallel) Wan VAE encode splits the frame.

``VLLM_OMNI_WAN_VAE_ENCODE_SPLIT`` selects the split:

* ``"tile"`` (default): diffusers' overlapping 256/192 tiles, blended at the seams --
  28 tiles at 1280x720, so 1.65x the pixels of the frame are encoded.
* ``"rank"``: one piece per rank on a grid matching the rank count (2x4 for 8 ranks at
  90x160 latent), each encoded with ``VLLM_OMNI_WAN_VAE_ENCODE_MARGIN`` latent px
  (default 2) of context on its inner edges that is cropped off again, so the pieces
  are concatenated instead of blended.

Neither split is exact against a single-GPU un-tiled encode. A finite margin is less
than the encoder's receptive field, and margin 0 is less accurate than ``"tile"``.
See ``docs/user_guide/diffusion/parallelism/vae_parallelism.md`` for when the split
applies and the measured speed/accuracy tradeoff.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import GridSpec, TileTask

logger = init_logger(__name__)

ENCODE_SPLIT_ENV = "VLLM_OMNI_WAN_VAE_ENCODE_SPLIT"
ENCODE_MARGIN_ENV = "VLLM_OMNI_WAN_VAE_ENCODE_MARGIN"
ENCODE_MARGIN_DEFAULT = 2


def encode_split_mode() -> str:
    value = os.environ.get(ENCODE_SPLIT_ENV, "tile").strip().lower() or "tile"
    if value not in ("tile", "rank"):
        raise ValueError(f"{ENCODE_SPLIT_ENV} must be 'tile' or 'rank', got {value!r}")
    return value


def encode_margin() -> int:
    margin = int(os.environ.get(ENCODE_MARGIN_ENV, str(ENCODE_MARGIN_DEFAULT)))
    if margin < 0:
        raise ValueError(f"{ENCODE_MARGIN_ENV} must be >= 0, got {margin}")
    return margin


def rank_grid(num_pieces: int, latent_height: int, latent_width: int) -> tuple[int, int]:
    """The (rows, cols) factorization of num_pieces whose pieces are closest to square;
    ties go to fewer rows, whose pieces span more of each (contiguous) row."""
    best, best_key = (1, num_pieces), None
    for rows in range(1, num_pieces + 1):
        if num_pieces % rows:
            continue
        cols = num_pieces // rows
        if rows > latent_height or cols > latent_width:
            continue
        piece_h, piece_w = latent_height / rows, latent_width / cols
        key = (max(piece_h, piece_w) / min(piece_h, piece_w), rows)
        if best_key is None or key < best_key:
            best, best_key = (rows, cols), key
    return best


def even_bounds(extent: int, parts: int) -> list[tuple[int, int]]:
    """Split [0, extent) into `parts` contiguous ranges whose sizes differ by at most one."""
    edges = [round(k * extent / parts) for k in range(parts + 1)]
    return list(zip(edges[:-1], edges[1:]))


def rank_split(vae: Any, x: torch.Tensor) -> tuple[list[TileTask], GridSpec]:
    """One piece per rank: latent-aligned, near-equal pieces on a rows x cols grid,
    each widened by the context margin on its inner edges."""
    _, _, num_frames, height, width = x.shape
    ratio = vae.spatial_compression_ratio
    if vae.config.patch_size is not None:
        assert ratio % vae.config.patch_size == 0
        ratio = ratio // vae.config.patch_size  # x is already patchified
    latent_height, latent_width = height // ratio, width // ratio
    num_pieces = min(int(vae.distributed_executor.parallel_size), vae.distributed_executor.world_size)
    rows, cols = rank_grid(num_pieces, latent_height, latent_width)
    margin = encode_margin()
    temporal_compression = vae.config.scale_factor_temporal
    frame_range = 1 + (num_frames - 1) // temporal_compression

    tasks: list[TileTask] = []
    crops: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for i, (h0, h1) in enumerate(even_bounds(latent_height, rows)):
        for j, (w0, w1) in enumerate(even_bounds(latent_width, cols)):
            # Latent-aligned sample window: owned range plus margin, clipped to the frame.
            eh0, eh1 = max(0, h0 - margin), min(latent_height, h1 + margin)
            ew0, ew1 = max(0, w0 - margin), min(latent_width, w1 + margin)
            hs, he = eh0 * ratio, height if eh1 == latent_height else eh1 * ratio
            ws, we = ew0 * ratio, width if ew1 == latent_width else ew1 * ratio
            time_list = []
            for k in range(frame_range):
                t = slice(0, 1) if k == 0 else slice(1 + temporal_compression * (k - 1), 1 + temporal_compression * k)
                time_list.append(x[:, :, t, hs:he, ws:we])
            crops[(i, j)] = (h0 - eh0, h1 - h0, w0 - ew0, w1 - w0)
            tasks.append(TileTask(len(tasks), (i, j), time_list, workload=(he - hs) * (we - ws)))

    logger.info_once(
        "Wan VAE distributed encode: %dx%d pieces of ~%dx%d latent px, margin %d (%s=rank)",
        rows,
        cols,
        latent_height // rows,
        latent_width // cols,
        margin,
        ENCODE_SPLIT_ENV,
    )
    grid_spec = GridSpec(
        split_dims=(3, 4),
        grid_shape=(rows, cols),
        tile_spec={"crops": crops, "latent_height": latent_height, "latent_width": latent_width},
        output_dtype=vae.dtype,
    )
    return tasks, grid_spec


def rank_merge(coord_tensor_map: dict[tuple[int, ...], torch.Tensor], grid_spec: GridSpec) -> torch.Tensor:
    """Crop each piece's margin and concatenate; no blending."""
    rows, cols = grid_spec.grid_shape
    crops = grid_spec.tile_spec["crops"]
    result_rows = []
    for i in range(rows):
        row = []
        for j in range(cols):
            top, h, left, w = crops[(i, j)]
            row.append(coord_tensor_map[(i, j)][:, :, :, top : top + h, left : left + w])
        result_rows.append(torch.cat(row, dim=-1))
    return torch.cat(result_rows, dim=3)
