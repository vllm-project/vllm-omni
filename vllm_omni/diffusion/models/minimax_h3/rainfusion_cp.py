# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""RainFusion layout helpers for pure AllGather-KV sequence parallelism.

RainFusion v2 selects 128-token blocks after a spatial 8x8 traversal.  With
AllGather-KV SP each rank owns only Q, therefore the traversal has to happen
*before* the model input is split.  This module owns that deterministic global
permutation and the physical padding contract required by all_gather.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


_RF_BLOCK_SIZE = 128
_RF_TILE_SIZE = 8


def _video_permutation(latent_grid: tuple[int, int, int]) -> torch.Tensor:
    """Return rf_v2's reordered-video index -> original-video index mapping.

    This intentionally mirrors MindIE-SD's ``rearrange_with_remaining``,
    including its first-frame and non-8-divisible H/W behavior.
    """
    frames, height, width = latent_grid
    if frames <= 0 or height <= 0 or width <= 0:
        raise ValueError(f"latent_grid must be positive, got {latent_grid!r}")

    indices = torch.arange(frames * height * width, dtype=torch.long).reshape(
        frames,
        height,
        width,
    )
    if height % _RF_TILE_SIZE == 0 and width % _RF_TILE_SIZE == 0:
        return (
            indices.reshape(
                frames,
                height // _RF_TILE_SIZE,
                _RF_TILE_SIZE,
                width // _RF_TILE_SIZE,
                _RF_TILE_SIZE,
            )
            .permute(0, 1, 3, 2, 4)
            .reshape(-1)
        )

    # The irregular path leaves frame zero in raster order.  For later frames,
    # rf_v2 emits the tiled main rectangle, then the remaining H rows, then the
    # remaining W columns of the main-H rectangle.
    first = indices[:1].reshape(-1)
    rest = indices[1:]
    height_main = height - height % _RF_TILE_SIZE
    width_main = width - width % _RF_TILE_SIZE
    parts = [first]
    if frames > 1 and height_main and width_main:
        parts.append(
            rest[:, :height_main, :width_main]
            .reshape(
                frames - 1,
                height_main // _RF_TILE_SIZE,
                _RF_TILE_SIZE,
                width_main // _RF_TILE_SIZE,
                _RF_TILE_SIZE,
            )
            .permute(0, 1, 3, 2, 4)
            .reshape(-1)
        )
    if frames > 1 and height_main < height:
        parts.append(rest[:, height_main:, :].reshape(-1))
    if frames > 1 and width_main < width and height_main:
        parts.append(rest[:, :height_main, width_main:].reshape(-1))
    return torch.cat(parts)


@dataclass(frozen=True, slots=True)
class RainFusionCPLayout:
    """Global and rank-local geometry for prearranged RainFusion CP.

    ``physical_len`` is deliberately a multiple of both the CP world size and
    128.  The final rank can therefore have fewer *logical* rows while still
    participating in fixed-shape all-gathers with a full physical shard.
    """

    prefix_len: int
    latent_grid: tuple[int, int, int]
    world_size: int
    rank: int
    used_len: int
    local_capacity: int
    physical_len: int
    q_global_start: int
    q_valid_len: int
    first_frame_block_num: int
    dense_block_start: int

    @classmethod
    def build(
        cls,
        *,
        prefix_len: int,
        latent_grid: tuple[int, int, int],
        world_size: int,
        rank: int,
    ) -> "RainFusionCPLayout":
        if world_size <= 1:
            raise ValueError(f"RainFusion CP needs world_size > 1, got {world_size}")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        if prefix_len < 0:
            raise ValueError(f"prefix_len must be non-negative, got {prefix_len}")
        video_len = math.prod(latent_grid)
        used_len = prefix_len + video_len
        local_capacity = math.ceil(
            math.ceil(used_len / world_size) / _RF_BLOCK_SIZE
        ) * _RF_BLOCK_SIZE
        physical_len = local_capacity * world_size
        q_global_start = rank * local_capacity
        q_valid_len = max(0, min(local_capacity, used_len - q_global_start))
        # The dense text prefix is moved after the video.  A partial tail video
        # block also becomes dense when it shares a 128-row block with prefix.
        dense_block_start = video_len // _RF_BLOCK_SIZE
        return cls(
            prefix_len=prefix_len,
            latent_grid=latent_grid,
            world_size=world_size,
            rank=rank,
            used_len=used_len,
            local_capacity=local_capacity,
            physical_len=physical_len,
            q_global_start=q_global_start,
            q_valid_len=q_valid_len,
            first_frame_block_num=math.ceil(
                (latent_grid[1] * latent_grid[2]) / _RF_BLOCK_SIZE
            ),
            dense_block_start=dense_block_start,
        )

    def permutation(self, device: torch.device) -> torch.Tensor:
        video_perm = _video_permutation(self.latent_grid).to(device=device)
        return torch.cat(
            (video_perm + self.prefix_len, torch.arange(self.prefix_len, device=device))
        )

    def inverse_permutation(self, device: torch.device) -> torch.Tensor:
        permutation = self.permutation(device)
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(self.used_len, device=device)
        return inverse

    def prearrange(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reorder logical rows and append fixed-size collective padding."""
        if tensor.shape[0] < self.used_len:
            raise ValueError(
                f"tensor has {tensor.shape[0]} rows, but RainFusion CP needs {self.used_len} logical rows"
            )
        reordered = tensor.index_select(0, self.permutation(tensor.device))
        if self.physical_len == self.used_len:
            return reordered
        return torch.cat(
            (
                reordered,
                reordered.new_zeros(
                    (self.physical_len - self.used_len, *reordered.shape[1:])
                ),
            )
        )

    def restore(self, tensor: torch.Tensor) -> torch.Tensor:
        """Drop collective padding and restore MiniMax's original packed order."""
        if tensor.shape[0] < self.used_len:
            raise ValueError(
                f"gathered tensor has {tensor.shape[0]} rows, but RainFusion CP needs {self.used_len} logical rows"
            )
        return tensor[: self.used_len].index_select(0, self.inverse_permutation(tensor.device))

    def attention_extra(self) -> dict[str, int | bool]:
        """Backend metadata; primitive values keep compiled/eager handoff simple."""
        return {
            "rainfusion_prearranged_cp": True,
            "rainfusion_cp_q_global_start": self.q_global_start,
            "rainfusion_cp_q_valid_len": self.q_valid_len,
            "rainfusion_cp_kv_valid_len": self.used_len,
            "rainfusion_cp_first_frame_block_num": self.first_frame_block_num,
            "rainfusion_cp_dense_block_start": self.dense_block_start,
        }
