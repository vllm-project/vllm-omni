# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Device-independent 3-D video tiling shared by sparse attention providers."""

import functools
import math

import torch


@functools.lru_cache(maxsize=32)
def get_tile_partition_indices(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    t_size, h_size, w_size = dit_seq_shape
    tile_t, tile_h, tile_w = tile_size
    indices = torch.arange(t_size * h_size * w_size, device=device, dtype=torch.long).reshape(t_size, h_size, w_size)
    tiles = []
    for tile_t_idx in range(math.ceil(t_size / tile_t)):
        for tile_h_idx in range(math.ceil(h_size / tile_h)):
            for tile_w_idx in range(math.ceil(w_size / tile_w)):
                tiles.append(
                    indices[
                        tile_t_idx * tile_t : min((tile_t_idx + 1) * tile_t, t_size),
                        tile_h_idx * tile_h : min((tile_h_idx + 1) * tile_h, h_size),
                        tile_w_idx * tile_w : min((tile_w_idx + 1) * tile_w, w_size),
                    ].flatten()
                )
    return torch.cat(tiles, dim=0)


@functools.lru_cache(maxsize=32)
def construct_variable_block_sizes(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    num_tiles = tuple(math.ceil(seq_dim / tile_dim) for seq_dim, tile_dim in zip(dit_seq_shape, tile_size))

    def _sizes(dim_len: int, tile: int, n_tiles: int) -> torch.Tensor:
        sizes = torch.full((n_tiles,), tile, dtype=torch.int32, device=device)
        remainder = dim_len - (n_tiles - 1) * tile
        sizes[-1] = remainder if remainder > 0 else tile
        return sizes

    t_sizes = _sizes(dit_seq_shape[0], tile_size[0], num_tiles[0])
    h_sizes = _sizes(dit_seq_shape[1], tile_size[1], num_tiles[1])
    w_sizes = _sizes(dit_seq_shape[2], tile_size[2], num_tiles[2])
    return (t_sizes[:, None, None] * h_sizes[None, :, None] * w_sizes[None, None, :]).reshape(-1)


@functools.lru_cache(maxsize=32)
def get_non_pad_index(variable_block_sizes: torch.Tensor, max_block_size: int) -> torch.Tensor:
    num_blocks = variable_block_sizes.shape[0]
    device = variable_block_sizes.device
    starts = torch.arange(num_blocks, device=device) * max_block_size
    padded_index = starts[:, None] + torch.arange(max_block_size, device=device)[None, :]
    valid = torch.arange(max_block_size, device=device)[None, :] < variable_block_sizes[:, None]
    return padded_index[valid]


@torch.compiler.disable
def get_tile_metadata(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    block_elements: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tile_partition_indices = get_tile_partition_indices(dit_seq_shape, tile_size, device)
    variable_block_sizes = construct_variable_block_sizes(dit_seq_shape, tile_size, device)
    non_pad_index = get_non_pad_index(variable_block_sizes, block_elements)
    untile_combined_index = non_pad_index[torch.argsort(tile_partition_indices)]
    return tile_partition_indices, variable_block_sizes, non_pad_index, untile_combined_index
