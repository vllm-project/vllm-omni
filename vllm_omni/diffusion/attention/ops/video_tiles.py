# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Device-independent 3-D video tiling shared by sparse attention providers."""

import functools
import math

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

if HAS_TRITON:

    @triton.jit
    def _pack_video_tiles_kernel(
        x_ptr,
        out_ptr,
        strides: tl.constexpr,
        frames: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        tile_t: tl.constexpr,
        tile_h: tl.constexpr,
        tile_w: tl.constexpr,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        padded_length: tl.constexpr,
        block: tl.constexpr,
    ):
        offset = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
        batch = tl.program_id(1).to(tl.int64)
        token = offset // (heads * head_dim)
        head = (offset // head_dim) % heads
        dim = offset % head_dim
        tile = token // (tile_t * tile_h * tile_w)
        local = token % (tile_t * tile_h * tile_w)

        height_tiles = tl.cdiv(height, tile_h)
        width_tiles = tl.cdiv(width, tile_w)
        start_t = tile // (height_tiles * width_tiles) * tile_t
        start_h = (tile // width_tiles) % height_tiles * tile_h
        start_w = tile % width_tiles * tile_w
        valid_t = tl.minimum(frames - start_t, tile_t)
        valid_h = tl.minimum(height - start_h, tile_h)
        valid_w = tl.minimum(width - start_w, tile_w)

        # Edge tiles compact their valid tokens before the zero-filled tail.
        src_t = start_t + local // (valid_h * valid_w)
        src_h = start_h + (local // valid_w) % valid_h
        src_w = start_w + local % valid_w
        src_token = (src_t * height + src_h) * width + src_w
        src_offset = batch * strides[0] + src_token * strides[1] + head * strides[2] + dim * strides[3]
        valid = (token < padded_length) & (local < valid_t * valid_h * valid_w)
        value = tl.load(x_ptr + src_offset, mask=valid, other=0)
        dst_offset = batch * padded_length * heads * head_dim + offset
        tl.store(out_ptr + dst_offset, value, mask=token < padded_length)


def pack_video_tiles(
    x: torch.Tensor,
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
) -> torch.Tensor:
    """Pack a BSHD video grid into contiguous tiles with zero-filled tails.

    The sequence dimension must equal the product of ``dit_seq_shape``.
    CUDA inference combines gathering and padding in one kernel. CPU and
    autograd retain the indexed PyTorch implementation.
    """
    if any(dim <= 0 for dim in (*dit_seq_shape, *tile_size)):
        raise ValueError("dit_seq_shape and tile_size must contain positive dimensions")
    if x.shape[1] != math.prod(dit_seq_shape):
        raise ValueError(f"sequence length {x.shape[1]} must equal dit_seq_shape product {math.prod(dit_seq_shape)}")

    padded_length = math.prod([(n + tile - 1) // tile for n, tile in zip(dit_seq_shape, tile_size)]) * math.prod(
        tile_size
    )
    shape = (x.shape[0], padded_length, x.shape[2], x.shape[3])
    if not HAS_TRITON or not x.is_cuda or x.requires_grad:
        partition, _, non_pad, _ = get_tile_metadata(dit_seq_shape, tile_size, math.prod(tile_size), x.device)
        output = torch.zeros(shape, dtype=x.dtype, device=x.device)
        output[:, non_pad] = x[:, partition]
        return output

    output = torch.empty(shape, dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(padded_length * x.shape[2] * x.shape[3], 1024), x.shape[0])
    _pack_video_tiles_kernel[grid](
        x, output, x.stride(), *dit_seq_shape, *tile_size, x.shape[2], x.shape[3], padded_length, block=1024
    )
    return output


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
