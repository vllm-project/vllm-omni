# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Four spatial tile lanes, each with a two-rank decoder projection shard."""

from types import SimpleNamespace

import torch
from torch import nn


def tile_tensor_groups(world, size=2):
    rank = world.rank_in_group
    tile_count = world.world_size // size
    tensor_group = None
    for lane in range(tile_count):
        ranks = world.ranks[lane * size : (lane + 1) * size]
        group = torch.distributed.new_group(ranks=ranks, backend="nccl")
        if rank // size == lane:
            tensor_group = group
    tile_group = None
    for shard in range(size):
        ranks = world.ranks[shard::size]
        group = torch.distributed.new_group(ranks=ranks, backend="nccl")
        if rank % size == shard:
            tile_group = SimpleNamespace(
                world_size=tile_count, rank_in_group=rank // size, device_group=group, ranks=ranks
            )
    return tile_group, tensor_group, rank % size


class RowProjection(nn.Module):
    """Reduce FP32 partial sums before the original BF16 output rounding."""

    def __init__(self, linear, group, rank, size):
        super().__init__()
        self.register_buffer("weight", linear.weight.chunk(size, dim=1)[rank].contiguous())
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().float())
        self.group = group

    def forward(self, x):
        result = torch.mm(x.reshape(-1, x.shape[-1]), self.weight.T, out_dtype=torch.float32)
        torch.distributed.all_reduce(result, group=self.group)
        if self.bias is not None:
            result.add_(self.bias)
        return result.to(x.dtype).reshape(*x.shape[:-1], self.weight.shape[0])
