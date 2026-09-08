# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2025-2026 SandAI. All Rights Reserved.

"""Packed token/head exchanges over caller-owned parallel groups."""

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class HeadParallelLayout:
    """Token counts in process-group rank order, shared by both exchanges.

    Every rank must provide the same split tuple. Counts describe input tokens,
    not routed expert rows. Head padding and replicated-token handling belong
    to the caller; heads must be evenly partitioned within this group.
    """

    token_counts: tuple[int, ...]
    rank: int

    def __post_init__(self) -> None:
        if not isinstance(self.token_counts, tuple) or not self.token_counts:
            raise ValueError("token_counts must be a nonempty tuple in process-group rank order")
        if any(type(count) is not int or count < 0 for count in self.token_counts):
            raise ValueError("token_counts must contain nonnegative integers")
        if type(self.rank) is not int or not 0 <= self.rank < self.world_size:
            raise ValueError("rank must index token_counts")

    @property
    def world_size(self) -> int:
        return len(self.token_counts)

    @property
    def local_tokens(self) -> int:
        return self.token_counts[self.rank]

    @property
    def total_tokens(self) -> int:
        return sum(self.token_counts)

    def validate_group(self, group: dist.ProcessGroup | None) -> None:
        if group is None:
            if self.world_size != 1:
                raise ValueError("a caller-owned process group is required for a multi-rank exchange")
        elif dist.get_world_size(group) != self.world_size or dist.get_rank(group) != self.rank:
            raise ValueError("token layout does not match the process group size/rank")


def scatter_heads_gather_tokens(
    tensor: torch.Tensor,
    group: dist.ProcessGroup | None,
    layout: HeadParallelLayout,
) -> torch.Tensor:
    """Exchange ``[local_tokens, heads, dim]`` for ``[total_tokens, local_heads, dim]``."""
    layout.validate_group(group)
    if tensor.ndim != 3 or tensor.shape[0] != layout.local_tokens:
        raise ValueError("input must be [local_tokens, heads, dim] matching the token layout")
    if tensor.shape[1] == 0 or tensor.shape[2] == 0 or tensor.shape[1] % layout.world_size:
        raise ValueError("positive head count must divide evenly across the group; head dimension must be positive")
    if layout.world_size == 1:
        return tensor

    sequence, heads, dim = tensor.shape
    local_heads = heads // layout.world_size
    # Destination rank first; each destination receives a contiguous head shard.
    send = tensor.reshape(sequence, layout.world_size, local_heads, dim).permute(1, 0, 2, 3).contiguous()
    output = tensor.new_empty((layout.total_tokens, local_heads, dim))
    if layout.total_tokens:
        row_width = local_heads * dim
        dist.all_to_all_single(
            output.view(-1),
            send.view(-1),
            output_split_sizes=[count * row_width for count in layout.token_counts],
            input_split_sizes=[sequence * row_width] * layout.world_size,
            group=group,
        )
    return output


def scatter_tokens_gather_heads(
    tensor: torch.Tensor,
    group: dist.ProcessGroup | None,
    layout: HeadParallelLayout,
) -> torch.Tensor:
    """Undo the exchange using the same token layout and process group."""
    layout.validate_group(group)
    if tensor.ndim != 3 or tensor.shape[0] != layout.total_tokens:
        raise ValueError("input must be [total_tokens, local_heads, dim] matching the token layout")
    if tensor.shape[1] == 0 or tensor.shape[2] == 0:
        raise ValueError("local head count and head dimension must be positive")
    if layout.world_size == 1:
        return tensor

    local_heads, dim = tensor.shape[1:]
    # Source rank first; restore its head shard after returning each token slice.
    output = tensor.new_empty((layout.world_size, layout.local_tokens, local_heads, dim))
    if layout.total_tokens:
        row_width = local_heads * dim
        dist.all_to_all_single(
            output.view(-1),
            tensor.contiguous().view(-1),
            output_split_sizes=[layout.local_tokens * row_width] * layout.world_size,
            input_split_sizes=[count * row_width for count in layout.token_counts],
            group=group,
        )
    return output.permute(1, 0, 2, 3).contiguous().view(layout.local_tokens, layout.world_size * local_heads, dim)
