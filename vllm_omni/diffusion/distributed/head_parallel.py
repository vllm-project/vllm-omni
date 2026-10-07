# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2025-2026 SandAI. All Rights Reserved.

"""Packed token/head exchanges over caller-owned parallel groups."""

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class HeadParallelLayout:
    """Token counts in process-group rank order, shared by both exchanges.

    Every rank must provide the same split tuple. Counts describe input tokens,
    not routed expert rows. Head padding and replicated-token handling belong
    to the caller; heads must be evenly partitioned within this group.

    ``replica_counts`` lets several ranks own the same head shard.  With ``R``
    parts per rank, rank ``r`` owns head shard ``r % (world_size // R)`` for
    part ``r // (world_size // R)`` of every rank's tokens, and
    ``replica_counts[r]`` splits rank ``r``'s tokens into those ``R``
    contiguous parts.  Without it each rank owns one shard for all tokens.
    """

    token_counts: tuple[int, ...]
    rank: int
    replica_counts: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.token_counts, tuple) or not self.token_counts:
            raise ValueError("token_counts must be a nonempty tuple in process-group rank order")
        if any(type(count) is not int or count < 0 for count in self.token_counts):
            raise ValueError("token_counts must contain nonnegative integers")
        if type(self.rank) is not int or not 0 <= self.rank < self.world_size:
            raise ValueError("rank must index token_counts")
        if self.replica_counts is None:
            return
        if (
            not isinstance(self.replica_counts, tuple)
            or len(self.replica_counts) != self.world_size
            or any(not isinstance(parts, tuple) or not parts for parts in self.replica_counts)
            or len({len(parts) for parts in self.replica_counts}) != 1
            or self.world_size % self.replicas
        ):
            raise ValueError("replica_counts must give every rank the same number of parts, dividing the group size")
        if any(
            any(type(count) is not int or count < 0 for count in parts) or sum(parts) != tokens
            for parts, tokens in zip(self.replica_counts, self.token_counts)
        ):
            raise ValueError("replica_counts must split every token count into nonnegative parts")

    @property
    def world_size(self) -> int:
        return len(self.token_counts)

    @property
    def replicas(self) -> int:
        return 1 if self.replica_counts is None else len(self.replica_counts[0])

    @property
    def shards(self) -> int:
        return self.world_size // self.replicas

    @property
    def local_tokens(self) -> int:
        return self.token_counts[self.rank]

    @property
    def total_tokens(self) -> int:
        return sum(self.token_counts)

    @property
    def local_parts(self) -> tuple[int, ...]:
        return (self.local_tokens,) if self.replica_counts is None else self.replica_counts[self.rank]

    @property
    def owned_counts(self) -> tuple[int, ...]:
        """Tokens this rank's head shard receives from each rank."""

        if self.replica_counts is None:
            return self.token_counts
        replica = self.rank // self.shards
        return tuple(parts[replica] for parts in self.replica_counts)

    @property
    def owned_tokens(self) -> int:
        return sum(self.owned_counts)

    def validate_group(self, group: dist.ProcessGroup | None) -> None:
        if group is None:
            if self.world_size != 1:
                raise ValueError("a caller-owned process group is required for a multi-rank exchange")
        elif dist.get_world_size(group) != self.world_size or dist.get_rank(group) != self.rank:
            raise ValueError("token layout does not match the process group size/rank")


@lru_cache(maxsize=16)
def replica_token_counts(token_counts: tuple[int, ...], replicas: int) -> tuple[tuple[int, ...], ...]:
    """Split each rank's tokens into ``replicas`` parts for :class:`HeadParallelLayout`.

    Parts start balanced; whole tokens then move between the parts of the
    leading ranks until replica ``i`` receives exactly the tokens of ranks
    ``[i * shards, (i + 1) * shards)``.  Every head owner therefore holds as
    many tokens as an exchange within its own run of ``shards`` ranks gives it.
    """

    if replicas < 1 or len(token_counts) % replicas:
        raise ValueError("replicas must divide the group size")
    shards = len(token_counts) // replicas
    parts = [[count // replicas + int(part < count % replicas) for part in range(replicas)] for count in token_counts]
    excess = [
        sum(rank_parts[part] for rank_parts in parts) - sum(token_counts[part * shards : (part + 1) * shards])
        for part in range(replicas)
    ]
    for rank_parts in parts:
        for give in range(replicas):
            for take in range(replicas):
                moved = min(excess[give], -excess[take], rank_parts[give])
                if moved > 0:
                    rank_parts[give] -= moved
                    rank_parts[take] += moved
                    excess[give] -= moved
                    excess[take] += moved
    return tuple(tuple(rank_parts) for rank_parts in parts)


def scatter_heads_gather_tokens(
    tensor: torch.Tensor,
    group: dist.ProcessGroup | None,
    layout: HeadParallelLayout,
) -> torch.Tensor:
    """Exchange ``[local_tokens, heads, dim]`` for ``[owned_tokens, local_heads, dim]``."""
    layout.validate_group(group)
    if tensor.ndim != 3 or tensor.shape[0] != layout.local_tokens:
        raise ValueError("input must be [local_tokens, heads, dim] matching the token layout")
    if tensor.shape[1] == 0 or tensor.shape[2] == 0 or tensor.shape[1] % layout.shards:
        raise ValueError("positive head count must divide evenly across the group; head dimension must be positive")
    if layout.world_size == 1:
        return tensor
    if layout.replicas > 1:
        return _scatter_heads_to_replicas(tensor, group, layout)

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
    if tensor.ndim != 3 or tensor.shape[0] != layout.owned_tokens:
        raise ValueError("input must be [owned_tokens, local_heads, dim] matching the token layout")
    if tensor.shape[1] == 0 or tensor.shape[2] == 0:
        raise ValueError("local head count and head dimension must be positive")
    if layout.world_size == 1:
        return tensor
    if layout.replicas > 1:
        return _gather_heads_from_replicas(tensor, group, layout)

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


def _scatter_heads_to_replicas(
    tensor: torch.Tensor,
    group: dist.ProcessGroup,
    layout: HeadParallelLayout,
) -> torch.Tensor:
    sequence, heads, dim = tensor.shape
    shards, parts = layout.shards, layout.local_parts
    local_heads = heads // shards
    # Destination rank ``part * shards + shard`` receives that head shard of
    # that token part, so the buffer holds [part, shard, part tokens] blocks.
    if len(set(parts)) == 1:
        send = tensor.reshape(len(parts), parts[0], shards, local_heads, dim).permute(0, 2, 1, 3, 4).contiguous()
    else:
        send = tensor.new_empty((sequence, heads, dim))
        start = 0
        for count in parts:
            block = send.view(-1)[start * heads * dim : (start + count) * heads * dim]
            block.view(shards, count, local_heads, dim).copy_(
                tensor[start : start + count].reshape(count, shards, local_heads, dim).permute(1, 0, 2, 3)
            )
            start += count
    output = tensor.new_empty((layout.owned_tokens, local_heads, dim))
    if layout.total_tokens:
        row_width = local_heads * dim
        dist.all_to_all_single(
            output.view(-1),
            send.view(-1),
            output_split_sizes=[count * row_width for count in layout.owned_counts],
            input_split_sizes=[count * row_width for count in parts for _ in range(shards)],
            group=group,
        )
    return output


def _gather_heads_from_replicas(
    tensor: torch.Tensor,
    group: dist.ProcessGroup,
    layout: HeadParallelLayout,
) -> torch.Tensor:
    local_heads, dim = tensor.shape[1:]
    shards, parts = layout.shards, layout.local_parts
    heads = shards * local_heads
    # Source rank ``part * shards + shard`` returns that head shard of that token part.
    received = tensor.new_empty((layout.local_tokens * heads * dim,))
    if layout.total_tokens:
        row_width = local_heads * dim
        dist.all_to_all_single(
            received,
            tensor.contiguous().view(-1),
            output_split_sizes=[count * row_width for count in parts for _ in range(shards)],
            input_split_sizes=[count * row_width for count in layout.owned_counts],
            group=group,
        )
    if len(set(parts)) == 1:
        return (
            received.view(len(parts), shards, parts[0], local_heads, dim)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(layout.local_tokens, heads, dim)
        )
    output = tensor.new_empty((layout.local_tokens, heads, dim))
    start = 0
    for count in parts:
        block = received[start * heads * dim : (start + count) * heads * dim]
        output[start : start + count].view(count, shards, local_heads, dim).copy_(
            block.view(shards, count, local_heads, dim).permute(1, 0, 2, 3)
        )
        start += count
    return output
