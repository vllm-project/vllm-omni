# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Derive each peer's packed layout from the shared demand masks."""

from typing import NamedTuple

import torch
from vllm.triton_utils import tl, triton


class PeerHop(NamedTuple):
    source: int
    destination: int
    slot: int


def peer_hops(rank: int, world: int, slots: int) -> tuple[PeerHop, ...]:
    return tuple(PeerHop((rank - hop) % world, (rank + hop) % world, (hop - 1) % slots) for hop in range(1, world))


@triton.jit
def _compact(
    gives, wants, plans, counts, rank, world: tl.constexpr, blocks: tl.constexpr, size: tl.constexpr, tile: tl.constexpr
):
    direction = tl.program_id(0)
    hop = tl.program_id(1)
    j = tl.arange(0, tile)
    if direction == 0:
        peer = (rank + hop + 1) % world
        keep = tl.load(gives + peer * size + j, j < size, other=False)
    else:
        peer = (rank + world - hop - 1) % world
        keep = tl.load(wants + peer * size + j, j < size, other=False)
    pos = tl.cumsum(keep.to(tl.int32)) - 1
    row = direction * (world - 1) + hop
    base = plans + row * 2 * size
    tl.store(base + pos, j // blocks, keep)
    tl.store(base + size + pos, j % blocks, keep)
    tl.store(counts + row, tl.sum(keep.to(tl.int32), 0))


class DemandManifest:
    """Store reusable scratch and compact dynamic masks into packed order."""

    def __init__(self, rank: int, world: int, heads: int, blocks: int, device: torch.device) -> None:
        self.rank, self.world, self.blocks = rank, world, blocks
        self.capacity = heads * blocks
        self.indices = torch.empty((2, world - 1, 2, self.capacity), dtype=torch.int32, device=device)
        self.counts = torch.empty((2, world - 1), dtype=torch.int32, device=device)

    def compact(self, gives: torch.Tensor, wants: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], ...]:
        _compact[(2, self.world - 1)](
            gives,
            wants,
            self.indices,
            self.counts,
            self.rank,
            self.world,
            self.blocks,
            self.capacity,
            triton.next_power_of_2(self.capacity),
        )
        # Read just the lengths needed to form the NCCL tensor slices.
        return tuple(
            tuple(self.indices[direction, hop, :, :count] for hop, count in enumerate(lengths))
            for direction, lengths in enumerate(self.counts.tolist())
        )


@triton.jit
def _compact_pairs(wants, plans, counts, rank, blocks: tl.constexpr, size: tl.constexpr, tile: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, tile)
    # wants is indexed by (consumer, owner, head, block). Contiguous
    # rank pairs (0,1) and (2,3) exchange through their opposite ranks.
    if row == 0:
        owner, first = rank, rank ^ 1
    elif row == 1:
        owner, first = rank ^ 1, rank
    elif row == 2:
        owner, first = rank, rank ^ 3
    elif row == 3:
        owner, first = rank ^ 2, rank ^ 1
    else:
        owner, first = rank ^ 3, rank
    prefix = tl.load(wants + (first * 4 + owner) * size + j, j < size, other=False)
    count = tl.sum(prefix.to(tl.int32), 0)
    pos = tl.cumsum(prefix.to(tl.int32)) - 1
    base = plans + row * 2 * size
    tl.store(base + pos, j // blocks, prefix)
    tl.store(base + size + pos, j % blocks, prefix)
    if row == 2 or row == 3:
        if row == 3:
            tl.store(counts + 5, count)
        # The forwarded demand occupies the prefix. Append only blocks
        # needed by the first receiver but absent from that prefix.
        peer = rank ^ 2 if row == 2 else rank
        suffix = tl.load(wants + (peer * 4 + owner) * size + j, j < size, other=False) & ~prefix
        pos = count + tl.cumsum(suffix.to(tl.int32)) - 1
        tl.store(base + pos, j // blocks, suffix)
        tl.store(base + size + pos, j % blocks, suffix)
        count += tl.sum(suffix.to(tl.int32), 0)
    tl.store(counts + row, count)


class PairRelayPlan(NamedTuple):
    local_send: torch.Tensor
    local_receive: torch.Tensor
    cross_send: torch.Tensor
    cross_receive: torch.Tensor
    forward_receive: torch.Tensor
    forward_count: int


class PairRelayManifest:
    """Compact a cross-pair union whose prefix is ready to forward."""

    def __init__(self, rank: int, heads: int, blocks: int, device: torch.device) -> None:
        self.rank, self.blocks = rank, blocks
        self.capacity = heads * blocks
        self.wants = torch.empty((4, 4, heads, blocks), dtype=torch.bool, device=device)
        self.indices = torch.empty((5, 2, self.capacity), dtype=torch.int32, device=device)
        self.counts = torch.empty(6, dtype=torch.int32, device=device)

    def compact(self) -> PairRelayPlan:
        _compact_pairs[(5,)](
            self.wants,
            self.indices,
            self.counts,
            self.rank,
            self.blocks,
            self.capacity,
            triton.next_power_of_2(self.capacity),
        )
        *lengths, forward_count = self.counts.tolist()
        local_send, local_receive, cross_send, cross_receive, forward_receive = (
            self.indices[row, :, :count] for row, count in enumerate(lengths)
        )
        return PairRelayPlan(local_send, local_receive, cross_send, cross_receive, forward_receive, forward_count)
