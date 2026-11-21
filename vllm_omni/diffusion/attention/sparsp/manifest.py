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
        self.host_counts = torch.empty(self.counts.shape, dtype=torch.int32, device="cpu", pin_memory=True)

    def compact(self, gives: torch.Tensor, wants: torch.Tensor) -> None:
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
        self.host_counts.copy_(self.counts, non_blocking=True)

    def plan(self) -> tuple[tuple[torch.Tensor, ...], ...]:
        """Read packet lengths after the planning stream has completed."""
        return tuple(
            tuple(self.indices[direction, hop, :, :count] for hop, count in enumerate(lengths))
            for direction, lengths in enumerate(self.host_counts.tolist())
        )


@triton.jit
def _cross_pair_matching(wants, rank, size: tl.constexpr, tile: tl.constexpr):
    j = tl.arange(0, tile)
    partner = rank ^ 1
    # Each socket independently picks one of its two perfect matchings to the
    # other socket. The cross-socket union is unchanged. Minimize the number
    # of blocks needed by the consumer that is reached via the local relay.
    demand = tl.load(wants + ((rank ^ 3) * 4 + rank) * size + j, j < size, other=False)
    demand_partner = tl.load(wants + ((partner ^ 3) * 4 + partner) * size + j, j < size, other=False)
    alternate = tl.load(wants + ((rank ^ 2) * 4 + rank) * size + j, j < size, other=False)
    alternate_partner = tl.load(wants + ((partner ^ 2) * 4 + partner) * size + j, j < size, other=False)
    decrease = (
        demand.to(tl.int32) + demand_partner.to(tl.int32) - alternate.to(tl.int32) - alternate_partner.to(tl.int32)
    )
    return 2 + (tl.sum(decrease, 0) > 0).to(tl.int32)


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
        outgoing = _cross_pair_matching(wants, rank, size, tile)
        tl.store(counts + 6, outgoing)
        owner, first = rank, rank ^ outgoing ^ 1
    elif row == 3:
        incoming = _cross_pair_matching(wants, rank ^ 2, size, tile)
        tl.store(counts + 7, incoming)
        owner, first = rank ^ incoming, rank ^ 1
    else:
        incoming = _cross_pair_matching(wants, rank ^ 2, size, tile)
        owner, first = rank ^ incoming ^ 1, rank
    prefix = tl.load(wants + (first * 4 + owner) * size + j, j < size, other=False)
    count = tl.sum(prefix.to(tl.int32), 0)
    pos = tl.cumsum(prefix.to(tl.int32)) - 1
    base = plans + row * 2 * size
    tl.store(base + pos, j // blocks, prefix)
    tl.store(base + size + pos, j % blocks, prefix)
    if row == 2 or row == 3:
        if row == 3:
            tl.store(counts + 5, count)
        peer = first ^ 1
        # The forwarded demand occupies the prefix. Append only blocks
        # needed by the first receiver but absent from that prefix.
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
    cross_destination: int
    cross_source: int


class PairRelayManifest:
    """Compact a cross-pair union whose prefix is ready to forward."""

    def __init__(self, rank: int, heads: int, blocks: int, device: torch.device) -> None:
        self.rank, self.blocks = rank, blocks
        self.capacity = heads * blocks
        self.wants = torch.empty((4, 4, heads, blocks), dtype=torch.bool, device=device)
        self.indices = torch.empty((5, 2, self.capacity), dtype=torch.int32, device=device)
        self.counts = torch.empty(8, dtype=torch.int32, device=device)
        self.host_counts = torch.empty(self.counts.shape, dtype=torch.int32, device="cpu", pin_memory=True)

    def compact(self) -> None:
        _compact_pairs[(5,)](
            self.wants,
            self.indices,
            self.counts,
            self.rank,
            self.blocks,
            self.capacity,
            triton.next_power_of_2(self.capacity),
        )
        self.host_counts.copy_(self.counts, non_blocking=True)

    def plan(self) -> PairRelayPlan:
        """Read packet lengths after the planning stream has completed."""
        *lengths, forward_count, outgoing, incoming = self.host_counts.tolist()
        local_send, local_receive, cross_send, cross_receive, forward_receive = (
            self.indices[row, :, :count] for row, count in enumerate(lengths)
        )
        return PairRelayPlan(
            local_send,
            local_receive,
            cross_send,
            cross_receive,
            forward_receive,
            forward_count,
            self.rank ^ outgoing,
            self.rank ^ incoming,
        )
