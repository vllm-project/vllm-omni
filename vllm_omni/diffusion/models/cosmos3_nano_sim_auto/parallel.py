# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Strict Ulysses context parallelism for Cosmos3-Nano-Sim-Auto.

The causal planner addresses tokens in global chunk coordinates and every
maskless pass is independent per head, so Ulysses needs no plan changes. The
GEN layers run on a zero-padded token shard of the packed chunk; one
all-to-all per layer gives every rank all of the chunk's queries, keys and
values for one contiguous head group, the unchanged executor attends them
against the same head group of the ring, the captions and cached controls,
and a second all-to-all returns the output to the token shard. The paged ring,
the caption pool and cached control K/V therefore hold every token for the
rank's head group, so the ring geometry (frame unit, block tables, write slots)
is the same as on one GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True, eq=False)
class UlyssesGroup:
    """The rank's Ulysses process group; head groups follow rank order."""

    group: Any
    rank: int
    world_size: int

    def __post_init__(self) -> None:
        if self.world_size < 2 or not 0 <= self.rank < self.world_size:
            raise ValueError(f"Invalid Ulysses rank {self.rank} of {self.world_size}.")

    def shard(self, num_tokens: int) -> SequenceShard:
        return SequenceShard(self, num_tokens)

    def local_heads(self, heads: int) -> int:
        if heads % self.world_size:
            raise ValueError(f"{heads} heads are not divisible by the Ulysses degree {self.world_size}.")
        return heads // self.world_size

    def head_slice(self, heads: int) -> slice:
        """This rank's contiguous group of ``heads`` (TP-local) heads."""
        size = self.local_heads(heads)
        return slice(self.rank * size, (self.rank + 1) * size)


@dataclass(frozen=True, eq=False)
class SequenceShard:
    """One forward's token split: ``ceil(num_tokens / U)`` tokens per rank, zero-padded at the end."""

    ulysses: UlyssesGroup
    num_tokens: int

    def __post_init__(self) -> None:
        if self.num_tokens < 1:
            raise ValueError("A Ulysses shard needs at least one token.")

    @property
    def local_tokens(self) -> int:
        return -(-self.num_tokens // self.ulysses.world_size)

    @property
    def padded_tokens(self) -> int:
        return self.local_tokens * self.ulysses.world_size

    def split(self, tensor: torch.Tensor) -> torch.Tensor:
        """This rank's ``[B, local_tokens, ...]`` slice of a ``[B, num_tokens, ...]`` tensor."""
        if tensor.shape[1] != self.num_tokens:
            raise ValueError(f"Expected {self.num_tokens} tokens to shard, got {tensor.shape[1]}.")
        start = self.ulysses.rank * self.local_tokens
        # A copy, so the full-sequence tensor is not kept alive through the GEN layers.
        return _pad_tokens(tensor, self.padded_tokens).narrow(1, start, self.local_tokens).contiguous()

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-gather ``[B, local_tokens, ...]`` shards into ``[B, num_tokens, ...]``."""
        self._check_local(tensor)
        parts = [torch.empty_like(tensor) for _ in range(self.ulysses.world_size)]
        dist.all_gather(parts, tensor.contiguous(), group=self.ulysses.group)
        return torch.cat(parts, dim=1)[:, : self.num_tokens]

    def heads_from_tokens(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``[B, local_tokens, H, D]`` Q/K/V -> ``[B, num_tokens, H / U, D]`` for this rank's head group.

        Q, K and V travel in one all-to-all: each destination rank's query, key
        and value head groups are packed next to each other, so the exchange is
        a pure permutation of three per-tensor exchanges and keeps the GQA
        grouping (query group ``r`` reads KV group ``r``).
        """
        for tensor in (q, k, v):
            self._check_local(tensor)
        world = self.ulysses.world_size
        batch, tokens, _, head_dim = q.shape
        if k.shape != v.shape or k.shape[-1] != head_dim:
            raise ValueError("Ulysses K/V must match each other and the query head size.")
        q_heads, kv_heads = self.ulysses.local_heads(q.shape[2]), self.ulysses.local_heads(k.shape[2])
        packed = torch.cat(
            [
                q.reshape(batch, tokens, world, q_heads, head_dim),
                k.reshape(batch, tokens, world, kv_heads, head_dim),
                v.reshape(batch, tokens, world, kv_heads, head_dim),
            ],
            dim=3,
        ).reshape(batch, tokens, world * (q_heads + 2 * kv_heads), head_dim)
        full = _all_to_all(packed, self.ulysses.group, scatter=2, gather=1)[:, : self.num_tokens]
        q, k, v = full.split([q_heads, kv_heads, kv_heads], dim=2)
        return q.contiguous(), k.contiguous(), v.contiguous()

    def tokens_from_heads(self, output: torch.Tensor) -> torch.Tensor:
        """``[B, num_tokens, H / U, D]`` attention output -> ``[B, local_tokens, H, D]``."""
        if output.shape[1] != self.num_tokens:
            raise ValueError(f"Expected {self.num_tokens} attention rows, got {output.shape[1]}.")
        return _all_to_all(_pad_tokens(output, self.padded_tokens), self.ulysses.group, scatter=1, gather=2)

    def _check_local(self, tensor: torch.Tensor) -> None:
        if tensor.shape[1] != self.local_tokens:
            raise ValueError(
                f"Expected {self.local_tokens} local tokens ({self.num_tokens} tokens / Ulysses "
                f"{self.ulysses.world_size}), got {tensor.shape[1]}."
            )


def _pad_tokens(tensor: torch.Tensor, tokens: int) -> torch.Tensor:
    padding = tokens - tensor.shape[1]
    if padding == 0:
        return tensor
    return torch.cat([tensor, tensor.new_zeros(tensor.shape[0], padding, *tensor.shape[2:])], dim=1)


def _all_to_all(tensor: torch.Tensor, group: Any, *, scatter: int, gather: int) -> torch.Tensor:
    from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D

    return SeqAllToAll4D.apply(group, tensor.contiguous(), scatter, gather, False)


def current_ulysses_group() -> UlyssesGroup | None:
    """The worker's Ulysses group, or ``None`` without one (degree 1 or no distributed state)."""
    from vllm_omni.diffusion.distributed import parallel_state

    if not parallel_state.model_parallel_is_initialized():
        return None
    size = parallel_state.get_ulysses_parallel_world_size()
    if size <= 1:
        return None
    return UlyssesGroup(parallel_state.get_sp_group().ulysses_group, parallel_state.get_ulysses_parallel_rank(), size)


__all__ = ["SequenceShard", "UlyssesGroup", "current_ulysses_group"]
