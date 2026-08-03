# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shard-layout arithmetic for BOOGU Image sequence parallelism.

BOOGU splits three independent sequences (noise image, reference images, and
instruction context) at one SP boundary. The framework records each sequence's
pre-padding global length under a ``shard_group`` key; everything else -- how
that global length maps onto a given rank's shard -- is arithmetic, so every
rank can derive the layout of *all* ranks without communicating.

That property is what lets the model build attention masks for the global,
rank-concatenated sequence that Ulysses produces after its all-to-all, instead
of all-gathering rank-local masks inside the attention layer.
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm_omni.diffusion.forward_context import get_sp_shard_original_seq_len


@dataclass(frozen=True, slots=True)
class ShardLayout:
    """How one globally padded sequence maps onto the SP ranks."""

    original_seq_len: int
    world_size: int
    rank: int

    @classmethod
    def resolve(cls, shard_group: str, *, local_seq_len: int) -> ShardLayout:
        """Read a boundary's layout from the ForwardContext.

        Falls back to an unsharded layout (world_size=1) when SP is off or the
        tensor was never split, in which case `local_seq_len` is the whole
        sequence.
        """
        from vllm_omni.diffusion.distributed.parallel_state import (
            get_sequence_parallel_rank,
            get_sequence_parallel_world_size,
        )

        original_seq_len = get_sp_shard_original_seq_len(shard_group)
        if original_seq_len is None:
            return cls(original_seq_len=local_seq_len, world_size=1, rank=0)
        return cls(
            original_seq_len=original_seq_len,
            world_size=get_sequence_parallel_world_size(),
            rank=get_sequence_parallel_rank(),
        )

    @property
    def padded_seq_len(self) -> int:
        ws = self.world_size
        return ((self.original_seq_len + ws - 1) // ws) * ws

    @property
    def local_seq_len(self) -> int:
        return self.padded_seq_len // self.world_size

    @property
    def padding_size(self) -> int:
        return self.padded_seq_len - self.original_seq_len

    def bounds(self, rank: int) -> tuple[int, int]:
        """Half-open [start, end) span of the global sequence owned by `rank`."""
        start = rank * self.local_seq_len
        return start, start + self.local_seq_len

    def valid_lengths(self, global_lengths: list[int], *, rank: int) -> list[int]:
        """Per-sample valid prefix lengths clipped to `rank`'s shard."""
        start, end = self.bounds(rank)
        return [max(0, min(int(length), end) - start) for length in global_lengths]

    def segment_lengths(self, global_segments: list[list[int]], *, rank: int) -> list[list[int]]:
        """Per-sample contiguous segments intersected with `rank`'s shard."""
        start, end = self.bounds(rank)
        local: list[list[int]] = []
        for sample in global_segments:
            sample_local: list[int] = []
            offset = 0
            for length in sample:
                segment_end = offset + int(length)
                sample_local.append(max(0, min(segment_end, end) - max(offset, start)))
                offset = segment_end
            local.append(sample_local)
        return local
