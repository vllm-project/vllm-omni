# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Rank-major head bucket metadata, independent of communication backend.

A bucket contains a head slice for EACH destination rank, not just one
contiguous range of global heads. Weights are packed on CPU before device transfer.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HeadBucketPlan:
    query_heads: int
    kv_heads: int
    head_dim: int
    world_size: int
    buckets: int

    def __post_init__(self):
        values = (
            self.query_heads,
            self.kv_heads,
            self.head_dim,
            self.world_size,
            self.buckets,
        )
        if any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in values):
            raise ValueError("Head counts, dimensions, ranks and buckets must be positive integers")
        if self.query_heads % self.kv_heads:
            raise ValueError("Query heads must be an integer multiple of KV heads")
        if self.kv_heads % self.world_size or self.buckets > self.kv_heads // self.world_size:
            raise ValueError("Ranks must own complete KV groups and every bucket must be nonempty")

    def head_ranges(self, bucket: int, *, kv: bool = False) -> tuple[range, ...]:
        if not 0 <= bucket < self.buckets:
            raise IndexError(bucket)
        per_rank_kv = self.kv_heads // self.world_size
        base, remainder = divmod(per_rank_kv, self.buckets)
        start_kv = bucket * base + min(bucket, remainder)
        width_kv = base + int(bucket < remainder)
        ratio = 1 if kv else self.query_heads // self.kv_heads
        return tuple(
            range(
                (rank * per_rank_kv + start_kv) * ratio,
                (rank * per_rank_kv + start_kv + width_kv) * ratio,
            )
            for rank in range(self.world_size)
        )

    def channel_indices(self, bucket: int, *, kv: bool = False) -> tuple[int, ...]:
        return tuple(
            head * self.head_dim + dim
            for group in self.head_ranges(bucket, kv=kv)
            for head in group
            for dim in range(self.head_dim)
        )

    def qkv_indices(self, bucket: int) -> tuple[int, ...]:
        """Rows of a weight packed as [all Q rows; all K rows; all V rows]."""
        q = self.channel_indices(bucket)
        kv = self.channel_indices(bucket, kv=True)
        q_width = self.query_heads * self.head_dim
        kv_width = self.kv_heads * self.head_dim
        return q + tuple(q_width + i for i in kv) + tuple(q_width + kv_width + i for i in kv)

    def pack_qkv(self, tensor, *, restore=False):
        """Reorder dense CPU QKV weights/bias; restore reverses bucket packing."""
        rows = torch.tensor([i for b in range(self.buckets) for i in self.qkv_indices(b)])
        if tensor.device.type != "cpu" or not tensor.is_floating_point():
            raise ValueError("Pack dense CPU floating-point weights before device transfer")
        if tensor.ndim not in (1, 2) or tensor.shape[0] != rows.numel():
            raise ValueError("Expected [all Q rows; all K rows; all V rows]")
        if restore:
            return torch.empty_like(tensor).index_copy_(0, rows, tensor)
        return tensor.index_select(0, rows)

    def qkv_view(self, packed, bucket):
        """Device forwards only slice contiguous, prepacked bucket rows."""
        heads = self.head_ranges(bucket, kv=True)[0]
        stride = self.world_size * (self.query_heads // self.kv_heads + 2) * self.head_dim
        return packed.narrow(0, heads.start * stride, len(heads) * stride)
