# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass, replace

import torch
import torch.distributed as dist

from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D


class CrossAllGatherKVParallelAttention(NoParallelAttention):
    """Gather sequence-sharded K/V, preserving either local or replicated Q.

    Ranks supply equal-sized K/V shards (pad and mask uneven lengths). Unlike
    self-attention, Q positions and lengths are independent of KV positions.
    Pair masks cover local KV columns; Q validity is never gathered.
    """

    def __init__(self, sp_group):
        self._sp_group = sp_group

    @property
    def enabled(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "allgather_kv"

    def pre_attention(self, query, key, value, attn_metadata):
        key = self._sp_group.all_gather(key, dim=1)
        value = self._sp_group.all_gather(value, dim=1)
        if attn_metadata is not None and attn_metadata.attn_mask is not None:
            mask = self._sp_group.all_gather(attn_metadata.attn_mask, dim=attn_metadata.attn_mask.ndim - 1)
            attn_metadata = replace(attn_metadata, attn_mask=mask)
        return query, key, value, attn_metadata, None


@dataclass
class _CrossHeadContext:
    group: dist.ProcessGroup


class ReplicatedQShardedKVParallelAttention(CrossAllGatherKVParallelAttention):
    """Reshard K/V to heads and compute disjoint replicated-Q heads per rank.

    Uses the same three-collective pattern as LTX2 video-to-audio attention.
    If GQA K/V heads cannot be split over SP ranks, gather K/V instead. Q is
    replicated, masks describe local KV columns, and output remains replicated.
    """

    @property
    def name(self) -> str:
        return "cross_reshard_kv"

    def pre_attention(self, query, key, value, attn_metadata):
        group = self._sp_group.device_group
        world = dist.get_world_size(group)
        rank = dist.get_rank(group)
        if query.shape[2] % world or key.shape[2] % world:
            return super().pre_attention(query, key, value, attn_metadata)
        heads = query.shape[2] // world
        query = query[:, :, rank * heads : (rank + 1) * heads].contiguous()
        key = SeqAllToAll4D.apply(group, key, 2, 1, False)
        value = SeqAllToAll4D.apply(group, value, 2, 1, False)
        if attn_metadata is not None and attn_metadata.attn_mask is not None:
            mask = self._sp_group.all_gather(attn_metadata.attn_mask, dim=attn_metadata.attn_mask.ndim - 1)
            if mask.ndim == 4 and mask.shape[1] != 1:
                mask = mask[:, rank * heads : (rank + 1) * heads]
            attn_metadata = replace(attn_metadata, attn_mask=mask)
        return query, key, value, attn_metadata, _CrossHeadContext(group)

    def post_attention(self, attn_output, ctx):
        if ctx is None:
            return attn_output
        local = attn_output.permute(2, 0, 1, 3).contiguous()
        gathered = torch.empty(
            (local.shape[0] * dist.get_world_size(ctx.group), *local.shape[1:]),
            device=local.device,
            dtype=local.dtype,
        )
        dist.all_gather_into_tensor(gathered, local, group=ctx.group)
        return gathered.permute(1, 2, 0, 3).contiguous()
