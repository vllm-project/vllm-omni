# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm_omni.diffusion.attention.parallel.base import ParallelAttentionContext
from vllm_omni.diffusion.distributed.group_coordinator import (
    SequenceParallelGroupCoordinator,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata


@dataclass(frozen=True, slots=True)
class _AllGatherKVCtx(ParallelAttentionContext):
    """Per-forward context for AllGather-KV sequence-parallel attention.

    Captured in pre_attention and consumed by post_attention to split the
    local attention output back into the joint (text) and image shards so
    the image shard can be AllGathered back to a full sequence.
    """

    joint_len: int = 0
    joint_strategy: str = "front"
    img_seq_local: int = 0


class AllGatherKVParallelAttention:
    """AllGather-KV sequence-parallel strategy (causal=False only).

    Each rank holds 1/P of Q (pre_processor sequence-split). AllGather
    collects full K/V on every rank, then a single local attention kernel
    computes Q_local x K_full. v1 is mutually exclusive with Ulysses/Ring.
    """

    def __init__(
        self,
        sp_group: SequenceParallelGroupCoordinator,
        scatter_idx: int = 2,
        gather_idx: int = 1,
    ) -> None:
        self._sp_group = sp_group
        self._ag_group = sp_group.ulysses_group
        self._sp_size = sp_group.ulysses_world_size
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

    @property
    def enabled(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "allgather_kv"

    def pre_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ):
        joint_q = joint_k = joint_v = None
        joint_strategy = "front"
        if attn_metadata is not None:
            joint_q = attn_metadata.joint_query
            joint_k = attn_metadata.joint_key
            joint_v = attn_metadata.joint_value
            joint_strategy = attn_metadata.joint_strategy or "front"

        k_img_full = self._sp_group.all_gather(key, dim=1)
        v_img_full = self._sp_group.all_gather(value, dim=1)

        if joint_k is not None:
            if joint_strategy == "front":
                k_full = torch.cat([joint_k, k_img_full], dim=1)
                v_full = torch.cat([joint_v, v_img_full], dim=1)
            else:
                k_full = torch.cat([k_img_full, joint_k], dim=1)
                v_full = torch.cat([v_img_full, joint_v], dim=1)
        else:
            k_full, v_full = k_img_full, v_img_full

        if joint_q is not None:
            if joint_strategy == "front":
                q_local = torch.cat([joint_q, query], dim=1)
            else:
                q_local = torch.cat([query, joint_q], dim=1)
        else:
            q_local = query

        ctx = _AllGatherKVCtx(
            name=self.name,
            joint_len=joint_q.shape[1] if joint_q is not None else 0,
            joint_strategy=joint_strategy,
            img_seq_local=query.shape[1],
        )
        return q_local, k_full, v_full, attn_metadata, ctx

    def post_attention(
        self,
        attn_output: torch.Tensor,
        ctx: ParallelAttentionContext | None,
    ) -> torch.Tensor:
        # Return the LOCAL image shard, not a full-sequence gather.
        # Each rank holds 1/P of Q, so attention output is naturally a local
        # shard (B, img_seq_local, H, D). Multi-layer models expect every
        # layer's output to remain sharded along the seq dim; the final
        # full-sequence aggregation happens in the model-level post_processor,
        # not inside each attention layer.
        assert ctx is not None
        if ctx.joint_len > 0:
            if ctx.joint_strategy == "front":
                joint_out, img_out_local = attn_output.split([ctx.joint_len, ctx.img_seq_local], dim=1)
                return torch.cat([joint_out, img_out_local], dim=1)
            img_out_local, joint_out = attn_output.split([ctx.img_seq_local, ctx.joint_len], dim=1)
            return torch.cat([img_out_local, joint_out], dim=1)
        return attn_output
