# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.parallel.base import ParallelAttentionContext
from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D
from vllm_omni.diffusion.distributed.group_coordinator import SequenceParallelGroupCoordinator


@dataclass(frozen=True, slots=True)
class _UlyssesCtx(ParallelAttentionContext):
    """Per-forward context for Ulysses sequence-parallel attention."""

    ulysses_pg: dist.ProcessGroup
    scatter_idx: int
    gather_idx: int
    use_sync: bool
    joint_len: int = 0
    joint_strategy: str = "front"


class UlyssesParallelAttention:
    """Ulysses sequence-parallel strategy (all-to-all over seq/head dims).

    This preserves the semantics previously implemented in
    `Attention._forward_ulysses`:
    - If `AttentionMetadata.joint_*` is provided, joint_query/key/value are
      concatenated *after* all-to-all.
    - joint_key/value are assumed to be replicated across SP ranks and are sliced
      by ulysses head rank before concatenation.
    """

    def __init__(
        self,
        sp_group: SequenceParallelGroupCoordinator,
        scatter_idx: int,
        gather_idx: int,
        use_sync: bool,
    ) -> None:
        self._sp_group = sp_group
        self._ulysses_pg = sp_group.ulysses_group
        self._scatter_idx = scatter_idx
        self._gather_idx = gather_idx
        self._use_sync = use_sync

    @property
    def enabled(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "ulysses"

    def pre_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ):
        joint_tensor_query = joint_tensor_key = joint_tensor_value = None
        joint_strategy = "front"
        joint_len = 0

        if attn_metadata is not None:
            joint_tensor_query = attn_metadata.joint_query
            joint_tensor_key = attn_metadata.joint_key
            joint_tensor_value = attn_metadata.joint_value
            joint_strategy = attn_metadata.joint_strategy

        is_joint = False
        if joint_tensor_query is not None and joint_tensor_key is not None and joint_tensor_value is not None:
            supported_joint_strategy = ["front", "rear"]
            if joint_strategy not in supported_joint_strategy:
                raise ValueError(
                    f"joint_strategy: {joint_strategy} not supported."
                    f" supported joint strategy: {supported_joint_strategy}"
                )

            # Slice joint_query for this Ulysses rank
            # joint_query is (B, S, H, D). We split H (dim 2).
            ulysses_world_size = self._sp_group.ulysses_world_size
            ulysses_rank = self._sp_group.ulysses_rank
            attn_heads_per_ulysses_rank = joint_tensor_query.shape[-2] // ulysses_world_size

            # Note: We use the same heads for Q/K/V
            joint_tensor_query = joint_tensor_query[
                ...,
                attn_heads_per_ulysses_rank * ulysses_rank : attn_heads_per_ulysses_rank * (ulysses_rank + 1),
                :,
            ]

            joint_len = joint_tensor_query.shape[1]

            is_joint = True
        elif joint_tensor_query is None and joint_tensor_key is None and joint_tensor_value is None:
            pass
        else:
            raise ValueError("joint_query, joint_key, and joint_value should be None or not None simultaneously.")

        if is_joint:
            # Slice joint key/value heads for this ulysses rank.
            # Using same slicing logic as query
            attn_heads_per_ulysses_rank_kv = joint_tensor_key.shape[-2] // ulysses_world_size

            joint_tensor_key = joint_tensor_key[
                ...,
                attn_heads_per_ulysses_rank_kv * ulysses_rank : attn_heads_per_ulysses_rank_kv * (ulysses_rank + 1),
                :,
            ]
            joint_tensor_value = joint_tensor_value[
                ...,
                attn_heads_per_ulysses_rank_kv * ulysses_rank : attn_heads_per_ulysses_rank_kv * (ulysses_rank + 1),
                :,
            ]

            # Update metadata with sliced tensors so Ring attention can use them if needed
            if attn_metadata is not None:
                attn_metadata.joint_key = joint_tensor_key
                attn_metadata.joint_value = joint_tensor_value

        # (bs, seq_len/P, head_cnt, head_size) -> (bs, seq_len, head_cnt/P, head_size)
        query = SeqAllToAll4D.apply(self._ulysses_pg, query, self._scatter_idx, self._gather_idx, self._use_sync)
        key = SeqAllToAll4D.apply(self._ulysses_pg, key, self._scatter_idx, self._gather_idx, self._use_sync)
        value = SeqAllToAll4D.apply(self._ulysses_pg, value, self._scatter_idx, self._gather_idx, self._use_sync)

        if is_joint:
            # Concatenate joint query AFTER AllToAll
            # Image query is now (B, S, H/P, D). Joint query is (B, S_txt, H/P, D).
            # This is dimensionally consistent.
            if joint_strategy == "rear":
                query = torch.cat([query, joint_tensor_query], dim=1)
            else:
                query = torch.cat([joint_tensor_query, query], dim=1)

        # Check if Ring Attention is also active (Hybrid mode)
        # If Ring is active, we should NOT concatenate joint_key/value to k/v here.
        # Instead, they should remain in attn_metadata and be passed to the Ring kernel.
        use_ring = self._sp_group.ring_world_size > 1

        if is_joint and not use_ring:
            # Concatenate joint key/value after all-to-all ONLY for pure Ulysses (Local Attention).
            if joint_strategy == "front":
                key = torch.cat([joint_tensor_key, key], dim=1)
                value = torch.cat([joint_tensor_value, value], dim=1)
            else:  # "rear"
                key = torch.cat([key, joint_tensor_key], dim=1)
                value = torch.cat([value, joint_tensor_value], dim=1)

        ctx = _UlyssesCtx(
            name=self.name,
            ulysses_pg=self._ulysses_pg,
            scatter_idx=self._scatter_idx,
            gather_idx=self._gather_idx,
            use_sync=self._use_sync,
            joint_len=joint_len,
            joint_strategy=joint_strategy,
        )
        return query, key, value, attn_metadata, ctx

    def post_attention(self, attn_output: torch.Tensor, ctx: ParallelAttentionContext | None) -> torch.Tensor:
        assert isinstance(ctx, _UlyssesCtx), f"Unexpected ctx type: {type(ctx)!r}"

        # If we have joint tensors (Text), they were Head-Sliced.
        # The main sequence (Image) was Sequence-Sliced.
        # attn_output contains [Joint_Sliced | Image_Sliced] (if strategy='front').

        if ctx.joint_len > 0:
            joint_len = ctx.joint_len

            if ctx.joint_strategy == "front":
                output_joint = attn_output[:, :joint_len]
                output_img = attn_output[:, joint_len:]
            else:
                output_img = attn_output[:, :-joint_len]
                output_joint = attn_output[:, -joint_len:]

            # 1. Process Image part: Standard Ulysses Reverse (AllToAll)
            # (bs, seq_len, head_cnt/P, head_size) -> (bs, seq_len/P, head_cnt, head_size)
            # SeqAllToAll4D handles: Scatter gather_idx, Gather scatter_idx.
            # Forward: Scatter 2 (H), Gather 1 (S).
            # Reverse: Scatter 1 (S), Gather 2 (H).
            output_img = SeqAllToAll4D.apply(ctx.ulysses_pg, output_img, ctx.gather_idx, ctx.scatter_idx, ctx.use_sync)

            # 2. Process Joint part: AllGather on Heads
            # Input: (B, JointLen, H/P, D). Output: (B, JointLen, H, D).
            # AllGather along dim 2.
            gathered_joint = [torch.zeros_like(output_joint) for _ in range(dist.get_world_size(ctx.ulysses_pg))]
            dist.all_gather(gathered_joint, output_joint, group=ctx.ulysses_pg)
            output_joint = torch.cat(gathered_joint, dim=2)

            # 3. Recombine
            if ctx.joint_strategy == "front":
                return torch.cat([output_joint, output_img], dim=1)
            else:
                return torch.cat([output_img, output_joint], dim=1)

        # Standard Ulysses Reverse
        return SeqAllToAll4D.apply(ctx.ulysses_pg, attn_output, ctx.gather_idx, ctx.scatter_idx, ctx.use_sync)
