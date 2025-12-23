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


class UlyssesParallelAttention:
    """Ulysses sequence-parallel strategy (all-to-all over seq/head dims).

    This preserves the semantics previously implemented in
    `Attention._forward_ulysses`:
    - If `AttentionMetadata.joint_*` is provided, joint_query is concatenated to
      query *before* all-to-all; joint_key/value are concatenated *after* all-to-all.
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
        joint_strategy = None

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
            if joint_strategy == "rear":
                query = torch.cat([query, joint_tensor_query], dim=1)
            else:
                query = torch.cat([joint_tensor_query, query], dim=1)
            is_joint = True
        elif joint_tensor_query is None and joint_tensor_key is None and joint_tensor_value is None:
            pass
        else:
            raise ValueError("joint_query, joint_key, and joint_value should be None or not None simultaneously.")

        if is_joint:
            # Slice joint key/value heads for this ulysses rank.
            ulysses_world_size = self._sp_group.ulysses_world_size
            ulysses_rank = self._sp_group.ulysses_rank
            attn_heads_per_ulysses_rank = joint_tensor_key.shape[-2] // ulysses_world_size
            joint_tensor_key = joint_tensor_key[
                ...,
                attn_heads_per_ulysses_rank * ulysses_rank : attn_heads_per_ulysses_rank * (ulysses_rank + 1),
                :,
            ]
            joint_tensor_value = joint_tensor_value[
                ...,
                attn_heads_per_ulysses_rank * ulysses_rank : attn_heads_per_ulysses_rank * (ulysses_rank + 1),
                :,
            ]

        # (bs, seq_len/P, head_cnt, head_size) -> (bs, seq_len, head_cnt/P, head_size)
        query = SeqAllToAll4D.apply(self._ulysses_pg, query, self._scatter_idx, self._gather_idx, self._use_sync)
        key = SeqAllToAll4D.apply(self._ulysses_pg, key, self._scatter_idx, self._gather_idx, self._use_sync)
        value = SeqAllToAll4D.apply(self._ulysses_pg, value, self._scatter_idx, self._gather_idx, self._use_sync)

        if is_joint:
            # Concatenate joint key/value after all-to-all (matches previous implementation).
            if joint_strategy == "front":
                key = torch.cat([joint_tensor_key, key], dim=1)
                value = torch.cat([joint_tensor_value, value], dim=1)
            else:  # "rear"
                key = torch.cat([key, joint_tensor_key], dim=1)
                value = torch.cat([value, joint_tensor_value], dim=1)

        attn_mask = attn_metadata.attn_mask
        if attn_mask is not None:
            assert attn_mask.ndim == 2, f"attn_mask.ndim != 2, {attn_mask.ndim}"
            if is_joint:
                # `attn_mask` is a per-token padding mask (bool, True=valid) for the
                # *local* pre-all-to-all sequence: [joint(text), local(image)].
                #
                # After Ulysses all-to-all:
                # - image seq is gathered from all ranks -> global image seq
                # - joint(text) query is replicated across ranks before all-to-all,
                #   so it appears `world_size` times along query seq
                joint_query_len = joint_tensor_query.shape[1]
                joint_attn_mask = (
                    attn_mask[:, :joint_query_len] if joint_strategy == "front" else attn_mask[:, -joint_query_len:]
                )
                curr_attn_mask = (
                    attn_mask[:, joint_query_len:] if joint_strategy == "front" else attn_mask[:, :-joint_query_len]
                )

                ulysses_world_size = dist.get_world_size(self._ulysses_pg)

                # Gather image part to match post-all-to-all global image sequence length.
                gathered_curr_mask = [torch.zeros_like(curr_attn_mask) for _ in range(ulysses_world_size)]
                dist.all_gather(gathered_curr_mask, curr_attn_mask, group=self._ulysses_pg)
                curr_attn_mask = torch.cat(gathered_curr_mask, dim=-1)

                replicated_joint_attn_mask = torch.cat([joint_attn_mask] * ulysses_world_size, dim=-1)

                query_attn_mask = (
                    torch.cat([replicated_joint_attn_mask, curr_attn_mask], dim=-1)
                    if joint_strategy == "front"
                    else torch.cat([curr_attn_mask, replicated_joint_attn_mask], dim=-1)
                )  # query uses replicated joint query tensor
                key_attn_mask = (
                    torch.cat([joint_attn_mask, curr_attn_mask], dim=-1)
                    if joint_strategy == "front"
                    else torch.cat([curr_attn_mask, joint_attn_mask], dim=-1)
                )

                assert query_attn_mask.shape[-1] == query.shape[1], (
                    f"query_attn_mask.shape[-1] != query.shape[1], {query_attn_mask.shape[-1]} != {query.shape[1]}"
                )
                assert key_attn_mask.shape[-1] == key.shape[1], (
                    f"key_attn_mask.shape[-1] != key.shape[1], {key_attn_mask.shape[-1]} != {key.shape[1]}"
                )

                # Build full attention mask for SDPA: (bs, 1, query_len, key_len)
                # For torch SDPA bool masks: True means "keep/attend".
                attn_metadata.attn_mask = (
                    query_attn_mask.to(torch.bool)[:, None, :, None] & key_attn_mask.to(torch.bool)[:, None, None, :]
                )

            else:
                gathered_mask = [torch.zeros_like(attn_mask) for _ in range(dist.get_world_size(self._ulysses_pg))]
                dist.all_gather(gathered_mask, attn_mask, group=self._ulysses_pg)
                attn_mask = torch.cat(gathered_mask, dim=-1)  # (bs, seq_len/P*P)
                assert attn_mask.shape[-1] == query.shape[1], (
                    f"attn_mask.shape[-1] != query.shape[1], {attn_mask.shape[-1]} != {query.shape[1]}"
                )
                attn_metadata.attn_mask = attn_mask

        ctx = _UlyssesCtx(
            name=self.name,
            ulysses_pg=self._ulysses_pg,
            scatter_idx=self._scatter_idx,
            gather_idx=self._gather_idx,
            use_sync=self._use_sync,
        )
        return query, key, value, attn_metadata, ctx

    def post_attention(self, attn_output: torch.Tensor, ctx: ParallelAttentionContext | None) -> torch.Tensor:
        assert isinstance(ctx, _UlyssesCtx), f"Unexpected ctx type: {type(ctx)!r}"
        # Reverse: (bs, seq_len, head_cnt/P, head_size) -> (bs, seq_len/P, head_cnt, head_size)
        return SeqAllToAll4D.apply(ctx.ulysses_pg, attn_output, ctx.gather_idx, ctx.scatter_idx, ctx.use_sync)
