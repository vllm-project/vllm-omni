# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from vllm_omni.diffusion.attention.backends.abstract import QueryRange
from vllm_omni.diffusion.attention.parallel.base import ParallelAttentionContext
from vllm_omni.diffusion.distributed.group_coordinator import (
    SequenceParallelGroupCoordinator,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata


@dataclass(frozen=True, slots=True)
class _AllGatherKVCtx(ParallelAttentionContext):
    pass


class AllGatherKVParallelAttention:
    """Compute local Q against AllGathered K/V."""

    def __init__(
        self,
        sp_group: SequenceParallelGroupCoordinator,
    ) -> None:
        self._sp_group = sp_group
        self._allgather_group = sp_group.allgather_group
        self._sp_size = sp_group.allgather_world_size
        self._sp_rank = sp_group.allgather_rank

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
        if joint_strategy not in {"front", "rear"}:
            raise ValueError(f"Unsupported joint_strategy: {joint_strategy!r}")

        k_img_full = self._sp_group.all_gather(key, dim=1, group=self._allgather_group)
        v_img_full = self._sp_group.all_gather(value, dim=1, group=self._allgather_group)

        if joint_k is not None:
            if joint_k.shape[2] != key.shape[2]:
                raise ValueError(
                    "AllGather-KV SP expects joint_k to be GQA-compressed with the "
                    f"same num_kv_heads as the image shard, got joint_k.heads="
                    f"{joint_k.shape[2]} vs key.heads={key.shape[2]}."
                )
            if joint_strategy == "front":
                k_full = torch.cat([joint_k, k_img_full], dim=1)
                v_full = torch.cat([joint_v, v_img_full], dim=1)
            else:
                k_full = torch.cat([k_img_full, joint_k], dim=1)
                v_full = torch.cat([v_img_full, joint_v], dim=1)
        else:
            k_full, v_full = k_img_full, v_img_full

        joint_len = joint_q.shape[1] if joint_q is not None else 0
        logical_q_full_len = joint_len + k_img_full.shape[1]
        query_global_offset = k_full.shape[1] - logical_q_full_len
        if query_global_offset < 0:
            raise ValueError(
                "AllGather-KV SP global K length is shorter than the logical Q length: "
                f"k_len={k_full.shape[1]}, logical_q_len={logical_q_full_len}."
            )

        if joint_q is not None:
            if joint_strategy == "front":
                q_local = torch.cat([joint_q, query], dim=1)
            else:
                q_local = torch.cat([query, joint_q], dim=1)
        else:
            q_local = query

        attn_metadata = self._slice_attn_metadata_for_local_query(
            attn_metadata,
            q_local_len=q_local.shape[1],
            img_seq_local=query.shape[1],
            img_seq_full=k_img_full.shape[1],
            joint_len=joint_len,
            joint_strategy=joint_strategy,
            query_global_offset=query_global_offset,
        )

        return q_local, k_full, v_full, attn_metadata, _AllGatherKVCtx(name=self.name)

    def _slice_attn_metadata_for_local_query(
        self,
        attn_metadata: AttentionMetadata | None,
        *,
        q_local_len: int,
        img_seq_local: int,
        img_seq_full: int,
        joint_len: int,
        joint_strategy: str,
        query_global_offset: int,
    ) -> AttentionMetadata | None:
        """Slice global query metadata for the local query shard."""
        if attn_metadata is None:
            return None

        img_start = self._sp_rank * img_seq_local
        img_end = img_start + img_seq_local
        if img_end > img_seq_full:
            raise ValueError(
                "AllGather-KV SP local image query range exceeds gathered image length: "
                f"rank={self._sp_rank}, img_start={img_start}, img_end={img_end}, img_seq_full={img_seq_full}."
            )

        if joint_strategy == "front":
            ranges = (
                *((QueryRange(0, joint_len, query_global_offset),) if joint_len else ()),
                QueryRange(joint_len, q_local_len, query_global_offset + joint_len + img_start),
            )
        else:
            ranges = (
                QueryRange(0, img_seq_local, query_global_offset + img_start),
                *((QueryRange(img_seq_local, q_local_len, query_global_offset + img_seq_full),) if joint_len else ()),
            )

        if attn_metadata.attn_mask is None:
            return replace(attn_metadata, query_ranges=ranges)

        mask = attn_metadata.attn_mask
        if mask.ndim != 4:
            return replace(attn_metadata, query_ranges=ranges)

        if mask.shape[-2] == q_local_len:
            local_mask = mask
        else:
            q_full_len = joint_len + img_seq_full
            if mask.shape[-2] != q_full_len:
                raise ValueError(
                    "AllGather-KV SP received an attention mask with incompatible Q length: "
                    f"mask_q={mask.shape[-2]}, expected local_q={q_local_len} or full_q={q_full_len} "
                    f"(joint_len={joint_len}, img_seq_local={img_seq_local}, img_seq_full={img_seq_full})."
                )

            parts = [
                mask[..., local_start : local_start + (r.local_end - r.local_start), :]
                for r in ranges
                for local_start in (r.global_start - query_global_offset,)
            ]
            local_mask = torch.cat(parts, dim=-2)

        if local_mask.shape[-2] != q_local_len:
            raise ValueError(
                "AllGather-KV SP produced an attention mask with incompatible local Q length: "
                f"mask_q={local_mask.shape[-2]}, q_local={q_local_len}."
            )
        return replace(attn_metadata, attn_mask=local_mask.contiguous(), query_ranges=ranges)

    def post_attention(
        self,
        attn_output: torch.Tensor,
        ctx: ParallelAttentionContext | None,
    ) -> torch.Tensor:
        return attn_output
