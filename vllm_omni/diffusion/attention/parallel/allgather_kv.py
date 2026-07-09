# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from vllm_omni.diffusion.attention.backends.utils.piecewise_attn import (
    build_segments,
)
from vllm_omni.diffusion.attention.parallel.base import ParallelAttentionContext
from vllm_omni.diffusion.distributed.group_coordinator import (
    SequenceParallelGroupCoordinator,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


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
        self._sp_rank = sp_group.ulysses_rank
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

        kv_repeat_num = 1
        if attn_metadata is not None:
            kv_repeat_num = int(attn_metadata.extra.get("kv_repeat_num", 1))

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

        k_full = _repeat_kv(k_full, kv_repeat_num)
        v_full = _repeat_kv(v_full, kv_repeat_num)

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
            joint_len=joint_q.shape[1] if joint_q is not None else 0,
            joint_strategy=joint_strategy,
        )

        ctx = _AllGatherKVCtx(
            name=self.name,
            joint_len=joint_q.shape[1] if joint_q is not None else 0,
            joint_strategy=joint_strategy,
            img_seq_local=query.shape[1],
        )
        return q_local, k_full, v_full, attn_metadata, ctx

    def _slice_full_attn_spans_for_local_query(
        self,
        spans_per_batch: list[list[tuple[int, int]]] | None,
        *,
        img_seq_local: int,
        img_seq_full: int,
        joint_len: int,
        joint_strategy: str,
    ) -> list[list[tuple[int, int]]] | None:
        if not spans_per_batch:
            return spans_per_batch
        if joint_strategy != "front":
            raise ValueError(
                "AllGather-KV SP supports full_attn_spans only with joint_strategy='front'. "
                f"Got joint_strategy={joint_strategy!r}."
            )

        img_start = self._sp_rank * img_seq_local
        img_end = min(img_start + img_seq_local, img_seq_full)
        local_spans_per_batch: list[list[tuple[int, int]]] = []

        for spans in spans_per_batch:
            local_spans: list[tuple[int, int]] = []
            for start, end, mode in build_segments(spans, query_offset=0, query_len=joint_len):
                if mode == "full":
                    local_spans.append((start, end))

            img_query_offset = joint_len + img_start
            for start, end, mode in build_segments(
                spans,
                query_offset=img_query_offset,
                query_len=img_end - img_start,
            ):
                if mode == "full":
                    local_spans.append(
                        (
                            joint_len + start - img_query_offset,
                            joint_len + end - img_query_offset,
                        )
                    )
            local_spans_per_batch.append(local_spans)

        return local_spans_per_batch

    def _slice_attn_metadata_for_local_query(
        self,
        attn_metadata: AttentionMetadata | None,
        *,
        q_local_len: int,
        img_seq_local: int,
        img_seq_full: int,
        joint_len: int,
        joint_strategy: str,
    ) -> AttentionMetadata | None:
        """Convert a full-Q dense mask into the local-Q mask used by AG-KV.

        HunyuanImage3 builds masks in global sequence coordinates. AllGather-KV
        keeps the K/V axis global but computes only this rank's Q rows, so a
        [B, H, Q_full, K_full] mask must be row-sliced to
        [B, H, Q_local, K_full].
        """
        if attn_metadata is None or attn_metadata.attn_mask is None:
            if attn_metadata is None:
                return None
            local_spans = self._slice_full_attn_spans_for_local_query(
                attn_metadata.full_attn_spans,
                img_seq_local=img_seq_local,
                img_seq_full=img_seq_full,
                joint_len=joint_len,
                joint_strategy=joint_strategy,
            )
            return replace(attn_metadata, full_attn_spans=local_spans)

        mask = attn_metadata.attn_mask
        if mask.ndim != 4:
            local_spans = self._slice_full_attn_spans_for_local_query(
                attn_metadata.full_attn_spans,
                img_seq_local=img_seq_local,
                img_seq_full=img_seq_full,
                joint_len=joint_len,
                joint_strategy=joint_strategy,
            )
            return replace(attn_metadata, full_attn_spans=local_spans)

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

            img_start = self._sp_rank * img_seq_local
            img_end = img_start + img_seq_local
            if img_end > img_seq_full:
                raise ValueError(
                    "AllGather-KV SP local image query range exceeds gathered image length: "
                    f"rank={self._sp_rank}, img_start={img_start}, img_end={img_end}, img_seq_full={img_seq_full}."
                )

            if joint_len > 0 and joint_strategy == "front":
                joint_mask = mask[..., :joint_len, :]
                img_mask = mask[..., joint_len + img_start : joint_len + img_end, :]
                local_mask = torch.cat([joint_mask, img_mask], dim=-2)
            elif joint_len > 0:
                img_mask = mask[..., img_start:img_end, :]
                joint_mask = mask[..., img_seq_full : img_seq_full + joint_len, :]
                local_mask = torch.cat([img_mask, joint_mask], dim=-2)
            else:
                local_mask = mask[..., img_start:img_end, :]

        if local_mask.shape[-2] != q_local_len:
            raise ValueError(
                "AllGather-KV SP produced an attention mask with incompatible local Q length: "
                f"mask_q={local_mask.shape[-2]}, q_local={q_local_len}."
            )
        local_spans = self._slice_full_attn_spans_for_local_query(
            attn_metadata.full_attn_spans,
            img_seq_local=img_seq_local,
            img_seq_full=img_seq_full,
            joint_len=joint_len,
            joint_strategy=joint_strategy,
        )
        return replace(attn_metadata, attn_mask=local_mask.contiguous(), full_attn_spans=local_spans)

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
