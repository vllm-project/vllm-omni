# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm_omni.diffusion.attention.parallel.allgather_kv import (
    AllGatherKVParallelAttention,
)
from vllm_omni.diffusion.attention.parallel.base import ParallelAttentionContext
from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
from vllm_omni.diffusion.distributed.group_coordinator import (
    SequenceParallelGroupCoordinator,
)
from vllm_omni.diffusion.forward_context import (
    get_forward_context,
    get_ulysses_mode,
    is_forward_context_available,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata


class UlyssesAllGatherKVParallelAttention(AllGatherKVParallelAttention):
    """Composed 2D sequence parallelism: Ulysses (heads) x AllGather-KV (sequence).

    With ``U = ulysses_degree`` and ``A = allgather_degree`` (``ring_degree``
    must be 1), each rank starts from the flat contiguous SP shard and runs::

        Ulysses all-to-all (U group):  Q, K, V -> [B, S / A,       H / U, D]
        K/V AllGather     (A group):   K, V    -> [B, S,           H / U, D]
        local-Q / global-KV attention: O       -> [B, S / A,       H / U, D]
        reverse Ulysses all-to-all:    O       -> [B, S / (U x A), H,     D]

    The kernel sees the same local-Q/global-KV problem as plain AllGather-KV
    (stable global K/V indexing, unlike Ring). The collective order and the
    rank layout it relies on are load-bearing; see
    ``build_ulysses_allgather_rank_groups`` for the invariant.

    Both halves are reused rather than reimplemented: Ulysses runs with
    ``defer_joint=True`` (joint q/k/v stay head-sliced in the metadata),
    then the inherited AllGather-KV ``pre_attention`` gathers the image K/V and
    re-attaches the joint tensors -- attaching them before the gather would
    replicate them once per AllGather rank.

    Fails closed on: ``ring_degree > 1``, causal attention, a non-2D
    ``joint_attn_mask``, a 2D ``joint_attn_mask`` combined with a 4D
    ``attn_mask``, 2D key masks that do not cover the gathered global image
    keys, and uneven per-rank region lengths. All 2D key-mask merging (image
    and joint) is owned by ``_merge_joint_attn_mask`` against the final,
    post-gather key layout; 4D masks keep the inherited AllGather query-range
    slicing.
    """

    def __init__(
        self,
        sp_group: SequenceParallelGroupCoordinator,
        scatter_idx: int,
        gather_idx: int,
        use_sync: bool,
        ulysses_a2a_permute: bool = False,
    ) -> None:
        super().__init__(sp_group)
        self._ulysses = UlyssesParallelAttention(
            sp_group,
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            use_sync=use_sync,
            ulysses_a2a_permute=ulysses_a2a_permute,
        )

    @property
    def name(self) -> str:
        return "ulysses_allgather_kv"

    def _assert_equal_region_lengths(self, region_len: int, device: torch.device) -> None:
        """Fail fast when the ``A`` regions do not have equal length.

        ``all_gather_into_tensor`` derives its output shape from the *local*
        input, so uneven regions would hang or corrupt instead of raising. In
        strict mode the Ulysses all-to-all already guarantees equal region
        lengths (seq must be evenly shardable), so no collective is spent;
        under advanced_uaa rank-local lengths may legitimately differ, so the
        check runs on every forward there -- unless every active SP boundary
        is framework-managed auto_pad, whose contract already guarantees equal
        local lengths.

        The cheap early returns live outside the ``torch.compiler.disable``d
        checker so strict-mode compiled forwards do not unconditionally split
        the graph to call a no-op.
        """
        if get_ulysses_mode(default="strict") == "strict" or self._sp_size <= 1:
            return
        if is_forward_context_available() and get_forward_context().sp_rank_local_seq_lens_equal:
            # auto_pad made every rank's shard equally long at each SP
            # boundary, so equality is known without a per-forward collective
            # or a host sync (mirrors the advanced_uaa fast path in Ulysses).
            return
        self._check_equal_region_lengths_collective(region_len, device)

    @torch.compiler.disable
    def _check_equal_region_lengths_collective(self, region_len: int, device: torch.device) -> None:
        local = torch.tensor([int(region_len)], dtype=torch.int64, device=device)
        gathered = [torch.empty_like(local) for _ in range(self._sp_size)]
        dist.all_gather(gathered, local, group=self._allgather_group)
        lengths = [int(t.item()) for t in gathered]
        if len(set(lengths)) != 1:
            raise ValueError(
                "Ulysses x AllGather-KV requires every AllGather rank to hold an equally long "
                "region after the Ulysses all-to-all, but got region lengths "
                f"{lengths} across allgather ranks. This means the shared sequence was not evenly "
                "shardable across the SP group. Choose a sequence length divisible by "
                "ulysses_degree * allgather_degree, or enable auto_pad in the model's _sp_plan."
            )

    @staticmethod
    def _merge_joint_attn_mask(attn_metadata: AttentionMetadata | None, key: torch.Tensor):
        """Own all 2D key-mask handling against the final, post-gather keys.

        ``defer_joint=True`` skips the Ulysses mask merge and the AllGather
        path passes 2D masks through untouched. A 2D mask marks invalid keys,
        so merging is concatenation: True-fill the missing side -- the joint
        side is sized from ``joint_key`` because ``joint_query`` can be empty
        during cached-KV reuse -- then concatenate in ``joint_strategy``
        order. A provided image mask must cover the gathered global image
        keys, which every rank sees identically.

        4D masks are not key-validity masks: they are left to the inherited
        AllGather query-range slicing, and combined with a 2D joint mask they
        fail closed instead of being merged.
        """
        if attn_metadata is None:
            return attn_metadata
        joint_mask = attn_metadata.joint_attn_mask
        img_mask = attn_metadata.attn_mask
        if joint_mask is None and img_mask is None:
            return attn_metadata

        # Rank check first: the merge only combines 2D masks (an absent side
        # counts as 2D -- it gets True-filled below).
        joint_rank = joint_mask.ndim if joint_mask is not None else 2
        img_rank = img_mask.ndim if img_mask is not None else 2
        if joint_rank != 2:
            raise NotImplementedError(
                f"Ulysses x AllGather-KV only supports a 2D joint_attn_mask (got ndim={joint_rank})."
            )
        if img_rank == 4:
            if joint_mask is not None:
                raise NotImplementedError(
                    "Ulysses x AllGather-KV cannot merge a 2D joint_attn_mask with a 4D "
                    "attn_mask: move the padding into a single mask, or set allgather_degree=1."
                )
            return attn_metadata
        if img_rank != 2:
            raise NotImplementedError(f"Ulysses x AllGather-KV only supports 2D or 4D attn_mask (got ndim={img_rank}).")

        joint_k = attn_metadata.joint_key
        if joint_mask is not None and joint_k is None:
            raise ValueError(
                "Ulysses x AllGather-KV got a joint_attn_mask without joint K/V: "
                "the mask describes joint keys that would never be gathered."
            )

        batch = key.shape[0]
        joint_len = joint_k.shape[1] if joint_k is not None else 0
        img_len = key.shape[1] - joint_len
        if img_mask is not None and tuple(img_mask.shape) != (batch, img_len):
            raise ValueError(
                "Ulysses x AllGather-KV expects a 2D attn_mask covering the gathered "
                f"global image keys [batch={batch}, keys={img_len}], got {tuple(img_mask.shape)}."
            )
        if joint_mask is not None and tuple(joint_mask.shape) != (batch, joint_len):
            raise ValueError(
                "Ulysses x AllGather-KV got a joint_attn_mask inconsistent with the joint "
                f"keys: expected [batch={batch}, keys={joint_len}], got {tuple(joint_mask.shape)}."
            )

        if joint_len == 0:
            # No joint keys: the image mask already covers every key.
            assert img_mask is not None
            attn_metadata.attn_mask = img_mask.bool().contiguous()
            return attn_metadata

        if joint_mask is None:
            joint_mask = torch.ones(batch, joint_len, dtype=torch.bool, device=key.device)
        if img_mask is None:
            img_mask = torch.ones(batch, img_len, dtype=torch.bool, device=key.device)
        joint_strategy = attn_metadata.joint_strategy or "front"
        merged = (
            torch.cat([joint_mask, img_mask], dim=1)
            if joint_strategy == "front"
            else torch.cat([img_mask, joint_mask], dim=1)
        )
        attn_metadata.attn_mask = merged.bool().contiguous()
        return attn_metadata

    def pre_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ):
        # 1. Ulysses: reshard the image shard to [B, S/A, H/U, D] and record the
        #    head-sliced joint tensors for later, without concatenating them.
        query, key, value, attn_metadata, ctx = self._ulysses.pre_attention(
            query,
            key,
            value,
            attn_metadata,
            defer_joint=True,
        )

        # 2. Gather the image K/V over the orthogonal group, then let the
        #    inherited AllGather-KV path slice the attention metadata, prepend
        #    the joint tensors, and build the local-Q/global-KV view.
        self._assert_equal_region_lengths(key.shape[1], key.device)
        query, key, value, attn_metadata, _ = super().pre_attention(query, key, value, attn_metadata)

        # 3. Own all 2D key-mask handling against the final key layout: merge
        #    joint/image masks (filling missing sides), or leave 4D masks to
        #    the inherited query-range slicing.
        attn_metadata = self._merge_joint_attn_mask(attn_metadata, key)

        # 4. The reverse transform is entirely Ulysses': it splits the joint
        #    part back out, undoes the image all-to-all, and head-gathers the
        #    joint output over the Ulysses group.
        return query, key, value, attn_metadata, ctx

    def post_attention(
        self,
        attn_output: torch.Tensor,
        ctx: ParallelAttentionContext | None,
    ) -> torch.Tensor:
        return self._ulysses.post_attention(attn_output, ctx)
