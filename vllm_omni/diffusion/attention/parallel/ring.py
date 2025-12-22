# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.parallel.base import ParallelAttentionContext
from vllm_omni.diffusion.distributed.group_coordinator import SequenceParallelGroupCoordinator


@dataclass(frozen=True, slots=True)
class _RingCtx(ParallelAttentionContext):
    """Per-forward context for Ring sequence-parallel attention."""
    # Ring attention typically doesn't need complex context for post-processing
    # as the output is already correctly sharded along sequence dimension.
    pass


class RingParallelAttention:
    """Ring sequence-parallel strategy.
    
    This strategy prepares inputs for Ring Attention.
    Key responsibilities:
    - Concatenate joint_query (Text) to query (Image) if present.
    - Keep joint_key/value separate in metadata for the Ring kernel to handle as static prefix.
    """

    def __init__(
        self,
        sp_group: SequenceParallelGroupCoordinator,
    ) -> None:
        self._sp_group = sp_group

    @property
    def enabled(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "ring"

    def pre_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ):
        joint_tensor_query = None
        joint_strategy = "front"

        if attn_metadata is not None:
            joint_tensor_query = attn_metadata.joint_query
            joint_strategy = attn_metadata.joint_strategy

        if joint_tensor_query is not None:
            if joint_strategy == "front":
                query = torch.cat([joint_tensor_query, query], dim=1)
            else:
                query = torch.cat([query, joint_tensor_query], dim=1)
            
            # Note: We do NOT concatenate joint_key/value here.
            # They are preserved in attn_metadata and will be passed 
            # explicitly to ring_flash_attn_func.

        ctx = _RingCtx(name=self.name)
        return query, key, value, attn_metadata, ctx

    def post_attention(self, attn_output: torch.Tensor, ctx: ParallelAttentionContext | None) -> torch.Tensor:
        # Ring attention output is already sharded correctly along sequence dimension.
        return attn_output

