# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

# import torch.distributed as dist # Not used directly here, but good practice if needed
from vllm_omni.diffusion.attention.backends.ring.ring_globals import HAS_FLASH_ATTN
from vllm_omni.diffusion.attention.backends.ring.ring_selector import AttnType
from vllm_omni.diffusion.attention.parallel.base import (
    ParallelAttentionContext,
    # ParallelAttentionStrategy, # Not used in type hint below currently
)
from vllm_omni.diffusion.distributed.group_coordinator import SequenceParallelGroupCoordinator

# from vllm_omni.diffusion.attention.backends.ring_selector import AttnType # Already imported above
from vllm_omni.diffusion.forward_context import get_forward_context

if TYPE_CHECKING:
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata


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
        attn_backend_pref: str | None = None,
    ) -> None:
        self._sp_group = sp_group
        self.attn_backend_pref = attn_backend_pref

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
            supported_joint_strategy = ["front", "rear"]
            if joint_strategy not in supported_joint_strategy:
                raise ValueError(f"joint_strategy: {joint_strategy} not supported.")

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

    def run_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
        softmax_scale: float | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        """Run the actual Ring Attention kernel."""
        if softmax_scale is None:
            softmax_scale = query.shape[-1] ** -0.5

        backend_pref = self.attn_backend_pref
        if backend_pref is None:
            try:
                config = get_forward_context().omni_diffusion_config
                # config might not have attention_backend attribute if not updated
                backend_pref = getattr(config, "attention_backend", None)
            except Exception:
                backend_pref = None

        # Fallback for FP32 or if Flash Attention is not available
        if query.dtype == torch.float32 or not HAS_FLASH_ATTN:
            if not HAS_FLASH_ATTN and backend_pref != "sdpa":
                logger = init_logger(__name__)
                logger.warning_once("Flash Attention is not available! Force enabling SDPA.")
            backend_pref = "sdpa"

        # Extract joint tensors
        joint_key, joint_value = None, None
        joint_strategy = "front"
        if attn_metadata is not None:
            joint_key = attn_metadata.joint_key
            joint_value = attn_metadata.joint_value
            if attn_metadata.joint_strategy is not None:
                joint_strategy = attn_metadata.joint_strategy

        if backend_pref == "sdpa" or backend_pref == "torch":
            from vllm_omni.diffusion.attention.backends.ring_pytorch_attn import ring_pytorch_attn_func

            return ring_pytorch_attn_func(
                query,
                key,
                value,
                softmax_scale=softmax_scale,
                causal=causal,
                group=self._sp_group.ring_group,
                op_type="efficient",
                joint_tensor_key=joint_key,
                joint_tensor_value=joint_value,
                joint_strategy=joint_strategy,
            )

        from vllm_omni.diffusion.attention.backends.ring_flash_attn import ring_flash_attn_func

        return ring_flash_attn_func(
            query,
            key,
            value,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            deterministic=False,
            # return_attn_probs=False, # Removed as it might not be supported in signature
            group=self._sp_group.ring_group,
            attn_type=AttnType.FA,
            joint_tensor_key=joint_key,
            joint_tensor_value=joint_value,
            joint_strategy=joint_strategy,
        )
