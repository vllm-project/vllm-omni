# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) Microsoft Corporation and Jiarui Fang
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team & Jiarui Fang
# Adapted from
# https://github.com/feifeibear/long-context-attention/blob/main/yunchang/attention/layer.py

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.selector import get_attn_backend
from vllm_omni.diffusion.data import get_current_omni_diffusion_config
from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D
from vllm_omni.diffusion.distributed.parallel_state import get_sequence_parallel_world_size, get_sp_group


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        # ulysses attention
        scatter_idx: int = 2,
        gather_idx: int = 1,
        use_sync: bool = False,
    ):
        super().__init__()
        self.attn_backend = get_attn_backend(-1)
        self.attn_impl_cls = self.attn_backend.get_impl_cls()
        self.attention = self.attn_impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
        )

        self.softmax_scale = softmax_scale
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.use_sync = use_sync
        self.ring_pg: dist.ProcessGroup | None = None
        self.ulysses_pg: dist.ProcessGroup | None = None
        self.use_ulysses = False

        try:
            config = get_current_omni_diffusion_config()
            if config.parallel_config.ulysses_degree > 1:
                self.use_ulysses = True
                # Get sequence parallel process group
                try:
                    sp_group = get_sp_group()
                    self.ring_pg = sp_group.ring_group
                    self.ulysses_pg = sp_group.ulysses_group
                    assert get_sequence_parallel_world_size() > 1, "Sequence parallel world size must be > 1"
                except (AssertionError, RuntimeError):
                    # If sequence parallel group is not initialized, disable Ulysses
                    self.use_ulysses = False
        except Exception:
            self.use_ulysses = False

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        if self.use_ulysses:
            return self._forward_ulysses(query, key, value, attn_metadata)
        else:
            # shape: (batch_size, seq_len, num_heads, head_size)
            attn_output = self.attention.forward(query, key, value, attn_metadata)
            return attn_output

    def _forward_ulysses(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> Tensor:
        """Ulysses attention forward pass with sequence parallelism."""
        # scatter 2, gather 1
        # (bs, seq_len/N, head_cnt, head_size) -> (bs, seq_len, head_cnt/N, head_size)
        query = SeqAllToAll4D.apply(self.ulysses_pg, query, self.scatter_idx, self.gather_idx, self.use_sync)
        key = SeqAllToAll4D.apply(self.ulysses_pg, key, self.scatter_idx, self.gather_idx, self.use_sync)
        value = SeqAllToAll4D.apply(self.ulysses_pg, value, self.scatter_idx, self.gather_idx, self.use_sync)

        softmax_scale = self.softmax_scale
        if softmax_scale is None:
            softmax_scale = query.shape[-1] ** -0.5

        joint_tensor_query, joint_tensor_key, joint_tensor_value = (
            attn_metadata.joint_query,
            attn_metadata.joint_key,
            attn_metadata.joint_value,
        )
        joint_strategy = attn_metadata.joint_strategy

        if joint_tensor_query is not None and joint_tensor_key is not None and joint_tensor_value is not None:
            supported_joint_strategy = ["front", "rear"]
            if joint_strategy not in supported_joint_strategy:
                raise ValueError(
                    f"joint_strategy: {joint_strategy} not supported. supported joint strategy: {supported_joint_strategy}"
                )
            elif joint_strategy == "rear":
                query = torch.cat([query, joint_tensor_query], dim=1)
                key = torch.cat([key, joint_tensor_key], dim=1)
                value = torch.cat([value, joint_tensor_value], dim=1)
            else:
                query = torch.cat([joint_tensor_query, query], dim=1)
                key = torch.cat([joint_tensor_key, key], dim=1)
                value = torch.cat([joint_tensor_value, value], dim=1)
        elif joint_tensor_query is None and joint_tensor_key is None and joint_tensor_value is None:
            pass
        else:
            raise ValueError(
                "joint_tensor_query, joint_tensor_key, and joint_tensor_value should be None or not None simultaneously."
            )

        # TODO: joint key and value (part of attn heads) according to the current rank are needed for ring attention

        context_layer = self.attention.forward(
            query,
            key,
            value,
            attn_metadata=attn_metadata,
        )

        if isinstance(context_layer, tuple):
            context_layer = context_layer[0]

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = SeqAllToAll4D.apply(self.ulysses_pg, context_layer, self.gather_idx, self.scatter_idx, self.use_sync)

        return output
