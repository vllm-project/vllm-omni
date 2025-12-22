# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) Microsoft Corporation and Jiarui Fang
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team & Jiarui Fang
# Adapted from
# https://github.com/feifeibear/long-context-attention/blob/main/yunchang/attention/layer.py

from typing import Optional
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.parallel import build_parallel_attention_strategy
from vllm_omni.diffusion.attention.selector import get_attn_backend
from vllm_omni.diffusion.data import get_current_omni_diffusion_config
from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D
from vllm_omni.diffusion.distributed.parallel_state import get_sequence_parallel_world_size, get_sp_group
from vllm_omni.utils.platform_utils import is_npu
from vllm_omni.diffusion.attention.ring_flash_attn import ring_flash_attn_func
from vllm_omni.diffusion.attention.backends.ring_selector import AttnType
from vllm_omni.diffusion.attention.backends.ring_globals import HAS_FLASH_ATTN


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
        self.causal = causal
        self.ring_pg: Optional[dist.ProcessGroup] = None
        self.ulysses_pg: Optional[dist.ProcessGroup] = None
        self.use_ulysses = False
        self.use_ring = False

        try:
            config = get_current_omni_diffusion_config()
            if config.parallel_config.ulysses_degree > 1:
                self.use_ulysses = True
            
            if config.parallel_config.ring_degree > 1:
                self.use_ring = True
            
            if self.use_ulysses or self.use_ring:
                # Get sequence parallel process group
                try:
                    sp_group = get_sp_group()
                    self.ring_pg = sp_group.ring_group
                    self.ulysses_pg = sp_group.ulysses_group
                    assert get_sequence_parallel_world_size() > 1, "Sequence parallel world size must be > 1"
                except (AssertionError, RuntimeError):
                    # If sequence parallel group is not initialized, disable Ulysses/Ring
                    self.use_ulysses = False
                    self.use_ring = False
        except Exception:
            self.use_ulysses = False
            self.use_ring = False

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        # Check backend preference from config
        try:
            config = get_current_omni_diffusion_config()
            backend_pref = config.attention_backend
        except Exception:
            backend_pref = None

        if self.use_ulysses:
            return self._forward_ulysses(query, key, value, attn_metadata)
        elif self.use_ring:
            return self._forward_ring(query, key, value, attn_metadata)
        else:
            # Decide whether to use ring_flash_attn_func (Flash Attention)
            use_fa = False
            if backend_pref == "flash_attn":
                use_fa = True
            elif backend_pref == "sdpa":
                use_fa = False
            else:
                # Default: use FA if available and not on NPU
                if HAS_FLASH_ATTN and not is_npu():
                    use_fa = True

            if use_fa and HAS_FLASH_ATTN:
                softmax_scale = self.softmax_scale
                if softmax_scale is None:
                    softmax_scale = query.shape[-1] ** -0.5
                
                return ring_flash_attn_func(
                    query,
                    key,
                    value,
                    dropout_p=0.0,
                    softmax_scale=softmax_scale,
                    causal=self.causal,
                    window_size=(-1, -1),
                    softcap=0.0,
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=False,
                    group=None,
                    attn_type=AttnType.FA,
                )
            
            # Use ring_pytorch_attn_func for SDPA/Torch backend if preferred, to ensure consistent implementation
            # This is optional but can help align behavior with ring attention implementation
            if backend_pref == "sdpa" or backend_pref == "torch":
                 from vllm_omni.diffusion.attention.ring_pytorch_attn import ring_pytorch_attn_func
                 softmax_scale = self.softmax_scale
                 if softmax_scale is None:
                     softmax_scale = query.shape[-1] ** -0.5
                 return ring_pytorch_attn_func(
                     query, key, value, 
                     softmax_scale=softmax_scale, 
                     causal=self.causal,
                     group=None,
                     op_type="flash" # Default to flash implementation of SDPA
                 )

            # shape: (batch_size, seq_len, num_heads, head_size)
            attn_output = self.attention.forward(query, key, value, attn_metadata)
            return attn_output

    def _forward_ring(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> Tensor:
        """Ring attention forward pass with sequence parallelism."""
        
        softmax_scale = self.softmax_scale
        if softmax_scale is None:
            softmax_scale = query.shape[-1] ** -0.5

        # Check backend preference from config
        try:
            config = get_current_omni_diffusion_config()
            backend_pref = config.attention_backend
        except Exception:
            backend_pref = None
        
        # Use ring_pytorch_attn_func for SDPA/Torch backend
        if backend_pref == "sdpa" or backend_pref == "torch":
             from vllm_omni.diffusion.attention.ring_pytorch_attn import ring_pytorch_attn_func
             return ring_pytorch_attn_func(
                 query, key, value, 
                 softmax_scale=softmax_scale, 
                 causal=self.causal,
                 group=self.ring_pg,
                 op_type="efficient" # Default to flash implementation of SDPA
             )
            
        out = ring_flash_attn_func(
            query,
            key,
            value,
            dropout_p=0.0, 
            softmax_scale=softmax_scale,
            causal=self.causal,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            deterministic=False,
            return_attn_probs=False,
            group=self.ring_pg,
            attn_type=AttnType.FA,
        )
        return out

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
        q = SeqAllToAll4D.apply(self.ulysses_pg, query, self.scatter_idx, self.gather_idx, self.use_sync)
        k = SeqAllToAll4D.apply(self.ulysses_pg, key, self.scatter_idx, self.gather_idx, self.use_sync)
        v = SeqAllToAll4D.apply(self.ulysses_pg, value, self.scatter_idx, self.gather_idx, self.use_sync)

        softmax_scale = self.softmax_scale
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5

        if self.use_ring:
            # Check backend preference from config
            try:
                config = get_current_omni_diffusion_config()
                backend_pref = config.attention_backend
            except Exception:
                backend_pref = None
            
            # Use ring_pytorch_attn_func for SDPA/Torch backend
            if backend_pref == "sdpa" or backend_pref == "torch":
                 from vllm_omni.diffusion.attention.ring_pytorch_attn import ring_pytorch_attn_func
                 context_layer = ring_pytorch_attn_func(
                     q, k, v, 
                     softmax_scale=softmax_scale, 
                     causal=self.causal,
                     group=self.ring_pg,
                     op_type="efficient" 
                 )
            else:
                context_layer = ring_flash_attn_func(
                    q,
                    k,
                    v,
                    dropout_p=0.0,
                    softmax_scale=softmax_scale,
                    causal=self.causal,
                    window_size=(-1, -1),
                    softcap=0.0,
                    alibi_slopes=None,
                    deterministic=False,
                    return_attn_probs=False,
                    group=self.ring_pg,
                    attn_type=AttnType.FA,
                )

        elif is_npu():
            context_layer = self.attention(
                q,
                k,
                v,
                num_heads=q.shape[-2],
                input_layout="BSND",
                scale=softmax_scale,
                softmax_lse_flag=True,
                pre_tokens=65535,
                next_tokens=65535,
            )
        else:
            context_layer = self.attention.forward(
                q,
                k,
                v,
                attn_metadata=attn_metadata,
            )

        if isinstance(context_layer, tuple):
            context_layer = context_layer[0]

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = SeqAllToAll4D.apply(self.ulysses_pg, context_layer, self.gather_idx, self.scatter_idx, self.use_sync)

        return output
