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
from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
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
        
        self.use_ring = False
        self.ring_pg = None

        try:
            config = get_current_omni_diffusion_config()
            if config.parallel_config.ring_degree > 1:
                self.use_ring = True
                try:
                    self.ring_pg = get_sp_group().ring_group
                except:
                    self.use_ring = False
        except Exception:
            self.use_ring = False

        self.parallel_strategy = build_parallel_attention_strategy(
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            use_sync=use_sync,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata = None,
    ) -> torch.Tensor:
        
        # 1. Prepare inputs (Communication / Resharding)
        # For Ulysses: AllToAll Q/K/V; Slicing joint_q/k/v
        # For Ring: Concat joint_q
        query, key, value, attn_metadata, ctx = self.parallel_strategy.pre_attention(
            query, key, value, attn_metadata
        )

        # 2. Kernel Execution (Computation)
        if self.use_ring:
            out = self._run_ring_attention(query, key, value, attn_metadata)
        else:
            out = self._run_local_attention(query, key, value, attn_metadata)

        # 3. Post-processing (Reverse Communication)
        # For Ulysses: AllToAll Output, and AllGather Joint Output
        out = self.parallel_strategy.post_attention(out, ctx)
        
        return out

    def _run_local_attention(self, query, key, value, attn_metadata):
        # Check backend preference from config
        try:
            config = get_current_omni_diffusion_config()
            backend_pref = config.attention_backend
        except Exception:
            backend_pref = None

        if backend_pref == "flash_attn" and query.dtype == torch.float32:
            backend_pref = "sdpa"

        if is_npu():
             return self.attention(
                query, key, value,
                num_heads=query.shape[-2],
                input_layout="BSND",
                scale=self.softmax_scale,
                softmax_lse_flag=True,
                pre_tokens=65535,
                next_tokens=65535,
            )[0]
        
        # Fallback to standard attention
        return self.attention.forward(query, key, value, attn_metadata)

    def _run_ring_attention(self, query, key, value, attn_metadata):
        softmax_scale = self.softmax_scale
        if softmax_scale is None:
            softmax_scale = query.shape[-1] ** -0.5

        try:
            config = get_current_omni_diffusion_config()
            backend_pref = config.attention_backend
        except Exception:
            backend_pref = None
        
        if backend_pref == "flash_attn" and query.dtype == torch.float32:
            backend_pref = "sdpa"

        # Extract joint tensors
        joint_key, joint_value = None, None
        joint_strategy = "front"
        if attn_metadata is not None:
            joint_key = attn_metadata.joint_key
            joint_value = attn_metadata.joint_value
            joint_strategy = attn_metadata.joint_strategy

        if backend_pref == "sdpa" or backend_pref == "torch":
             from vllm_omni.diffusion.attention.ring_pytorch_attn import ring_pytorch_attn_func
             return ring_pytorch_attn_func(
                 query, key, value, 
                 softmax_scale=softmax_scale, 
                 causal=self.causal,
                 group=self.ring_pg,
                 op_type="flash", 
                 joint_tensor_key=joint_key,
                 joint_tensor_value=joint_value,
                 joint_strategy=joint_strategy,
             )
            
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
            group=self.ring_pg,
            attn_type=AttnType.FA,
            joint_tensor_key=joint_key,
            joint_tensor_value=joint_value,
            joint_strategy=joint_strategy,
        )
