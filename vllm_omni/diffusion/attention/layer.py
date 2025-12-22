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
import torch.nn as nn

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.parallel import build_parallel_attention_strategy
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.parallel import build_parallel_attention_strategy
from vllm_omni.diffusion.attention.selector import get_attn_backend
from vllm_omni.diffusion.data import get_current_omni_diffusion_config
from vllm_omni.diffusion.distributed.comm import SeqAllToAll4D
from vllm_omni.diffusion.distributed.parallel_state import (
    get_sequence_parallel_world_size, 
    get_sp_group, 
    get_ulysses_parallel_world_size,
    get_ulysses_parallel_rank
)
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
        
        # If backend is flash_attn but dtype is float32, force sdpa
        if backend_pref == "flash_attn" and query.dtype == torch.float32:
            backend_pref = "sdpa" # Force backend_pref to sdpa for fallback

        # Extract joint tensors and modify query if needed
        joint_key, joint_value = None, None
        joint_strategy = "front"
        if attn_metadata is not None:
            if attn_metadata.joint_query is not None:
                if attn_metadata.joint_strategy == "front":
                    query = torch.cat([attn_metadata.joint_query, query], dim=1)
                else:
                    query = torch.cat([query, attn_metadata.joint_query], dim=1)
            
            joint_key = attn_metadata.joint_key
            joint_value = attn_metadata.joint_value
            joint_strategy = attn_metadata.joint_strategy

        # Use ring_pytorch_attn_func for SDPA/Torch backend
        if backend_pref == "sdpa" or backend_pref == "torch":
             from vllm_omni.diffusion.attention.ring_pytorch_attn import ring_pytorch_attn_func
             return ring_pytorch_attn_func(
                 query, key, value, 
                 softmax_scale=softmax_scale, 
                 causal=self.causal,
                 group=self.ring_pg,
                 op_type="flash", # Default to flash implementation of SDPA
                 joint_tensor_key=joint_key,
                 joint_tensor_value=joint_value,
                 joint_strategy=joint_strategy,
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
            joint_tensor_key=joint_key,
            joint_tensor_value=joint_value,
            joint_strategy=joint_strategy,
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
        
        # Handle Joint inputs for Ulysses
        q_joint, k_joint, v_joint = None, None, None
        joint_len = 0
        joint_strategy = "front"
        
        if attn_metadata is not None:
             joint_strategy = attn_metadata.joint_strategy
             if attn_metadata.joint_query is not None:
                 # Slice heads for Ulysses rank
                 ulysses_world_size = get_ulysses_parallel_world_size()
                 ulysses_rank = get_ulysses_parallel_rank()
                 
                 # joint_query is (B, S, H, D). Split H (dim 2).
                 # chunk creates views, which is fine
                 q_joint = attn_metadata.joint_query.chunk(ulysses_world_size, dim=2)[ulysses_rank]
                 k_joint = attn_metadata.joint_key.chunk(ulysses_world_size, dim=2)[ulysses_rank]
                 v_joint = attn_metadata.joint_value.chunk(ulysses_world_size, dim=2)[ulysses_rank]
                 
                 joint_len = q_joint.shape[1]

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
            
            # If backend is flash_attn but dtype is float32, force sdpa
            if backend_pref == "flash_attn" and query.dtype == torch.float32:
                backend_pref = "sdpa" # Force backend_pref to sdpa for fallback

            # Concatenate joint query to local query
            if q_joint is not None:
                if joint_strategy == "front":
                    q = torch.cat([q_joint, q], dim=1)
                else:
                    q = torch.cat([q, q_joint], dim=1)

            # Use ring_pytorch_attn_func for SDPA/Torch backend
            if backend_pref == "sdpa" or backend_pref == "torch":
                 from vllm_omni.diffusion.attention.ring_pytorch_attn import ring_pytorch_attn_func
                 context_layer = ring_pytorch_attn_func(
                     q, k, v, 
                     softmax_scale=softmax_scale, 
                     causal=self.causal,
                     group=self.ring_pg,
                     op_type="flash",
                     joint_tensor_key=k_joint,
                     joint_tensor_value=v_joint,
                     joint_strategy=joint_strategy,
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
                    joint_tensor_key=k_joint,
                    joint_tensor_value=v_joint,
                    joint_strategy=joint_strategy,
                )

        elif is_npu():
            # NPU implementation might not support joint tensors natively yet?
            # If q_joint is not None, we should concatenate them
            if q_joint is not None:
                if joint_strategy == "front":
                    q = torch.cat([q_joint, q], dim=1)
                    k = torch.cat([k_joint, k], dim=1)
                    v = torch.cat([v_joint, v], dim=1)
                else:
                    q = torch.cat([q, q_joint], dim=1)
                    k = torch.cat([k, k_joint], dim=1)
                    v = torch.cat([v, v_joint], dim=1)
            
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
            # Standard Ulysses logic for attention computation
            # If joint tensors exist, we must concatenate them to Q, K, V
            if q_joint is not None:
                if joint_strategy == "front":
                    q = torch.cat([q_joint, q], dim=1)
                    k = torch.cat([k_joint, k], dim=1)
                    v = torch.cat([v_joint, v], dim=1)
                else:
                    q = torch.cat([q, q_joint], dim=1)
                    k = torch.cat([k, k_joint], dim=1)
                    v = torch.cat([v, v_joint], dim=1)
            
            context_layer = self.attention.forward(
                q,
                k,
                v,
                attn_metadata=None, # Already integrated into q, k, v
            )

        if isinstance(context_layer, tuple):
            context_layer = context_layer[0]

        # Post-processing: Split Joint and Img outputs
        if joint_len > 0:
            if joint_strategy == "front":
                output_joint = context_layer[:, :joint_len]
                output_img = context_layer[:, joint_len:]
            else:
                output_img = context_layer[:, :-joint_len]
                output_joint = context_layer[:, -joint_len:]
            
            # Scatter 1, Gather 2 for Img part
            output_img = SeqAllToAll4D.apply(self.ulysses_pg, output_img, self.gather_idx, self.scatter_idx, self.use_sync)
            
            # AllGather for Joint part (Head Gather)
            # output_joint is (B, JointLen, H_local, D). We want (B, JointLen, H_total, D).
            # dist.all_gather expects list of tensors.
            gathered_joint = [torch.zeros_like(output_joint) for _ in range(get_ulysses_parallel_world_size())]
            dist.all_gather(gathered_joint, output_joint, group=self.ulysses_pg)
            
            # Concatenate along Head dimension (dim 2)
            # Note: all_gather returns tensors in rank order. 
            # Ulysses distributes heads 0..H/N to rank 0, etc.
            # So concatenation order corresponds to head order.
            output_joint = torch.cat(gathered_joint, dim=2)
            
            if joint_strategy == "front":
                output = torch.cat([output_joint, output_img], dim=1)
            else:
                output = torch.cat([output_img, output_joint], dim=1)
        else:
            # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
            # scatter 1, gather 2
            output = SeqAllToAll4D.apply(self.ulysses_pg, context_layer, self.gather_idx, self.scatter_idx, self.use_sync)

        return output
