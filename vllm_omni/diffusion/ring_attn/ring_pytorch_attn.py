# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Jiarui Fang.
# Adapted from https://github.com/feifeibear/long-context-attention

# adapted from https://github.com/huggingface/picotron/blob/main/picotron/context_parallel/context_parallel.py
# Copyright 2024 The HuggingFace Inc. team and Jiarui Fang.

import math
import torch
import torch.nn.functional as F
from typing import Any, Optional, Tuple
from ..ring_attn.kernels import select_flash_attn_impl, AttnType
from .utils import RingComm, update_out_and_lse
from ..ring_attn.kernels.attention import pytorch_attn_forward, pytorch_attn_backward

def ring_pytorch_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    attn_type: AttnType = AttnType.FA,
    attn_processor=None,
):
    return RingAttentionFunc.apply(group, q, k, v, softmax_scale, causal)

class RingAttentionFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, group, q, k, v, sm_scale, is_causal):

        comm = RingComm(group)
        #TODO(fmom): add flex attention
        #TODO(fmom): add flash attention
        #TODO(fmom): Find a better to save these tensors without cloning
        k_og = k.clone()
        v_og = v.clone()
        out, lse = None, None
        next_k, next_v = None, None

        if sm_scale is None:
            sm_scale = q.shape[-1] ** -0.5

        for step in range(comm.world_size):
            if step + 1 != comm.world_size:
                next_k = comm.send_recv(k)
                next_v = comm.send_recv(v)
                comm.commit()

            if not is_causal or step <= comm.rank:
                block_out, block_lse  = pytorch_attn_forward(
                    q, k, v, softmax_scale = sm_scale, causal = is_causal and step == 0
                )
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
                
            if step + 1 != comm.world_size:
                comm.wait()
                k = next_k
                v = next_v

        out = out.to(q.dtype)

        ctx.save_for_backward(q, k_og, v_og, out, lse.squeeze(-1))
        ctx.sm_scale = sm_scale
        ctx.is_causal = is_causal
        ctx.group = group

        return out

    @staticmethod
    def backward(ctx, dout, *args):


        q, k, v, out, softmax_lse = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        is_causal = ctx.is_causal

        kv_comm = RingComm(ctx.group)
        d_kv_comm = RingComm(ctx.group)

        dq, dk, dv = None, None, None
        next_dk, next_dv = None, None
        
        block_dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        block_dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
        block_dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

        next_dk, next_dv = None, None
        next_k, next_v = None, None

        for step in range(kv_comm.world_size):
            if step + 1 != kv_comm.world_size:
                next_k = kv_comm.send_recv(k)
                next_v = kv_comm.send_recv(v)
                kv_comm.commit()

            if step <= kv_comm.rank or not is_causal:
                bwd_causal = is_causal and step == 0

                block_dq_buffer, block_dk_buffer, block_dv_buffer = pytorch_attn_backward(
                    dout, q, k, v, out, softmax_lse = softmax_lse, softmax_scale = sm_scale, causal = bwd_causal
                )

                if dq is None:
                    dq = block_dq_buffer.to(torch.float32)
                    dk = block_dk_buffer.to(torch.float32)
                    dv = block_dv_buffer.to(torch.float32)
                else:
                    dq += block_dq_buffer
                    d_kv_comm.wait()
                    dk = block_dk_buffer + next_dk
                    dv = block_dv_buffer + next_dv
            elif step != 0:
                d_kv_comm.wait()
                dk = next_dk
                dv = next_dv

            if step + 1 != kv_comm.world_size:
                kv_comm.wait()
                k = next_k
                v = next_v

            next_dk = d_kv_comm.send_recv(dk)
            next_dv = d_kv_comm.send_recv(dv)
            d_kv_comm.commit()

        d_kv_comm.wait()

        return dq, next_dk, next_dv, None, None
