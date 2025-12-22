# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Jiarui Fang.
# Adapted from https://github.com/feifeibear/long-context-attention

# adapted from https://github.com/huggingface/picotron/blob/main/picotron/context_parallel/context_parallel.py
# Copyright 2024 The HuggingFace Inc. team and Jiarui Fang.

import math
import torch
import torch.nn.functional as F
from typing import Any, Optional, Tuple
from vllm_omni.diffusion.distributed.ring import RingComm
from vllm_omni.diffusion.attention.utils import update_out_and_lse
from vllm_omni.diffusion.attention.backends.ring_kernels import pytorch_attn_forward, pytorch_attn_backward

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
    op_type="efficient",
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_strategy="front",
):
    return RingAttentionFunc.apply(group, q, k, v, softmax_scale, causal, op_type, joint_tensor_key, joint_tensor_value, joint_strategy)

class RingAttentionFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, group, q, k, v, sm_scale, is_causal, op_type, joint_tensor_key=None, joint_tensor_value=None, joint_strategy="front"):

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
                step_k = k
                step_v = v
                if step == 0 and joint_tensor_key is not None:
                    if joint_strategy == "front":
                        step_k = torch.cat([joint_tensor_key, step_k], dim=1)
                        step_v = torch.cat([joint_tensor_value, step_v], dim=1)
                    else:
                        step_k = torch.cat([step_k, joint_tensor_key], dim=1)
                        step_v = torch.cat([step_v, joint_tensor_value], dim=1)

                block_out, block_lse  = pytorch_attn_forward(
                    q, step_k, step_v, softmax_scale = sm_scale, causal = is_causal and step == 0, op_type=op_type
                )
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
                
            if step + 1 != comm.world_size:
                comm.wait()
                k = next_k
                v = next_v

        out = out.to(q.dtype)

        ctx.save_for_backward(q, k_og, v_og, out, lse.squeeze(-1), joint_tensor_key, joint_tensor_value)
        ctx.sm_scale = sm_scale
        ctx.is_causal = is_causal
        ctx.group = group
        ctx.op_type = op_type
        ctx.joint_strategy = joint_strategy

        return out

    @staticmethod
    def backward(ctx, dout, *args):


        q, k, v, out, softmax_lse, joint_tensor_key, joint_tensor_value = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        is_causal = ctx.is_causal
        op_type = ctx.op_type
        joint_strategy = ctx.joint_strategy

        kv_comm = RingComm(ctx.group)
        d_kv_comm = RingComm(ctx.group)

        dq, dk, dv = None, None, None
        next_dk, next_dv = None, None
        d_joint_k, d_joint_v = None, None
        
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
                
                step_k = k
                step_v = v
                if step == 0 and joint_tensor_key is not None:
                    if joint_strategy == "front":
                        step_k = torch.cat([joint_tensor_key, step_k], dim=1)
                        step_v = torch.cat([joint_tensor_value, step_v], dim=1)
                    else:
                        step_k = torch.cat([step_k, joint_tensor_key], dim=1)
                        step_v = torch.cat([step_v, joint_tensor_value], dim=1)
                    
                    # Resize buffers for step 0
                    curr_block_dk_buffer = torch.empty(step_k.shape, dtype=k.dtype, device=k.device)
                    curr_block_dv_buffer = torch.empty(step_v.shape, dtype=v.dtype, device=v.device)
                else:
                    curr_block_dk_buffer = block_dk_buffer
                    curr_block_dv_buffer = block_dv_buffer

                block_dq_buffer, curr_block_dk_buffer, curr_block_dv_buffer = pytorch_attn_backward(
                    dout, q, step_k, step_v, out, softmax_lse = softmax_lse, softmax_scale = sm_scale, causal = bwd_causal, op_type=op_type
                )
                
                # Extract parts for joint
                if step == 0 and joint_tensor_key is not None:
                    joint_len = joint_tensor_key.shape[1]
                    if joint_strategy == "front":
                         d_joint_k = curr_block_dk_buffer[:, :joint_len].to(torch.float32)
                         d_joint_v = curr_block_dv_buffer[:, :joint_len].to(torch.float32)
                         block_dk_part = curr_block_dk_buffer[:, joint_len:].to(torch.float32)
                         block_dv_part = curr_block_dv_buffer[:, joint_len:].to(torch.float32)
                    else:
                         block_dk_part = curr_block_dk_buffer[:, :-joint_len].to(torch.float32)
                         block_dv_part = curr_block_dv_buffer[:, :-joint_len].to(torch.float32)
                         d_joint_k = curr_block_dk_buffer[:, -joint_len:].to(torch.float32)
                         d_joint_v = curr_block_dv_buffer[:, -joint_len:].to(torch.float32)
                else:
                     block_dk_part = curr_block_dk_buffer
                     block_dv_part = curr_block_dv_buffer


                if dq is None:
                    dq = block_dq_buffer.to(torch.float32)
                    dk = block_dk_part.to(torch.float32)
                    dv = block_dv_part.to(torch.float32)
                else:
                    dq += block_dq_buffer
                    d_kv_comm.wait()
                    dk = block_dk_part + next_dk
                    dv = block_dv_part + next_dv
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

        return dq, next_dk, next_dv, None, None, None, d_joint_k, d_joint_v, None
