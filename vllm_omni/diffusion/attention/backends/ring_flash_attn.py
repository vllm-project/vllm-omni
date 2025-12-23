# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Jiarui Fang.
# Adapted from https://github.com/feifeibear/long-context-attention

import torch
import torch.distributed as dist
# from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from vllm_omni.diffusion.distributed.ring import RingComm
from vllm_omni.diffusion.attention.backends.ring_utils import update_out_and_lse
from vllm_omni.diffusion.attention.backends.ring_selector import select_flash_attn_impl, AttnType

def ring_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    attn_type: AttnType = AttnType.FA,
    attn_processor=None,
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_strategy="front",
):
    comm = RingComm(process_group)

    out = None
    lse = None

    next_k, next_v = None, None

    # Check and adjust q, k, v to be contiguous
    if not q.is_contiguous(): q = q.contiguous()
    if not k.is_contiguous(): k = k.contiguous()
    if not v.is_contiguous(): v = v.contiguous()

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k: torch.Tensor = comm.send_recv(k)
            next_v: torch.Tensor = comm.send_recv(v)
            comm.commit()

        if not causal or step <= comm.rank:
            step_k = k
            step_v = v
            if step == 0 and joint_tensor_key is not None:
                if joint_strategy == "front":
                    step_k = torch.cat([joint_tensor_key, step_k], dim=1)
                    step_v = torch.cat([joint_tensor_value, step_v], dim=1)
                else:
                    step_k = torch.cat([step_k, joint_tensor_key], dim=1)
                    step_v = torch.cat([step_v, joint_tensor_value], dim=1)
            
            fn = select_flash_attn_impl(attn_type, stage="fwd-only", attn_processor=attn_processor)
            block_out, block_lse = fn(
                q,
                step_k,
                step_v,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal and step == 0,
                window_size=window_size,
                softcap=softcap,
                alibi_slopes=alibi_slopes,
                return_softmax=True and dropout_p > 0,
            )
            # print(f"Rank {comm.rank} Step {step}: q={q.shape} block_out={block_out.shape} block_lse={block_lse.shape}")
            
            # Ensure block_out is contiguous if needed, though usually it is from FA
            
            if attn_type == AttnType.SPARSE_SAGE:
                out, lse = block_out, block_lse
            else:
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    out = out.to(q.dtype)
    if attn_type != AttnType.SPARSE_SAGE:
        lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


def ring_flash_attn_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    attn_type: AttnType = AttnType.FA,
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_strategy="front",
):
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
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
        if step <= kv_comm.rank or not causal:
            bwd_causal = causal and step == 0
            fn = select_flash_attn_impl(attn_type, stage="bwd-only")
            
            step_k = k
            step_v = v
            if step == 0 and joint_tensor_key is not None:
                if joint_strategy == "front":
                    step_k = torch.cat([joint_tensor_key, step_k], dim=1)
                    step_v = torch.cat([joint_tensor_value, step_v], dim=1)
                else:
                    step_k = torch.cat([step_k, joint_tensor_key], dim=1)
                    step_v = torch.cat([step_v, joint_tensor_value], dim=1)
                
                # Resize buffers for step 0 if joint tensors are used
                # We need larger buffers for dk/dv in this step
                curr_block_dk_buffer = torch.empty(step_k.shape, dtype=k.dtype, device=k.device)
                curr_block_dv_buffer = torch.empty(step_v.shape, dtype=v.dtype, device=v.device)
            else:
                curr_block_dk_buffer = block_dk_buffer
                curr_block_dv_buffer = block_dv_buffer

            fn(
                dout,
                q,
                step_k,
                step_v,
                out,
                softmax_lse,
                block_dq_buffer,
                curr_block_dk_buffer,
                curr_block_dv_buffer,
                dropout_p,
                softmax_scale,
                bwd_causal,
                window_size,
                softcap,
                alibi_slopes,
                deterministic,
                rng_state=None,
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
            
            # Extract gradients for joint tensors and regular tensors if step 0
            if step == 0 and joint_tensor_key is not None:
                # Split curr_block_dk/dv_buffer
                joint_len = joint_tensor_key.shape[1]
                if joint_strategy == "front":
                     d_joint_k = curr_block_dk_buffer[:, :joint_len].to(torch.float32)
                     d_joint_v = curr_block_dv_buffer[:, :joint_len].to(torch.float32)
                     dk_part = curr_block_dk_buffer[:, joint_len:].to(torch.float32)
                     dv_part = curr_block_dv_buffer[:, joint_len:].to(torch.float32)
                else:
                     dk_part = curr_block_dk_buffer[:, :-joint_len].to(torch.float32)
                     dv_part = curr_block_dv_buffer[:, :-joint_len].to(torch.float32)
                     d_joint_k = curr_block_dk_buffer[:, -joint_len:].to(torch.float32)
                     d_joint_v = curr_block_dv_buffer[:, -joint_len:].to(torch.float32)
                
                # If dq was initialized in this step, dk/dv need to be set correctly
                # In the logic above: if dq is None ...
                # Wait, the logic `if dq is None` assumes first step computed corresponds to first accumulation?
                # Actually `if dq is None` happens at first VALID step.
                # If causal=True, step 0 might be skipped? 
                # "if step <= kv_comm.rank or not causal" -> step 0 is always executed if rank >= 0.
                
                if dq is None: # Should not happen if step 0 executed
                    pass 
                else:
                    # If this was the first update to dk/dv
                    if dk is None: # Actually dk is init in `if dq is None` block
                         dk = dk_part
                         dv = dv_part
                    else:
                         # This block is tricky because dk/dv are accumulating.
                         # If step 0 is executed, it contributes to local k/v gradients.
                         # But wait, `dk` variable holds gradients for *circulating* k/v?
                         # The logic: `dk = block_dk_buffer + next_dk`.
                         # `next_dk` comes from previous iteration (which processed step+1 k/v).
                         # So `dk` is accumulating gradient for current `k` (which is `k` at step `rank`).
                         
                         # If step 0: we computed grad for `cat([joint, k_rank])`? 
                         # No. `k` in the loop is changing.
                         # In Forward: `k` iterates.
                         # In Backward: `k` iterates? 
                         # Yes: `next_k = kv_comm.send_recv(k)`.
                         
                         # So at step `s`, we have `k` from rank `rank-s`.
                         # And we compute `block_dk` for THAT `k`.
                         # And we accumulate `block_dk` into `dk`?
                         # The variable `dk` tracks the gradient for the *currently held* `k` block?
                         # No.
                         
                         # Let's look at standard Ring Attention Backward.
                         # `next_dk = d_kv_comm.send_recv(dk)`.
                         # We send `dk` to neighbor.
                         # So `dk` travels along with `k`?
                         # Yes.
                         
                         pass

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

    return dq.to(torch.bfloat16), next_dk.to(q.dtype), next_dv.to(q.dtype), d_joint_k, d_joint_v


class RingFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
        attn_type,
        attn_processor,
        joint_tensor_key=None,
        joint_tensor_value=None,
        joint_strategy="front",
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        out, softmax_lse = ring_flash_attn_forward(
            group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=False,
            attn_type=attn_type,
            attn_processor=attn_processor,
            joint_tensor_key=joint_tensor_key,
            joint_tensor_value=joint_tensor_value,
            joint_strategy=joint_strategy,
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse, joint_tensor_key, joint_tensor_value)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        ctx.attn_type = attn_type
        ctx.attn_processor = attn_processor
        ctx.joint_strategy = joint_strategy
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, joint_tensor_key, joint_tensor_value = ctx.saved_tensors
        dq, dk, dv, d_joint_k, d_joint_v = ring_flash_attn_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            softcap=ctx.softcap,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
            attn_type=ctx.attn_type,
            joint_tensor_key=joint_tensor_key,
            joint_tensor_value=joint_tensor_value,
            joint_strategy=ctx.joint_strategy,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, d_joint_k, d_joint_v, None


def ring_flash_attn_qkvpacked_func(
    qkv,
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
):
    return RingFlashAttnFunc.apply(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        attn_type,
        None, # attn_processor
        None, # joint_tensor_key
        None, # joint_tensor_value
        "front", # joint_strategy
    )


def ring_flash_attn_kvpacked_func(
    q,
    kv,
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
):
    return RingFlashAttnFunc.apply(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        attn_type,
        None, # attn_processor
        None, # joint_tensor_key
        None, # joint_tensor_value
        "front", # joint_strategy
    )


def ring_flash_attn_func(
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
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_strategy="front",
):
    return RingFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        softcap,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        attn_type,
        attn_processor,
        joint_tensor_key,
        joint_tensor_value,
        joint_strategy,
    )