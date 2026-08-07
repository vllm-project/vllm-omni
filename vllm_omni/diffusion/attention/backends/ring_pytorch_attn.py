# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Jiarui Fang.
# Adapted from https://github.com/feifeibear/long-context-attention

# adapted from https://github.com/huggingface/picotron/blob/main/picotron/context_parallel/context_parallel.py
# Copyright 2024 The HuggingFace Inc. team and Jiarui Fang.


import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.ring.ring_kernels import pytorch_attn_forward
from vllm_omni.diffusion.attention.backends.ring.ring_utils import update_out_and_lse
from vllm_omni.diffusion.distributed.comm import RingComm

logger = init_logger(__name__)


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
    attn_mask=None,
):
    return RingAttentionFunc.apply(
        group,
        q,
        k,
        v,
        softmax_scale,
        causal,
        op_type,
        joint_tensor_key,
        joint_tensor_value,
        joint_strategy,
        attn_mask,
    )


class RingAttentionFunc(torch.autograd.Function):
    """Ring Attention autograd function using PyTorch SDPA (inference only, no backward)."""

    @staticmethod
    def forward(
        ctx,
        group,
        q,
        k,
        v,
        sm_scale,
        is_causal,
        op_type,
        joint_tensor_key=None,
        joint_tensor_value=None,
        joint_strategy="front",
        attn_mask=None,
    ):
        # Validate causal + joint_strategy combination
        # When causal=True and joint_strategy="rear", the causal mask would incorrectly
        # prevent local query tokens from attending to joint key tokens (which are
        # concatenated at the end). This breaks the semantics where joint tokens
        # (e.g., text conditioning) should be visible to all local tokens.
        if is_causal and joint_tensor_key is not None and joint_strategy == "rear":
            raise ValueError(
                "joint_strategy='rear' is not compatible with causal=True in Ring Attention. "
                "When using causal attention with joint tokens, use joint_strategy='front' "
                "to ensure joint tokens act as a visible prefix for all local tokens. "
                "With 'rear' strategy, the causal mask would incorrectly block local tokens "
                "from seeing the joint tokens."
            )

        comm = RingComm(group)
        # Ensure tensors are contiguous for P2P communication
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        out, lse = None, None
        next_k, next_v = None, None
        original_k_len = k.shape[1]
        joint_key_len = 0 if joint_tensor_key is None else joint_tensor_key.shape[1]
        joint_query_len = max(0, q.shape[1] - original_k_len)

        def _slice_attn_mask_for_step(step: int, step_k_len: int) -> torch.Tensor | None:
            if attn_mask is None:
                return None
            if joint_strategy != "front":
                raise ValueError("Ring attention mask slicing currently supports joint_strategy='front' only.")

            owner_rank = (comm.rank - step) % comm.world_size
            image_query_start = joint_query_len + comm.rank * original_k_len
            image_key_start = joint_key_len + owner_rank * original_k_len

            query_mask = attn_mask
            if joint_query_len:
                joint_query_mask = query_mask[..., :joint_query_len, :]
                image_query_mask = query_mask[
                    ...,
                    image_query_start : image_query_start + original_k_len,
                    :,
                ]
                query_mask = torch.cat([joint_query_mask, image_query_mask], dim=-2)
            else:
                query_mask = query_mask[
                    ...,
                    image_query_start : image_query_start + original_k_len,
                    :,
                ]

            step_mask = query_mask[..., image_key_start : image_key_start + step_k_len]
            if step == 0 and joint_key_len:
                joint_mask = query_mask[..., :joint_key_len]
                step_mask = torch.cat([joint_mask, step_mask], dim=-1)
            return None if torch.all(step_mask) else step_mask

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

                step_attn_mask = _slice_attn_mask_for_step(step, k.shape[1])
                block_out, block_lse = pytorch_attn_forward(
                    q,
                    step_k,
                    step_v,
                    softmax_scale=sm_scale,
                    causal=is_causal and step == 0,
                    op_type=op_type,
                    attn_mask=step_attn_mask,
                )
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)

            if step + 1 != comm.world_size:
                comm.wait()
                k = next_k
                v = next_v

        out = out.to(q.dtype)

        return out
