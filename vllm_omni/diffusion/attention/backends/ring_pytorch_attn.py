# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2024, Jiarui Fang.
# Adapted from https://github.com/feifeibear/long-context-attention

# adapted from https://github.com/huggingface/picotron/blob/main/picotron/context_parallel/context_parallel.py
# Copyright 2024 The HuggingFace Inc. team and Jiarui Fang.


import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.ring.ring_kernels import pytorch_attn_forward
from vllm_omni.diffusion.attention.backends.ring.ring_utils import (
    ring_kv_block_valid_length,
    update_out_and_lse,
)
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
    valid_kv_length: int | None = None,
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
        valid_kv_length,
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
        valid_kv_length: int | None = None,
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

        # Trimming the circulated K/V blocks changes their length while the
        # query block keeps the full padded length. A causal SDPA mask over a
        # rectangular (seqlen_q, seqlen_k) tile is aligned to the bottom right,
        # so a trimmed block would shift the diagonal and let queries attend to
        # the wrong key positions -- silently, with no shape error. The two are
        # therefore an unsupported combination rather than a slow path: the
        # caller must either drop the padding before the ring or run
        # non-causal.
        if is_causal and valid_kv_length is not None:
            raise ValueError(
                "valid_kv_length is not supported with causal=True in Ring Attention. "
                "Trimming a circulated K/V block shortens seqlen_k while seqlen_q keeps "
                "the padded length, and the causal diagonal is bottom-right aligned, so "
                "the mask would silently shift. Unpad before the ring, or use causal=False."
            )

        comm = RingComm(group)
        # Ensure tensors are contiguous for P2P communication
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

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
                block_rank = (comm.rank - step) % comm.world_size
                block_valid_length = ring_kv_block_valid_length(
                    valid_kv_length,
                    k.shape[1],
                    block_rank,
                    comm.world_size,
                )
                step_k = k[:, :block_valid_length]
                step_v = v[:, :block_valid_length]
                if step == 0 and joint_tensor_key is not None:
                    if joint_strategy == "front":
                        step_k = torch.cat([joint_tensor_key, step_k], dim=1)
                        step_v = torch.cat([joint_tensor_value, step_v], dim=1)
                    else:
                        step_k = torch.cat([step_k, joint_tensor_key], dim=1)
                        step_v = torch.cat([step_v, joint_tensor_value], dim=1)

                if step_k.shape[1] > 0:
                    block_out, block_lse = pytorch_attn_forward(
                        q,
                        step_k,
                        step_v,
                        softmax_scale=sm_scale,
                        causal=is_causal and step == 0,
                        op_type=op_type,
                    )
                    out, lse = update_out_and_lse(out, lse, block_out, block_lse, lse_layout="bhs")

            if step + 1 != comm.world_size:
                comm.wait()
                k = next_k
                v = next_v

        out = out.to(q.dtype)

        return out
