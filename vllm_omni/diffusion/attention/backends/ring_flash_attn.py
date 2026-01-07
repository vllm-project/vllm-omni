# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Jiarui Fang.
# Adapted from https://github.com/feifeibear/long-context-attention


import torch

from vllm_omni.diffusion.attention.backends.ring.ring_selector import AttnType, select_flash_attn_impl
from vllm_omni.diffusion.attention.backends.ring.ring_utils import update_out_and_lse
from vllm_omni.diffusion.distributed.comm import RingComm


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
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k: torch.Tensor
            next_v: torch.Tensor
            next_k = comm.send_recv(k)
            next_v = comm.send_recv(v)
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


class RingFlashAttnFunc(torch.autograd.Function):
    """Ring Flash Attention autograd function (inference only, no backward)."""

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
        return out if not return_softmax else (out, softmax_lse, None)


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
        None,  # attn_processor
        None,  # joint_tensor_key
        None,  # joint_tensor_value
        "front",  # joint_strategy
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
        None,  # attn_processor
        None,  # joint_tensor_key
        None,  # joint_tensor_value
        "front",  # joint_strategy
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
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, None]:
    """Ring Attention forward pass using Flash Attention backend.

    Implements Ring Attention with sequence parallelism using a ring-based P2P
    communication pattern. The sequence dimension is sharded across devices, and
    Key/Value blocks are circulated through the ring to accumulate attention results.

    Args:
        q (torch.Tensor): Query tensor of shape (batch, seq_len, num_heads, head_dim).
            Sequence dimension is sharded across the ring group.
        k (torch.Tensor): Key tensor of shape (batch, seq_len, num_heads, head_dim).
            Sequence dimension is sharded across the ring group.
        v (torch.Tensor): Value tensor of shape (batch, seq_len, num_heads, head_dim).
            Sequence dimension is sharded across the ring group.
        dropout_p (float): Dropout probability. Defaults to 0.0.
        softmax_scale (float | None): Scaling factor for softmax.
            If None, computed as head_dim^(-0.5).
        causal (bool): Whether to apply causal masking. Defaults to False.
        window_size (tuple[int, int]): Sliding window size for attention.
            (-1, -1) means no windowing.
        softcap (float): Soft capping value for attention logits. Defaults to 0.0.
        alibi_slopes (torch.Tensor | None): ALiBi slopes for positional bias.
            Not supported.
        deterministic (bool): Whether to use deterministic algorithms.
            Defaults to False.
        return_attn_probs (bool): If True, returns (out, softmax_lse, None).
            Defaults to False.
        group (ProcessGroup | None): Process group for ring communication.
            Defaults to None.
        attn_type (AttnType): Flash Attention implementation type
            (AttnType.FA, AttnType.FA3, etc.).
        attn_processor (Callable | None): Custom attention processor for sparse
            attention. Defaults to None.
        joint_tensor_key (torch.Tensor | None): Additional key tensor for joint
            attention (e.g., text + image). Concatenated only at step=0.
            Defaults to None.
        joint_tensor_value (torch.Tensor | None): Additional value tensor for
            joint attention (e.g., text + image). Concatenated only at step=0.
            Defaults to None.
        joint_strategy (str): Concatenation strategy ("front" or "back").
            Defaults to "front".

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, None]]:
            - If return_attn_probs is False: Output tensor (batch, seq_len, num_heads, head_dim).
            - If return_attn_probs is True: A tuple (out, softmax_lse, None).
    """
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
