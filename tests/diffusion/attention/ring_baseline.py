# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The Ring forward as it stood before this branch's optimizations.

Test-only, so the shipped backend keeps exactly one implementation. Two
benchmarks use it as their A/B baseline: ``bench_ring_flash_attn_opt.py``
compares it in-process, and ``bench_wan_t2v_ring_opt.py`` injects it into
spawned engine workers.

It is a byte-for-byte copy of the upstream forward, so the only difference
against the shipped path is that it circulates K and V as two separate
transfers instead of one packed transfer. Any output difference is therefore
attributable to packing, not to a reimplementation artifact.
"""

import torch

from vllm_omni.diffusion.attention.backends.ring.ring_selector import AttnType, select_flash_attn_impl
from vllm_omni.diffusion.attention.backends.ring.ring_utils import ring_kv_block_valid_length, update_out_and_lse
from vllm_omni.diffusion.distributed.comm import RingComm

__all__ = ["baseline_ring_flash_attn_forward", "baseline_ring_flash_attn_func"]


def baseline_ring_flash_attn_forward(
    process_group,
    q,
    k,
    v,
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
    valid_kv_length: int | None = None,
):
    """Drop-in replacement for the shipped ``ring_flash_attn_forward``."""
    comm = RingComm(process_group)
    out = None
    lse = None
    next_k, next_v = None, None
    use_fused_merge = attn_type in (AttnType.FA, AttnType.FA3)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k = comm.send_recv(k)
            next_v = comm.send_recv(v)
            comm.commit()

        if not causal or step <= comm.rank:
            block_rank = (comm.rank - step) % comm.world_size
            block_valid_length = ring_kv_block_valid_length(valid_kv_length, k.shape[1], block_rank, comm.world_size)
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
                if attn_type == AttnType.SPARSE_SAGE:
                    out, lse = block_out, block_lse
                else:
                    out, lse = update_out_and_lse(
                        out, lse, block_out, block_lse, lse_layout="bhs", use_fused_merge=use_fused_merge
                    )

        if step + 1 != comm.world_size:
            comm.wait()
            k = next_k
            v = next_v

    assert out is not None
    out = out.to(q.dtype)
    if attn_type != AttnType.SPARSE_SAGE:
        assert lse is not None
        lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse


def baseline_ring_flash_attn_func(q, k, v, *, softmax_scale=None, group=None, return_attn_probs=False, **kwargs):
    """Match ``ring_flash_attn_func``'s call shape for the baseline forward."""
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out, lse = baseline_ring_flash_attn_forward(group, q, k, v, softmax_scale=softmax_scale, **kwargs)
    if return_attn_probs:
        return out, lse.contiguous(), None
    return out
