# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Official Hopper FA3 FP8 compute over an incremental FP8 KV workspace."""

import torch
from vllm import _custom_ops as ops
from vllm.triton_utils import triton

from .attention_partition import broadcast_partition_attention, merge_partitions
from .native_attention import append_current


def fp8_forward(runtime, q, k, v, layer, scale):
    if runtime.fp8_incremental:
        from .fp8_incremental import forward

        return forward(runtime, q, k, v, layer, scale)
    if runtime.fp8_precondition:
        from .fp8_precondition import forward

        return forward(runtime, q, k, v, layer, scale)
    p = runtime.plan
    h, d = q.shape[1:]
    # The existing dynamic per-token CUDA quantizer treats each head as one
    # token here, yielding one descale per head without a CPU scalar read.
    transposed = q.transpose(0, 1).contiguous().view(h, -1)
    quant, descale = ops.scaled_fp8_quant(transposed, use_per_token_if_dynamic=True)
    q8 = quant.view(h, q.shape[0], d).transpose(0, 1).contiguous()
    q_scale = descale.view(1, h)
    output = torch.empty_like(q)
    ko, vo = runtime.state["buffer"][layer]
    ks, vs = runtime.kv_scales[layer]
    append_current[(triton.cdiv(k.shape[0] * h * d, 4096),)](
        k,
        v,
        ko,
        vo,
        runtime.past_rows,
        k.shape[0],
        h,
        d,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        4096,
        k_scale=ks,
        v_scale=vs,
        fp8=True,
        num_warps=8,
    )
    for i, (_, start, end) in enumerate(p.launches):
        if i == 0 and runtime.reuse_text:
            output[start:end].copy_(runtime.text_output[layer])
            continue
        if i == 1 and runtime.media_partitions > 1:
            parts = runtime.media_partitions
            partial, lse = broadcast_partition_attention(
                q8[start:end],
                ko,
                vo,
                runtime.partition_cuk,
                triton.cdiv(p.capacity, parts),
                scale,
                parts,
                q_scale.expand(parts, -1).contiguous(),
                runtime.fp8_partition_scales[layer, 0],
                runtime.fp8_partition_scales[layer, 1],
            )
            ls_part, ls_head, ls_row = lse.stride()
            merge_partitions[(end - start, h)](
                partial,
                lse,
                output[start:end],
                end - start,
                h,
                d,
                parts,
                ls_head,
                ls_row,
                triton.next_power_of_2(parts),
                triton.next_power_of_2(d),
                lse_part=ls_part,
                num_warps=runtime.merge_warps,
            )
        else:
            runtime.function(
                q8[start:end],
                ko,
                vo,
                end - start,
                runtime.cuq[i],
                p.capacity,
                cu_seqlens_k=runtime.cuk[i],
                softmax_scale=scale,
                causal=False,
                num_splits=0,
                fa_version=3,
                out=output[start:end],
                q_descale=q_scale,
                k_descale=ks[None],
                v_descale=vs[None],
                scheduler_metadata=runtime.scheduler_metadata[i],
            )
        if i == 0 and runtime.store_text:
            runtime.text_output[layer].copy_(output[start:end])
    return output
