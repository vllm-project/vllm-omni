# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Quality-oriented FA3 FP8 with centered, orthogonally rotated operands.

K centering cancels in softmax. V centering and its orthogonal rotation are
undone after attention. BF16 master pages remain unmodified. This first path
rebuilds the working operands each NFE; it is not the incremental fast path.
"""

import torch
from vllm import _custom_ops as ops
from vllm.triton_utils import tl, triton

from .fp8_hadamard import rotate


@triton.jit
def gather_operands(
    storage,
    k,
    v,
    slots,
    past_rows,
    output,
    layer: tl.constexpr,
    group: tl.constexpr,
    cap: tl.constexpr,
    physical: tl.constexpr,
    rows: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    ks0: tl.constexpr,
    ks1: tl.constexpr,
    vs0: tl.constexpr,
    vs1: tl.constexpr,
    block: tl.constexpr,
):
    side = tl.program_id(1).to(tl.int64)
    x = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    hd = heads * dim
    row, channel = x // hd, x % hd
    past = tl.load(past_rows)
    slot = tl.load(slots + group * cap + row, (row < past) & (row < cap), 0).to(tl.int64)
    old = tl.load(storage + (layer * 2 + side) * physical * hd + slot * hd + channel, (row < past) & (row < cap), 0)
    current = row - past
    if tl.program_id(1) == 0:
        new = tl.load(
            k + current * ks0 + channel // dim * ks1 + channel % dim, (current >= 0) & (current < rows) & (row < cap), 0
        )
    else:
        new = tl.load(
            v + current * vs0 + channel // dim * vs1 + channel % dim, (current >= 0) & (current < rows) & (row < cap), 0
        )
    value = tl.where(row < past, old, new)
    tl.store(output + side * cap * hd + x, value, row < cap)


def prepare(runtime, pool):
    if not hasattr(pool, "_realization_fp8_precondition"):
        device = pool.storage.device
        generator = torch.Generator(device=device).manual_seed(713)
        sign = torch.randint(0, 2, (pool.storage.shape[-1],), device=device, generator=generator) * 2 - 1
        pool._realization_fp8_precondition = dict(
            signs=sign.float(),
            raw=torch.empty(
                (2, runtime.plan.capacity, *pool.storage.shape[-2:]), dtype=pool.storage.dtype, device=device
            ),
        )
    runtime.precondition_state = pool._realization_fp8_precondition
    runtime.master_storage = pool.storage


def quantize_heads(x):
    rows, heads, dim = x.shape
    raw = x.transpose(0, 1).contiguous().reshape(heads, -1).to(torch.bfloat16)
    quant, scale = ops.scaled_fp8_quant(raw, use_per_token_if_dynamic=True)
    return quant.view(heads, rows, dim).transpose(0, 1).contiguous(), scale.view(1, heads)


def forward(runtime, q, k, v, layer, scale):
    state = runtime.precondition_state
    raw = state["raw"]
    cap, heads, dim = raw.shape[1:]
    gather_operands[(triton.cdiv(cap * heads * dim, 4096), 2)](
        runtime.master_storage,
        k,
        v,
        runtime.plan.slots,
        runtime.past_rows,
        raw,
        layer,
        runtime.media_group,
        cap,
        runtime.master_storage.shape[2] * runtime.master_storage.shape[3],
        k.shape[0],
        heads,
        dim,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        4096,
    )
    count = runtime.past_rows + k.shape[0]
    values = raw.float()
    means = values.sum(dim=1, keepdim=True) / count
    q8, qs = quantize_heads(rotate(q, state["signs"]))
    k8, ks = quantize_heads(rotate(raw[0], state["signs"], center=means[0], count=count))
    v8, vs = quantize_heads(rotate(raw[1], state["signs"], center=means[1], count=count))
    output = torch.empty_like(q)
    for i, (_, start, end) in enumerate(runtime.plan.launches):
        if i == 0 and runtime.reuse_text:
            output[start:end].copy_(runtime.text_output[layer])
            continue
        out = runtime.function(
            q8[start:end],
            k8,
            v8,
            end - start,
            runtime.cuq[i],
            cap,
            cu_seqlens_k=runtime.cuk[i],
            softmax_scale=scale,
            causal=False,
            num_splits=0,
            fa_version=3,
            q_descale=qs,
            k_descale=ks,
            v_descale=vs,
            scheduler_metadata=runtime.scheduler_metadata[i],
        )
        output[start:end].copy_(rotate(out, state["signs"], center=means[1], inverse=True))
        if i == 0 and runtime.store_text:
            runtime.text_output[layer].copy_(output[start:end])
    return output
