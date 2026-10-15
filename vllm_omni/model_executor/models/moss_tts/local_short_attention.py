# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Single-query attention for the frame-local MOSS depth transformer.

The caller has already written this token's K/V and provides only its valid
prefix. There is no persistent state here. BF16 probability rounding matches
the FlashAttention P/V multiply, but reduction order is not bit-identical.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _short_attention(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    qs0: tl.constexpr,
    qs1: tl.constexpr,
    ks0: tl.constexpr,
    ks1: tl.constexpr,
    ks2: tl.constexpr,
    vs0: tl.constexpr,
    vs1: tl.constexpr,
    vs2: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    tokens: tl.constexpr,
    block_d: tl.constexpr,
    block_t: tl.constexpr,
):
    row = tl.program_id(0)
    batch, head = row // heads, row % heads
    d, t = tl.arange(0, block_d), tl.arange(0, block_t)
    query = tl.load(q_ptr + batch * qs0 + head * qs1 + d, d < dim, 0).to(tl.float32)
    keys = tl.load(
        k_ptr + batch * ks0 + head * ks1 + t[:, None] * ks2 + d[None, :],
        (t[:, None] < tokens) & (d[None, :] < dim),
        0,
    ).to(tl.float32)
    logits = tl.sum(keys * query[None, :], axis=1) * (dim**-0.5 * 1.4426950408889634)
    logits = tl.where(t < tokens, logits, float("-inf"))
    probs = tl.exp2(logits - tl.max(logits, axis=0))
    denominator = tl.sum(probs, axis=0)
    values = tl.load(
        v_ptr + batch * vs0 + head * vs1 + t[:, None] * vs2 + d[None, :],
        (t[:, None] < tokens) & (d[None, :] < dim),
        0,
    )
    probs = probs.to(values.dtype).to(tl.float32)
    output = tl.sum(values.to(tl.float32) * probs[:, None], axis=0) / denominator
    tl.store(out_ptr + row * dim + d, output, d < dim)


@torch.library.custom_op("vllm_omni::moss_local_short_attention", mutates_args=())
def local_short_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    batch, heads, seq_len, dim = query.shape
    if not (
        query.is_cuda
        and key.device == query.device
        and value.device == query.device
        and query.dtype == key.dtype == value.dtype == torch.bfloat16
        and seq_len == 1
        and 0 < dim <= 128
        and key.shape == value.shape
        and key.shape[:2] == query.shape[:2]
        and key.shape[-1] == dim
        and 0 < key.shape[2] <= 16
        and query.stride(-1) == key.stride(-1) == value.stride(-1) == 1
    ):
        raise ValueError("MOSS short attention requires CUDA BF16 single-query inputs and 1..16 valid keys")
    output = query.new_empty((batch, heads, 1, dim))
    _short_attention[(batch * heads,)](
        query,
        key,
        value,
        output,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        heads,
        dim,
        key.shape[2],
        triton.next_power_of_2(dim),
        triton.next_power_of_2(key.shape[2]),
        num_warps=1,
    )
    return output


@local_short_attention.register_fake
def _fake_local_short_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return query.new_empty(query.shape)
