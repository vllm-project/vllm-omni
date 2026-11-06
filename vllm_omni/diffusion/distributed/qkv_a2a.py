# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage and exchange equal q_ptr/k_ptr/v_ptr together over symmetric memory."""

import torch
from vllm.triton_utils import tl, triton

from vllm_omni.diffusion.distributed.a2a_permute import _ensure_built, _get_symm_buffer


@triton.jit
def _stage(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    elements: tl.constexpr,
    sequence: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    world_size: tl.constexpr,
    q_stride0: tl.constexpr,
    q_stride1: tl.constexpr,
    k_stride0: tl.constexpr,
    k_stride1: tl.constexpr,
    v_stride0: tl.constexpr,
    v_stride1: tl.constexpr,
    block_size: tl.constexpr,
):
    index = tl.program_id(0) * block_size + tl.arange(0, block_size)
    plane = tl.program_id(1)
    lc = heads // world_size * dim
    row = index // (heads * dim)
    token = row % sequence
    batch = row // sequence
    channel = index % (heads * dim)
    if plane == 0:
        value = tl.load(q_ptr + batch * q_stride0 + token * q_stride1 + channel, index < elements, 0)
    elif plane == 1:
        value = tl.load(k_ptr + batch * k_stride0 + token * k_stride1 + channel, index < elements, 0)
    else:
        value = tl.load(v_ptr + batch * v_stride0 + token * v_stride1 + channel, index < elements, 0)
    dest = (row * world_size + channel // lc) * (3 * lc) + plane * lc + channel % lc
    tl.store(output_ptr + dest, value, index < elements)


@triton.jit
def _unpack(
    input_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    elements: tl.constexpr,
    sequence: tl.constexpr,
    rows_per_block: tl.constexpr,
    local_cols: tl.constexpr,
    world_size: tl.constexpr,
    block_size: tl.constexpr,
):
    index = tl.program_id(0) * block_size + tl.arange(0, block_size)
    plane = tl.program_id(1)
    token = (index // local_cols) % (world_size * sequence)
    batch = index // (local_cols * world_size * sequence)
    rank = token // sequence
    row = batch * sequence + token % sequence
    src = ((rank * rows_per_block + row) * 3 + plane) * local_cols + index % local_cols
    value = tl.load(input_ptr + src, index < elements, 0)
    if plane == 0:
        tl.store(q_ptr + index, value, index < elements)
    elif plane == 1:
        tl.store(k_ptr + index, value, index < elements)
    else:
        tl.store(v_ptr + index, value, index < elements)


def eligible(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, world_size: int) -> bool:
    return (
        world_size > 1
        and q.ndim == 4
        and q.device.type == "cuda"
        and q.dtype == torch.bfloat16
        and q.shape == k.shape == v.shape
        and q.shape[2] % world_size == 0
        and q.shape[1] > 0
        and all(
            t.device == q.device and t.dtype == q.dtype and t.stride(-1) == 1 and t.stride(-2) == t.shape[-1]
            for t in (q, k, v)
        )
    )


@torch.library.custom_op("vllm_omni_a2a::qkv_fwd_batched", mutates_args=(), device_types="cuda")
def qkv_fwd_batched(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, group_name: str, world_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _ensure_built()
    batch, sequence, heads, dim = q.shape
    rows, local_cols = batch * sequence, heads // world_size * dim
    symm = _get_symm_buffer((rows, world_size, 3 * local_cols), q.dtype, q.device, group_name)
    _stage[(triton.cdiv(q.numel(), 1024), 3)](
        q,
        k,
        v,
        symm,
        q.numel(),
        sequence,
        heads,
        dim,
        world_size,
        *q.stride()[:2],
        *k.stride()[:2],
        *v.stride()[:2],
        1024,
    )
    received = torch.empty((world_size, rows, 3 * local_cols), device=q.device, dtype=q.dtype)
    torch.ops.a2ap.all_to_all_permute(symm, received, 1, 0, group_name)
    output_query = torch.empty((batch, world_size * sequence, heads // world_size, dim), device=q.device, dtype=q.dtype)
    output_key = torch.empty_like(output_query)
    output_value = torch.empty_like(output_query)
    outputs = (output_query, output_key, output_value)
    _unpack[(triton.cdiv(q.numel(), 1024), 3)](
        received,
        *outputs,
        q.numel(),
        sequence,
        rows,
        local_cols,
        world_size,
        1024,
    )
    return outputs


@qkv_fwd_batched.register_fake
def _(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, group_name: str, world_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (q.shape[0], q.shape[1] * world_size, q.shape[2] // world_size, q.shape[3])
    return q.new_empty(shape), q.new_empty(shape), q.new_empty(shape)
