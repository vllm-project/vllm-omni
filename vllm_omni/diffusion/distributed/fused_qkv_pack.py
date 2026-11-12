# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Private, bit-preserving layout kernels for opt-in packed ROCm Ulysses.

Import this module only after the caller selects fused staging. Communication
and buffer ownership remain in packed_qkv; these kernels only copy layouts.
"""

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_omni.diffusion.distributed.packed_qkv import _fused_qkv_eligible

if HAS_TRITON:

    @triton.jit
    def _pack_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        send_ptr,
        batch: tl.constexpr,
        tokens: tl.constexpr,
        heads: tl.constexpr,
        width: tl.constexpr,
        world: tl.constexpr,
        query_stride0: tl.constexpr,
        query_stride1: tl.constexpr,
        query_stride2: tl.constexpr,
        query_stride3: tl.constexpr,
        key_stride0: tl.constexpr,
        key_stride1: tl.constexpr,
        key_stride2: tl.constexpr,
        key_stride3: tl.constexpr,
        value_stride0: tl.constexpr,
        value_stride1: tl.constexpr,
        value_stride2: tl.constexpr,
        value_stride3: tl.constexpr,
        block_size: tl.constexpr,
    ):
        # One lane moves the same coordinate from all three producers.
        i = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = i < batch * tokens * heads * width
        d = i % width
        h = (i // width) % heads
        t = (i // (width * heads)) % tokens
        b = i // (width * heads * tokens)
        lh = heads // world
        destination = h // lh
        local_head = h % lh
        wire = (((destination * batch + b) * tokens + t) * lh + local_head) * (3 * width) + d
        q = tl.load(
            query_ptr + b * query_stride0 + t * query_stride1 + h * query_stride2 + d * query_stride3,
            mask=mask,
            other=0,
        )
        k = tl.load(key_ptr + b * key_stride0 + t * key_stride1 + h * key_stride2 + d * key_stride3, mask=mask, other=0)
        v = tl.load(
            value_ptr + b * value_stride0 + t * value_stride1 + h * value_stride2 + d * value_stride3,
            mask=mask,
            other=0,
        )
        tl.store(send_ptr + wire, q, mask=mask)
        tl.store(send_ptr + wire + width, k, mask=mask)
        tl.store(send_ptr + wire + 2 * width, v, mask=mask)

    @triton.jit
    def _unpack_kernel(
        receive_ptr,
        query_ptr,
        key_ptr,
        value_ptr,
        batch: tl.constexpr,
        tokens: tl.constexpr,
        heads: tl.constexpr,
        width: tl.constexpr,
        world: tl.constexpr,
        block_size: tl.constexpr,
    ):
        lh = heads // world
        i = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = i < batch * world * tokens * lh * width
        d = i % width
        h = (i // width) % lh
        global_t = (i // (width * lh)) % (world * tokens)
        b = i // (width * lh * world * tokens)
        source = global_t // tokens
        local_t = global_t % tokens
        wire = (((source * batch + b) * tokens + local_t) * lh + h) * (3 * width) + d
        q = tl.load(receive_ptr + wire, mask=mask, other=0)
        k = tl.load(receive_ptr + wire + width, mask=mask, other=0)
        v = tl.load(receive_ptr + wire + 2 * width, mask=mask, other=0)
        tl.store(query_ptr + i, q, mask=mask)
        tl.store(key_ptr + i, k, mask=mask)
        tl.store(value_ptr + i, v, mask=mask)


def _pack_qkv(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, world_size: int) -> torch.Tensor:
    """Create [destination,batch,tokens,H_local,QKV,width] without a stacked intermediate."""
    if not HAS_TRITON:
        raise RuntimeError("Fused QKV staging requires Triton")
    if not _fused_qkv_eligible(query, key, value, world_size):
        raise ValueError("Fused QKV staging requires eligible ROCm inference tensors")
    batch, tokens, heads, width = query.shape
    send = torch.empty(
        (world_size, batch, tokens, heads // world_size, 3, width), dtype=query.dtype, device=query.device
    )
    with torch.accelerator.device_index(query.device.index):
        _pack_kernel[(triton.cdiv(query.numel(), 1024),)](
            query,
            key,
            value,
            send,
            batch,
            tokens,
            heads,
            width,
            world_size,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            block_size=1024,
            num_warps=4,
        )
    return send


def _unpack_qkv(receive: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize three independently owned contiguous [batch,world*tokens,H_local,width] tensors."""
    if not HAS_TRITON:
        raise RuntimeError("Fused QKV staging requires Triton")
    if not (
        receive.ndim == 6
        and receive.shape[-2] == 3
        and receive.shape[0] > 1
        and all(size > 0 for size in receive.shape)
        and receive.is_contiguous()
        and receive.is_cuda
        and torch.version.hip is not None
        and receive.dtype in (torch.bfloat16, torch.float16, torch.float32, torch.int32)
        and not receive.requires_grad
        and not receive.is_neg()
        and not receive.is_conj()
        and receive.numel() < 2**31
    ):
        raise ValueError(
            "Fused unpack requires a bounded contiguous ROCm [world,batch,tokens,H_local,3,width] inference buffer"
        )
    world, batch, tokens, local_heads, _, width = receive.shape
    shape = (batch, world * tokens, local_heads, width)
    outputs = tuple(torch.empty(shape, dtype=receive.dtype, device=receive.device) for _ in range(3))
    with torch.accelerator.device_index(receive.device.index):
        _unpack_kernel[(triton.cdiv(outputs[0].numel(), 1024),)](
            receive,
            *outputs,
            batch,
            tokens,
            local_heads * world,
            width,
            world,
            block_size=1024,
            num_warps=4,
        )
    return outputs[0], outputs[1], outputs[2]
