# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""FA3 for full codec chunks that overwrite the entire KV ring.

Caller contract: lengths are either zero or padded T, slots are unique. This
matches StreamingExecutionContext.valid_rows * T, not arbitrary partial rows.
The general ring op remains the fallback for chunks shorter than the ring.
"""

import math

import torch
from vllm.triton_utils import tl, triton
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func  # noqa: F401

from .slot_attention import _advance, _rotate, slot_ring_attention


@triton.jit
def _prepare(
    q_ptr,
    k_ptr,
    v_ptr,
    rope_ptr,
    cache_ptr,
    end_ptr,
    slots_ptr,
    lengths_ptr,
    qr_ptr,
    kr_ptr,
    out_ptr,
    used_ptr,
    q0: tl.constexpr,
    q1: tl.constexpr,
    q2: tl.constexpr,
    q3: tl.constexpr,
    k0: tl.constexpr,
    k1: tl.constexpr,
    k2: tl.constexpr,
    k3: tl.constexpr,
    v0: tl.constexpr,
    v1: tl.constexpr,
    v2: tl.constexpr,
    v3: tl.constexpr,
    rs: tl.constexpr,
    es: tl.constexpr,
    ss: tl.constexpr,
    ls: tl.constexpr,
    s: tl.constexpr,
    h: tl.constexpr,
    c: tl.constexpr,
    d: tl.constexpr,
    t: tl.constexpr,
    rotate: tl.constexpr,
    scale: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(slots_ptr + row * ss)
    length = tl.load(lengths_ptr + row * ls)
    start = tl.load(end_ptr + slot * es)
    x = tl.program_id(1) * block + tl.arange(0, block)
    dim = x % d
    head = (x // d) % h
    token = x // (h * d)
    source = token + (t - c)
    valid = (token < c) & (length > 0)
    physical = (start + source) % c
    dst = ((slot * h + head) * c + physical) * d + dim
    kb = k_ptr + row * k0 + head * k1 + source * k2
    kval = tl.load(kb + dim * k3, valid, 0)
    vval = tl.load(v_ptr + row * v0 + head * v1 + source * v2 + dim * v3, valid, 0)
    if rotate:
        position = tl.load(rope_ptr + row * rs) + source
        kp = tl.load(kb + (dim ^ 1) * k3, valid, 0)
        kval = _rotate(kval, kp, dim, position, scale)
        qb = q_ptr + row * q0 + head * q1 + source * q2
        qval = tl.load(qb + dim * q3, valid, 0)
        qp = tl.load(qb + (dim ^ 1) * q3, valid, 0)
        qval = _rotate(qval, qp, dim, position, scale)
        tl.store(qr_ptr + row * c * h * d + x, qval, token < c)
        tl.store(kr_ptr + row * c * h * d + x, kval, token < c)
    tl.store(cache_ptr + dst, kval, valid)
    tl.store(cache_ptr + s * h * c * d + dst, vval, valid)
    tl.store(out_ptr + row * t * h * d + x, 0, (token < t) & ((token < t - c) | (length == 0)))
    if tl.program_id(1) == 0:
        tl.store(used_ptr + row, tl.where(length > 0, c, 0))


@torch.library.custom_op("vllm_omni::moss_codec_direct_fa3", mutates_args=("cache", "end"))
def direct_slot(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache: torch.Tensor,
    end: torch.Tensor,
    slots: torch.Tensor,
    lengths: torch.Tensor,
    context: int = -1,
    rope_offset: torch.Tensor | None = None,
    max_period: float = 10000.0,
) -> torch.Tensor:
    b, h, t, d = q.shape
    s, c = cache.shape[1], cache.shape[3]
    assert t >= c and d == 64 and q.dtype == torch.bfloat16
    strides = (end.stride(0), slots.stride(0), lengths.stride(0))
    output = torch.empty((b, t, h, d), device=q.device, dtype=q.dtype)
    used = torch.empty(b, device=q.device, dtype=torch.int32)
    rotate = rope_offset is not None
    if rotate:
        qr = torch.empty((b, c, h, d), device=q.device, dtype=q.dtype)
        kr = torch.empty_like(qr)
    else:
        qr = q.transpose(1, 2)[:, t - c :]
        kr = k.transpose(1, 2)[:, t - c :]
    rope = rope_offset if rotate else end
    _prepare[(b, triton.cdiv(t * h * d, 256))](
        q,
        k,
        v,
        rope,
        cache,
        end,
        slots,
        lengths,
        qr,
        kr,
        output,
        used,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        rope.stride(0),
        *strides,
        s,
        h,
        c,
        d,
        t,
        rotate,
        -math.log(max_period) * 2 / d,
        256,
    )
    torch.ops._vllm_fa3_C.fwd(
        qr,
        kr,
        v.transpose(1, 2)[:, t - c :],
        None,
        None,
        None,
        output[:, t - c :],
        None,
        None,
        None,
        used,
        used,
        c,
        c,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        d**-0.5,
        True,
        context - 1 if context > 0 else -1,
        0,
        0.0,
        True,
        None,
        1,
        None,
        0,
        None,
        1,
        0,
        None,
    )
    _advance[(1,)](end, slots, lengths, *strides, b, triton.next_power_of_2(b))
    return output.transpose(1, 2)


@direct_slot.register_fake
def _(q, k, v, cache, end, slots, lengths, context=-1, rope_offset=None, max_period=10000.0):
    b, h, t, d = q.shape
    return torch.empty((b, t, h, d), device=q.device, dtype=q.dtype).transpose(1, 2)


@torch.library.custom_op("vllm_omni::moss_codec_selective_attention", mutates_args=("cache", "end"))
def selective_slot(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache: torch.Tensor,
    end: torch.Tensor,
    slots: torch.Tensor,
    lengths: torch.Tensor,
    context: int = -1,
    rope_offset: torch.Tensor | None = None,
    max_period: float = 10000.0,
) -> torch.Tensor:
    # vLLM reuses the AOT graph across first (T=1) and steady (T=15) codec
    # calls. Dispatch must be opaque to tracing, or the first shape specializes
    # the branch and a shorter subsequent chunk incorrectly enters direct FA3.
    if q.shape[2] >= cache.shape[3] and q.shape[0] >= 8 and q.shape[3] == 64 and q.dtype == torch.bfloat16:
        return direct_slot(q, k, v, cache, end, slots, lengths, context, rope_offset, max_period)
    value = slot_ring_attention(q, k, v, cache, end, slots, lengths, context, rope_offset, max_period)
    # Both runtime branches must match the same fake output strides, including
    # singleton dimensions. This also preserves the no-copy head merge in FA3.
    b, h, t, d = q.shape
    output = torch.empty((b, t, h, d), device=q.device, dtype=q.dtype)
    output.copy_(value.transpose(1, 2))
    return output.transpose(1, 2)


@selective_slot.register_fake
def _(q, k, v, cache, end, slots, lengths, context=-1, rope_offset=None, max_period=10000.0):
    b, h, t, d = q.shape
    return torch.empty((b, t, h, d), device=q.device, dtype=q.dtype).transpose(1, 2)


selective_slot.supports_rope = True
