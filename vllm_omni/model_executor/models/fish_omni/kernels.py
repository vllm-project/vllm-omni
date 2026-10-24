# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803 - Triton constexpr parameters use kernel-style capitals.
"""Fused Triton kernels for the Drama 3 codebook tower and frame sampling.

The eager tower launched about 100 kernels per layer (norms, masks, rotary,
concatenation); these kernels keep it to the GEMMs, attention and a handful of
fused element-wise launches. Numerics follow vLLM's native Gemma RMSNorm:
fp32 statistics, ``weight + 1`` in fp32, fused-add residual rounded to the
activation dtype.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


def _triton_device(tensor: torch.Tensor) -> bool:
    """Triton tower and sampling kernels are compiled for CUDA."""
    return tensor.device.type == "cuda"


def eager_qk_rmsnorm_rope_gate(
    q_gate: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    mrope_section: list[int] | tuple[int, int, int] | None = None,
    norm_beta: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch QK-norm + NeoX RoPE + gate split.

    The fused Triton kernel does not compile on 910B. Per-head layout is
    ``[q | gate]``, RMSNorm is Gemma-style (``weight + norm_beta``), and RoPE
    covers the first ``rotary_dim`` channels.
    """
    tokens = q_gate.shape[0]
    q_heads = q_gate.view(tokens, num_q_heads, 2 * head_dim)
    query = q_heads[..., :head_dim]
    gate = q_heads[..., head_dim:]
    key = k.view(tokens, num_kv_heads, head_dim)

    def norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        inv_rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (xf * inv_rms * (weight.float() + norm_beta)).to(dtype=x.dtype)

    def rope(x: torch.Tensor) -> torch.Tensor:
        half = rotary_dim // 2
        rot = x[..., :rotary_dim].float()
        tail = x[..., rotary_dim:]
        x1, x2 = rot[..., :half], rot[..., half:]
        if positions.ndim == 1:
            picked = cos_sin_cache.index_select(0, positions.to(torch.long))
            cos = picked[:, :half].float()[:, None, :]
            sin = picked[:, half:rotary_dim].float()[:, None, :]
        else:
            index = torch.arange(half, device=x.device)
            section_h = 0 if mrope_section is None else int(mrope_section[1])
            section_w = 0 if mrope_section is None else int(mrope_section[2])
            is_h = (index % 3 == 1) & (index < 3 * section_h)
            is_w = (index % 3 == 2) & (index < 3 * section_w)
            pos = positions[0].to(torch.long)[:, None].expand(tokens, half).clone()
            pos[:, is_h] = positions[1].to(torch.long)[:, None]
            pos[:, is_w] = positions[2].to(torch.long)[:, None]
            picked = cos_sin_cache.index_select(0, pos.reshape(-1)).view(tokens, half, rotary_dim)
            cos = picked[..., :half].float()
            sin = picked[..., half:rotary_dim].float()
        rotated = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).to(dtype=x.dtype)
        return torch.cat((rotated, tail), dim=-1)

    query = rope(norm(query, q_weight))
    key = rope(norm(key, k_weight))
    return (
        query.reshape(tokens, num_q_heads * head_dim).contiguous(),
        key.reshape(tokens, num_kv_heads * head_dim).contiguous(),
        gate.reshape(tokens, num_q_heads * head_dim).contiguous(),
    )


@triton.jit
def _gemma_norm_row(x, w_ptr, offs, mask, n: tl.constexpr, eps):
    var = tl.sum(x * x, axis=0) / n
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32) + 1.0
    return x * tl.rsqrt(var + eps) * w


@triton.jit
def _mix_norm_kernel(
    hidden_ptr,
    hidden_stride,
    code_ptr,
    code_mask_ptr,
    table_ptr,
    table_rows,
    extra_ptr,
    extra_stride,
    enorm_ptr,
    hnorm_ptr,
    out_ptr,
    out_stride,
    eps,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_EXTRA: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    code = tl.load(code_ptr + row)
    code = tl.minimum(tl.maximum(code, 0), table_rows - 1)
    keep = tl.load(code_mask_ptr + row).to(tl.float32)
    emb = tl.load(table_ptr + code * D + offs, mask=mask, other=0.0).to(tl.float32) * keep
    if HAS_EXTRA:
        emb = emb.to(out_ptr.dtype.element_ty).to(tl.float32)
        extra = tl.load(extra_ptr + row * extra_stride + offs, mask=mask, other=0.0).to(tl.float32)
        emb = emb + extra
    emb = emb.to(out_ptr.dtype.element_ty).to(tl.float32)
    hidden = tl.load(hidden_ptr + row * hidden_stride + offs, mask=mask, other=0.0).to(tl.float32)
    out_row = out_ptr + row * out_stride
    tl.store(out_row + offs, _gemma_norm_row(emb, enorm_ptr, offs, mask, D, eps), mask=mask)
    tl.store(out_row + D + offs, _gemma_norm_row(hidden, hnorm_ptr, offs, mask, D, eps), mask=mask)


def tower_mix_norm(
    hidden: torch.Tensor,
    code: torch.Tensor,
    code_mask: torch.Tensor,
    table: torch.Tensor,
    extra: torch.Tensor | None,
    enorm_weight: torch.Tensor,
    hnorm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """``cat(enorm(mask * table[code] (+ extra)), hnorm(hidden))``, ``[rows, 2 * dim]``."""
    rows, dim = hidden.shape
    out = torch.empty((rows, 2 * dim), dtype=hidden.dtype, device=hidden.device)
    if rows == 0:
        return out
    if not _triton_device(hidden):
        return _eager_tower_mix_norm(hidden, code, code_mask, table, extra, enorm_weight, hnorm_weight, eps, out)
    _mix_norm_kernel[(rows,)](
        hidden,
        hidden.stride(0),
        code,
        code_mask,
        table,
        table.shape[0],
        extra if extra is not None else hidden,
        extra.stride(0) if extra is not None else 0,
        enorm_weight,
        hnorm_weight,
        out,
        out.stride(0),
        eps,
        D=dim,
        BLOCK=triton.next_power_of_2(dim),
        HAS_EXTRA=extra is not None,
        num_warps=8,
    )
    return out


@triton.jit
def _gemma_norm_kernel(
    x_ptr,
    x_stride,
    residual_ptr,
    residual_stride,
    w_ptr,
    out_ptr,
    out_stride,
    residual_out_ptr,
    eps,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    x = tl.load(x_ptr + row * x_stride + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        x = x + tl.load(residual_ptr + row * residual_stride + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(residual_out_ptr + row * D + offs, x.to(residual_out_ptr.dtype.element_ty), mask=mask)
    tl.store(out_ptr + row * out_stride + offs, _gemma_norm_row(x, w_ptr, offs, mask, D, eps), mask=mask)


def gemma_rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, residual: torch.Tensor | None = None
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Zero-centered RMSNorm; with ``residual`` returns ``(norm(x + r), x + r)``."""
    rows, dim = x.shape
    out = torch.empty_like(x)
    residual_out = torch.empty_like(x) if residual is not None else out
    if rows and not _triton_device(x):
        return _eager_gemma_rms_norm(x, weight, eps, residual, out, residual_out)
    if rows:
        _gemma_norm_kernel[(rows,)](
            x,
            x.stride(0),
            residual if residual is not None else x,
            residual.stride(0) if residual is not None else 0,
            weight,
            out,
            out.stride(0),
            residual_out,
            eps,
            D=dim,
            BLOCK=triton.next_power_of_2(dim),
            HAS_RESIDUAL=residual is not None,
            num_warps=8,
        )
    return (out, residual_out) if residual is not None else out


@triton.jit
def _table_norm_kernel(
    x_ptr,
    x_stride,
    code_ptr,
    code_mask_ptr,
    table_ptr,
    table_rows,
    w_ptr,
    out_ptr,
    residual_out_ptr,
    eps,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    code = tl.load(code_ptr + row)
    code = tl.minimum(tl.maximum(code, 0), table_rows - 1)
    keep = tl.load(code_mask_ptr + row).to(tl.float32)
    x = tl.load(x_ptr + row * x_stride + offs, mask=mask, other=0.0).to(tl.float32)
    x = x + tl.load(table_ptr + code * D + offs, mask=mask, other=0.0).to(tl.float32) * keep
    tl.store(residual_out_ptr + row * D + offs, x.to(residual_out_ptr.dtype.element_ty), mask=mask)
    tl.store(out_ptr + row * D + offs, _gemma_norm_row(x, w_ptr, offs, mask, D, eps), mask=mask)


def gemma_rms_norm_table(
    x: torch.Tensor,
    code: torch.Tensor,
    code_mask: torch.Tensor,
    table: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``s = x + mask * table[code]``; returns ``(norm(s), s)`` like :func:`gemma_rms_norm`."""
    rows, dim = x.shape
    out = torch.empty((rows, dim), dtype=x.dtype, device=x.device)
    residual_out = torch.empty_like(out)
    if rows and not _triton_device(x):
        return _eager_gemma_rms_norm_table(x, code, code_mask, table, weight, eps, out, residual_out)
    if rows:
        _table_norm_kernel[(rows,)](
            x,
            x.stride(0),
            code,
            code_mask,
            table,
            table.shape[0],
            weight,
            out,
            residual_out,
            eps,
            D=dim,
            BLOCK=triton.next_power_of_2(dim),
            num_warps=8,
        )
    return out, residual_out


@triton.jit
def _sigmoid_gate_kernel(x_ptr, gate_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, (x * tl.sigmoid(gate)).to(out_ptr.dtype.element_ty), mask=mask)


def sigmoid_gate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """``x * sigmoid(gate)`` for contiguous tensors of one shape."""
    n = x.numel()
    if n and not _triton_device(x):
        return (x.float() * torch.sigmoid(gate.float())).to(dtype=x.dtype)
    out = torch.empty_like(x)
    if n:
        block = 2048
        _sigmoid_gate_kernel[(triton.cdiv(n, block),)](x, gate, out, n, BLOCK=block, num_warps=4)
    return out


@triton.jit
def _draw_kernel(
    logits_ptr,
    logits_stride,
    vocab,
    top_values_ptr,
    top_ids_ptr,
    top_stride,
    noise_ptr,
    noise_stride,
    temperature_ptr,
    top_k_ptr,
    top_p_ptr,
    id_map_ptr,
    out_ptr,
    min_temperature,
    BLOCK_V: tl.constexpr,
    K: tl.constexpr,
    HAS_MAP: tl.constexpr,
):
    row = tl.program_id(0)
    temperature = tl.maximum(tl.load(temperature_ptr + row), min_temperature)
    offs = tl.arange(0, BLOCK_V)
    logits = tl.load(logits_ptr + row * logits_stride + offs, mask=offs < vocab, other=float("-inf"))
    logits = logits.to(tl.float32) / temperature
    row_max = tl.max(logits, axis=0)
    total = tl.sum(tl.exp(logits - row_max), axis=0)
    ranks = tl.arange(0, K)
    values = tl.load(top_values_ptr + row * top_stride + ranks).to(tl.float32) / temperature
    ids = tl.load(top_ids_ptr + row * top_stride + ranks)
    probs = tl.exp(values - row_max) / total
    cumulative = tl.cumsum(probs, axis=0)
    keep = ((cumulative <= tl.load(top_p_ptr + row)) & (ranks < tl.load(top_k_ptr + row))) | (ranks == 0)
    noise = -tl.log(tl.maximum(tl.load(noise_ptr + row * noise_stride + ranks), 1e-30))
    score = tl.where(keep, probs / noise, -1.0)
    choice = tl.argmax(score, axis=0)
    token = tl.sum(tl.where(ranks == choice, ids, 0), axis=0)
    if HAS_MAP:
        token = tl.load(id_map_ptr + token)
    tl.store(out_ptr + row, token.to(tl.int64))


def draw_top_k_top_p(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    uniform: torch.Tensor,
    max_top_k: int,
    id_map: torch.Tensor | None = None,
    top: tuple[torch.Tensor, torch.Tensor] | None = None,
    min_temperature: float = 1e-5,
) -> torch.Tensor:
    """Sample one index per row in fp32, top-k capped at ``max_top_k``.

    Ranks ``>= top_k`` and candidates whose inclusive cumulative probability
    exceeds ``top_p`` (over the whole row's distribution) are dropped; rank 0 is
    always kept, so a temperature at ``min_temperature`` takes the argmax.
    ``uniform`` is ``[rows, >= max_top_k]`` U(0, 1) noise; ``top`` reuses a
    ``torch.topk(logits, max_top_k)`` result shared by several draws; ``id_map``
    maps column indices to token ids.
    """
    rows, vocab = logits.shape
    out = torch.empty(rows, dtype=torch.int64, device=logits.device)
    if rows == 0:
        return out
    if not _triton_device(logits):
        return _eager_draw_top_k_top_p(
            logits, temperature, top_k, top_p, uniform, max_top_k, id_map, top, min_temperature
        )
    if top is None:
        top = torch.topk(logits, max_top_k, dim=-1)
    values, ids = top
    _draw_kernel[(rows,)](
        logits,
        logits.stride(0),
        vocab,
        values,
        ids,
        values.stride(0),
        uniform,
        uniform.stride(0),
        temperature,
        top_k,
        top_p,
        id_map if id_map is not None else out,
        out,
        min_temperature,
        BLOCK_V=triton.next_power_of_2(vocab),
        K=max_top_k,
        HAS_MAP=id_map is not None,
        num_warps=8,
    )
    return out


def _gemma_from_fp32(values: torch.Tensor, weight: torch.Tensor, eps: float, dtype: torch.dtype) -> torch.Tensor:
    variance = (values * values).sum(dim=-1, keepdim=True) / values.shape[-1]
    return (values * torch.rsqrt(variance + eps) * (weight.float() + 1.0)).to(dtype=dtype)


def _eager_tower_mix_norm(
    hidden: torch.Tensor,
    code: torch.Tensor,
    code_mask: torch.Tensor,
    table: torch.Tensor,
    extra: torch.Tensor | None,
    enorm_weight: torch.Tensor,
    hnorm_weight: torch.Tensor,
    eps: float,
    out: torch.Tensor,
) -> torch.Tensor:
    dim = hidden.shape[-1]
    kept = table[code.clamp(0, table.shape[0] - 1)].float() * code_mask.float().unsqueeze(-1)
    if extra is not None:
        kept = kept.to(dtype=out.dtype).float() + extra.float()
    kept = kept.to(dtype=out.dtype).float()
    out[:, :dim] = _gemma_from_fp32(kept, enorm_weight, eps, out.dtype)
    out[:, dim:] = _gemma_from_fp32(hidden.float(), hnorm_weight, eps, out.dtype)
    return out


def _eager_gemma_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None,
    out: torch.Tensor,
    residual_out: torch.Tensor,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    summed = x.float()
    if residual is not None:
        summed = summed + residual.float()
        residual_out.copy_(summed.to(dtype=residual_out.dtype))
    out.copy_(_gemma_from_fp32(summed, weight, eps, out.dtype))
    return (out, residual_out) if residual is not None else out


def _eager_gemma_rms_norm_table(
    x: torch.Tensor,
    code: torch.Tensor,
    code_mask: torch.Tensor,
    table: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    residual_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    kept = code_mask.float().unsqueeze(-1)
    summed = x.float() + table[code.clamp(0, table.shape[0] - 1)].float() * kept
    residual_out.copy_(summed.to(dtype=residual_out.dtype))
    out.copy_(_gemma_from_fp32(summed, weight, eps, out.dtype))
    return out, residual_out


def _eager_draw_top_k_top_p(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    uniform: torch.Tensor,
    max_top_k: int,
    id_map: torch.Tensor | None,
    top: tuple[torch.Tensor, torch.Tensor] | None,
    min_temperature: float,
) -> torch.Tensor:
    scaled_temperature = temperature.float().clamp_min(min_temperature).unsqueeze(-1)
    logits_f = logits.float() / scaled_temperature
    row_max = logits_f.amax(dim=-1, keepdim=True)
    total = torch.exp(logits_f - row_max).sum(dim=-1, keepdim=True)
    if top is None:
        values, ids = torch.topk(logits, max_top_k, dim=-1)
    else:
        values, ids = top
    probs = torch.exp(values.float() / scaled_temperature - row_max) / total
    cumulative = torch.cumsum(probs, dim=-1)
    ranks = torch.arange(max_top_k, device=logits.device)
    keep = ((cumulative <= top_p.float().unsqueeze(-1)) & (ranks < top_k.unsqueeze(-1))) | (ranks == 0)
    noise = -torch.log(uniform[:, :max_top_k].clamp_min(1e-30))
    score = torch.where(keep, probs / noise, torch.full_like(probs, -1.0))
    token = ids.gather(-1, score.argmax(dim=-1, keepdim=True)).squeeze(-1)
    if id_map is not None:
        token = id_map[token]
    return token.to(dtype=torch.int64)


__all__ = ["draw_top_k_top_p", "gemma_rms_norm", "gemma_rms_norm_table", "sigmoid_gate", "tower_mix_norm"]
