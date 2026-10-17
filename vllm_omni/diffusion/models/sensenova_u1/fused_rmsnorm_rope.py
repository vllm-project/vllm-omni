# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def qk_norm_rope_kernel(
    q,
    k,
    query,
    key,
    value,
    value_output,
    q_norm_weight,
    k_norm_weight,
    q_norm_hw_weight,
    k_norm_hw_weight,
    cos_t,
    sin_t,
    cos_h,
    sin_h,
    cos_w,
    sin_w,
    q_stride_b: tl.constexpr,
    q_stride_s: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_s: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_d: tl.constexpr,
    query_stride_b: tl.constexpr,
    query_stride_h: tl.constexpr,
    query_stride_s: tl.constexpr,
    query_stride_d: tl.constexpr,
    key_stride_b: tl.constexpr,
    key_stride_h: tl.constexpr,
    key_stride_s: tl.constexpr,
    key_stride_d: tl.constexpr,
    cos_t_stride_b: tl.constexpr,
    cos_t_stride_s: tl.constexpr,
    cos_t_stride_d: tl.constexpr,
    cos_h_stride_b: tl.constexpr,
    cos_h_stride_s: tl.constexpr,
    cos_h_stride_d: tl.constexpr,
    cos_w_stride_b: tl.constexpr,
    cos_w_stride_s: tl.constexpr,
    cos_w_stride_d: tl.constexpr,
    value_stride_b: tl.constexpr,
    value_stride_s: tl.constexpr,
    value_stride_h: tl.constexpr,
    value_stride_d: tl.constexpr,
    value_out_stride_b: tl.constexpr,
    value_out_stride_h: tl.constexpr,
    value_out_stride_s: tl.constexpr,
    value_out_stride_d: tl.constexpr,
    write_value: tl.constexpr,
    head_q: tl.constexpr,
    head_dim: tl.constexpr,
    eps: tl.constexpr = 1e-6,
):
    token_idx = tl.program_id(0)
    head_pid = tl.program_id(1)
    batch_idx = tl.program_id(2)
    is_q = head_pid < head_q
    head_idx = tl.where(is_q, head_pid, head_pid - head_q)

    offs = tl.arange(0, head_dim)
    t_mask = offs < head_dim // 2
    hw_mask = ~t_mask

    in_base = tl.where(
        is_q,
        q + batch_idx * q_stride_b + token_idx * q_stride_s + head_idx * q_stride_h,
        k + batch_idx * k_stride_b + token_idx * k_stride_s + head_idx * k_stride_h,
    )
    vals = tl.load(in_base + offs * tl.where(is_q, q_stride_d, k_stride_d)).to(tl.float32)

    t_vals = tl.where(t_mask, vals, 0.0)
    hw_vals = tl.where(hw_mask, vals, 0.0)
    rms_t = tl.rsqrt(tl.sum(t_vals * t_vals, axis=0) / (head_dim // 2) + eps)
    rms_hw = tl.rsqrt(tl.sum(hw_vals * hw_vals, axis=0) / (head_dim // 2) + eps)
    rms = tl.where(t_mask, rms_t, rms_hw)

    t_weight = tl.load(tl.where(is_q, q_norm_weight, k_norm_weight) + offs, mask=t_mask, other=0.0)
    hw_weight = tl.load(
        tl.where(is_q, q_norm_hw_weight, k_norm_hw_weight) + (offs - head_dim // 2),
        mask=hw_mask,
        other=0.0,
    )
    weight = tl.where(t_mask, t_weight, hw_weight)
    normed = (vals * rms).to(query.dtype.element_ty) * weight

    pair_offs = tl.where(
        offs < head_dim // 4,
        offs + head_dim // 4,
        tl.where(
            offs < head_dim // 2,
            offs - head_dim // 4,
            tl.where(
                offs < head_dim // 8 * 5,
                offs + head_dim // 8,
                tl.where(
                    offs < head_dim // 4 * 3,
                    offs - head_dim // 8,
                    tl.where(offs < head_dim // 8 * 7, offs + head_dim // 8, offs - head_dim // 8),
                ),
            ),
        ),
    )
    pair_vals = tl.load(in_base + pair_offs * tl.where(is_q, q_stride_d, k_stride_d)).to(tl.float32)
    pair_rms = tl.where(pair_offs < head_dim // 2, rms_t, rms_hw)
    pair_t_mask = pair_offs < head_dim // 2
    pair_hw_mask = ~pair_t_mask
    pair_t_weight = tl.load(
        tl.where(is_q, q_norm_weight, k_norm_weight) + pair_offs,
        mask=pair_t_mask,
        other=0.0,
    )
    pair_hw_weight = tl.load(
        tl.where(is_q, q_norm_hw_weight, k_norm_hw_weight) + (pair_offs - head_dim // 2),
        mask=pair_hw_mask,
        other=0.0,
    )
    pair_weight = tl.where(pair_offs < head_dim // 2, pair_t_weight, pair_hw_weight)
    pair_normed = (pair_vals * pair_rms).to(query.dtype.element_ty) * pair_weight

    cos_t_vals = tl.load(
        cos_t + batch_idx * cos_t_stride_b + token_idx * cos_t_stride_s + offs * cos_t_stride_d,
        mask=t_mask,
        other=0.0,
    )
    sin_t_vals = tl.load(
        sin_t + batch_idx * cos_t_stride_b + token_idx * cos_t_stride_s + offs * cos_t_stride_d,
        mask=t_mask,
        other=0.0,
    )
    hw_local = offs - head_dim // 2
    h_local = hw_local
    w_local = hw_local - head_dim // 4
    h_mask = (offs >= head_dim // 2) & (offs < head_dim // 4 * 3)
    w_mask = offs >= head_dim // 4 * 3
    cos_h_vals = tl.load(
        cos_h + batch_idx * cos_h_stride_b + token_idx * cos_h_stride_s + h_local * cos_h_stride_d,
        mask=h_mask,
        other=0.0,
    )
    sin_h_vals = tl.load(
        sin_h + batch_idx * cos_h_stride_b + token_idx * cos_h_stride_s + h_local * cos_h_stride_d,
        mask=h_mask,
        other=0.0,
    )
    cos_w_vals = tl.load(
        cos_w + batch_idx * cos_w_stride_b + token_idx * cos_w_stride_s + w_local * cos_w_stride_d,
        mask=w_mask,
        other=0.0,
    )
    sin_w_vals = tl.load(
        sin_w + batch_idx * cos_w_stride_b + token_idx * cos_w_stride_s + w_local * cos_w_stride_d,
        mask=w_mask,
        other=0.0,
    )
    cos_vals = cos_t_vals + cos_h_vals + cos_w_vals
    sin_vals = sin_t_vals + sin_h_vals + sin_w_vals

    sign = tl.where(
        (offs < head_dim // 4)
        | ((offs >= head_dim // 2) & (offs < head_dim // 8 * 5))
        | ((offs >= head_dim // 4 * 3) & (offs < head_dim // 8 * 7)),
        -1.0,
        1.0,
    )
    out = normed * cos_vals + sign * pair_normed * sin_vals

    out_base = tl.where(
        is_q,
        query + batch_idx * query_stride_b + head_idx * query_stride_h + token_idx * query_stride_s,
        key + batch_idx * key_stride_b + head_idx * key_stride_h + token_idx * key_stride_s,
    )
    tl.store(out_base + offs * tl.where(is_q, query_stride_d, key_stride_d), out)
    if write_value:
        if not is_q:
            src = value + batch_idx * value_stride_b + token_idx * value_stride_s + head_idx * value_stride_h
            dst = (
                value_output
                + batch_idx * value_out_stride_b
                + head_idx * value_out_stride_h
                + token_idx * value_out_stride_s
            )
            tl.store(dst + offs * value_out_stride_d, tl.load(src + offs * value_stride_d))


def triton_qk_norm_rope(
    q,
    k,
    q_norm_weight,
    k_norm_weight,
    q_norm_hw_weight,
    k_norm_hw_weight,
    cos_t,
    sin_t,
    cos_h,
    sin_h,
    cos_w,
    sin_w,
    eps,
    *,
    key_output=None,
    value=None,
    value_output=None,
):
    batch_size, seq_len, head_q, head_dim = q.shape
    head_k = k.shape[2]

    query = torch.empty((batch_size, seq_len, head_q, head_dim), device=q.device, dtype=q.dtype).transpose(1, 2)
    key = key_output
    if key is not None and key.shape != (batch_size, head_k, seq_len, head_dim):
        raise ValueError("key_output must match the projected K shape")
    if value_output is not None and (value is None or value_output.shape != (batch_size, head_k, seq_len, head_dim)):
        raise ValueError("value_output requires V and must match its projected shape")
    if key is None:
        key = torch.empty((batch_size, seq_len, head_k, head_dim), device=k.device, dtype=k.dtype).transpose(1, 2)
    grid = (
        seq_len,
        head_q + head_k,
        batch_size,
    )

    qk_norm_rope_kernel[grid](
        q,
        k,
        query,
        key,
        value,
        value_output,
        q_norm_weight,
        k_norm_weight,
        q_norm_hw_weight,
        k_norm_hw_weight,
        cos_t,
        sin_t,
        cos_h,
        sin_h,
        cos_w,
        sin_w,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        key.stride(3),
        cos_t.stride(0),
        cos_t.stride(1),
        cos_t.stride(2),
        cos_h.stride(0),
        cos_h.stride(1),
        cos_h.stride(2),
        cos_w.stride(0),
        cos_w.stride(1),
        cos_w.stride(2),
        *(value.stride() if value is not None else (0, 0, 0, 0)),
        *(value_output.stride() if value_output is not None else (0, 0, 0, 0)),
        write_value=value_output is not None,
        head_q=head_q,
        head_dim=head_dim,
        eps=eps,
    )
    return query, key
