# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Q/K RMSNorm followed by packed RoPE (non-interleaved or interleaved).

The public contract is shared by diffusion attention implementations:

* ``q`` and ``k`` are ``[tokens, heads, head_dim]``;
* norm weights are one-dimensional ``[head_dim]`` tensors;
* ``rope_table`` is ``[tokens, rotary_dim]`` and stores
  ``[cos(theta), sin(theta)]`` with ``theta`` of width ``rotary_dim // 2``;
  its dtype is either the activation dtype or float32;
* ``interleaved=False`` rotates half-split pairs ``(d, d + rotary_dim/2)``
  (MiniMax-H3); ``interleaved=True`` rotates adjacent pairs ``(2i, 2i + 1)``
  with ``theta_i`` (Boogu-Image, matching its ``apply_rotary_emb``).

``fused_joint_qkv_norm_rope`` is the dual-stream variant: two ``[B, S_i, H, D]``
streams with their own norm weights (text / image in Flux.2) are normalized,
rotated and written — together with their V rows — straight into the joint
``[B, S_0 + S_1, H, D]`` Q/K/V that attention consumes, replacing the block's
four norms, three ``cat`` copies and two RoPE passes with one launch.

The CUDA fast path fuses RMSNorm and RoPE without materializing normalized Q/K
or rotary-product intermediates. The interleaved mode supports any even
``rotary_dim <= head_dim <= 256`` (``tl.arange`` padding to the next power of
two); the half-split mode keeps the pre-existing kernel and its MiniMax-H3
geometry contract (``head_dim == 128``, ``rotary_dim == 96``). Ascend
composes its RMSNorm and rotary fused primitives on the MiniMax-H3 geometry;
unsupported inputs use the eager reference.
"""

from __future__ import annotations

from importlib.util import find_spec

import torch
import torch.nn.functional as F
from torch.library import Library
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_omni.diffusion import envs
from vllm_omni.platforms import current_omni_platform

_FUSED_HEAD_DIM = 128
_FUSED_ROTARY_DIM = 96
# Interleaved-mode geometry bounds (any even rotary_dim <= head_dim <= this).
_FUSED_MAX_HEAD_DIM = 256
_HEADS_PER_PROGRAM = 8
# The combined interleaved kernel: small tiles + one warp measured fastest
# (latency-bound shapes; see the kernel docstring).
_INTERLEAVED_HEADS_PER_PROGRAM = 4


def _apply_rope_table(
    x: torch.Tensor,
    rope_table: torch.Tensor,
    rotary_dim: int,
    *,
    interleaved: bool = False,
    dtype: torch.dtype | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Eager RoPE application: the fallback and the unit-test reference.

    Shapes: ``x`` is the normalized ``[tokens, heads, head_dim]`` tensor;
    ``rope_table`` is ``[tokens, rotary_dim]`` storing
    ``[cos(theta) | sin(theta)]`` with ``theta`` of width ``rotary_dim // 2``.

    ``interleaved`` selects the pairing: half-split ``(d, d + rotary_dim/2)``
    sharing ``theta_d`` (MiniMax-H3) or adjacent pairs ``(2i, 2i + 1)``
    sharing ``theta_i`` (Boogu-Image). ``dtype`` is the precision the
    rotation arithmetic runs in and ``output_dtype`` the precision the result
    is cast to; both default to ``x``'s dtype (the historical half-split
    behaviour). The interleaved caller passes ``dtype=torch.float32`` to
    match the Triton kernel's composition — fp32 rotation of the
    bf16-rounded normalized value, one final rounding — which is also how
    vLLM's own ``fused_qk_norm_rope`` Triton kernel and Boogu's eager
    ``apply_rotary_emb`` order their rounding.
    """
    compute_dtype = x.dtype if dtype is None else dtype
    out_dtype = x.dtype if output_dtype is None else output_dtype
    half = rotary_dim // 2
    x_c = x.to(compute_dtype)
    cos = rope_table[..., :half].to(compute_dtype).unsqueeze(1)
    sin = rope_table[..., half:].to(compute_dtype).unsqueeze(1)
    first = x_c[..., 0:rotary_dim:2] if interleaved else x_c[..., :half]
    second = x_c[..., 1:rotary_dim:2] if interleaved else x_c[..., half:rotary_dim]
    rotated_first = first * cos - second * sin
    rotated_second = second * cos + first * sin
    if interleaved:
        # stack(..., dim=-1) pairs (new_even_i, new_odd_i) into
        # [..., half, 2]; flatten(-2) lays the pairs back out as
        # [e0, o0, e1, o1, ...] — back to their interleaved positions.
        rotated = torch.stack(
            (rotated_first, rotated_second),
            dim=-1,
        ).flatten(-2)
        return torch.cat((rotated, x_c[..., rotary_dim:]), dim=-1).to(out_dtype)
    else:
        return torch.cat(
            (
                rotated_first,
                rotated_second,
                x_c[..., rotary_dim:],
            ),
            dim=-1,
        ).to(out_dtype)


if HAS_TRITON:

    @triton.jit
    def _rms_norm_rope_kernel(
        x_ptr,
        weight_ptr,
        rope_table_ptr,
        out_ptr,
        x_stride_t,
        x_stride_h,
        x_stride_d,
        rope_stride_t,
        out_stride_t,
        out_stride_h,
        out_stride_d,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        rotary_half: tl.constexpr,
        eps: tl.constexpr,
        heads_per_program: tl.constexpr,
    ):
        token = tl.program_id(0)
        head_group = tl.program_id(1)
        heads = head_group * heads_per_program + tl.arange(0, heads_per_program)
        dims = tl.arange(0, head_dim)
        mask = heads[:, None] < num_heads
        offsets = token * x_stride_t + heads[:, None] * x_stride_h + dims[None, :] * x_stride_d

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + dims).to(tl.float32)
        inv_rms = tl.rsqrt(tl.sum(x * x, axis=1) / head_dim + eps)
        normalized = (x * inv_rms[:, None] * weight[None, :]).to(tl.bfloat16)

        rotary_dim = rotary_half * 2
        pair_dims = tl.where(
            dims < rotary_half,
            dims + rotary_half,
            tl.where(dims < rotary_dim, dims - rotary_half, dims),
        )
        pair_offsets = token * x_stride_t + heads[:, None] * x_stride_h + pair_dims[None, :] * x_stride_d
        pair_x = tl.load(x_ptr + pair_offsets, mask=mask, other=0.0).to(tl.float32)
        pair_weight = tl.load(weight_ptr + pair_dims).to(tl.float32)
        pair_normalized = (pair_x * inv_rms[:, None] * pair_weight[None, :]).to(tl.bfloat16)

        freq_dims = tl.where(
            dims < rotary_half,
            dims,
            tl.where(dims < rotary_dim, dims - rotary_half, 0),
        )
        table_offsets = token * rope_stride_t + freq_dims
        cos = tl.load(rope_table_ptr + table_offsets).to(tl.float32)
        sin = tl.load(rope_table_ptr + table_offsets + rotary_half).to(tl.float32)
        first = normalized.to(tl.float32) * cos - pair_normalized.to(tl.float32) * sin
        second = normalized.to(tl.float32) * cos + pair_normalized.to(tl.float32) * sin
        output = tl.where(
            dims < rotary_dim,
            tl.where(dims < rotary_half, first, second),
            normalized.to(tl.float32),
        )

        out_offsets = token * out_stride_t + heads[:, None] * out_stride_h + dims[None, :] * out_stride_d
        tl.store(out_ptr + out_offsets, output, mask=mask)

    @triton.jit
    def _qk_norm_rope_kernel(
        q_ptr,
        k_ptr,
        q_weight_ptr,
        k_weight_ptr,
        rope_table_ptr,
        out_q_ptr,
        out_k_ptr,
        q_stride_t,
        q_stride_h,
        q_stride_d,
        k_stride_t,
        k_stride_h,
        k_stride_d,
        rope_stride_t,
        out_q_stride_t,
        out_q_stride_h,
        out_q_stride_d,
        out_k_stride_t,
        out_k_stride_h,
        out_k_stride_d,
        num_q_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        padded_dim: tl.constexpr,  # next power of 2 >= head_dim (tl.arange needs a power of 2)
        rotary_half: tl.constexpr,
        eps: tl.constexpr,
        heads_per_program: tl.constexpr,
        q_head_groups: tl.constexpr,
        interleaved: tl.constexpr,
    ):
        """Fused per-head RMSNorm + RoPE for Q and K in one launch.

        Grid axis 1 assigns the first ``q_head_groups`` programs of a token
        to Q and the rest to K, so the small K grid hides inside Q's waves
        instead of paying its own latency-bound launch. ``interleaved``
        selects the pairing, mirroring ``_apply_rope_table``:

        * ``True`` — adjacent pairs ``(2i, 2i+1)`` sharing ``theta_i``.
        * ``False`` — half-split pairs ``(d, d + rotary_half)`` sharing
          ``theta_d``. Production traffic still routes half-split inputs
          through ``_rms_norm_rope_kernel``; this mode is test-covered and
          awaits the maintainers' call before taking over that path.

        The modes differ only in the pair/frequency/selector index
        expressions; the arithmetic is shared. Beyond the rotary width
        (partial rotary or lane padding) the output is the normalized value
        unchanged.
        """
        token = tl.program_id(0)
        head_group = tl.program_id(1)
        dims = tl.arange(0, padded_dim)
        dim_mask = dims < head_dim
        if head_group < q_head_groups:
            heads = head_group * heads_per_program + tl.arange(0, heads_per_program)
            # Valid-domain mask: rows beyond the head count (when it is not a
            # multiple of heads_per_program) and padded lanes beyond head_dim.
            mask = (heads[:, None] < num_q_heads) & dim_mask[None, :]
            in_ptr = q_ptr
            weight_ptr = q_weight_ptr
            out_ptr = out_q_ptr
            in_stride_t, in_stride_h, in_stride_d = q_stride_t, q_stride_h, q_stride_d
            out_stride_t, out_stride_h, out_stride_d = (
                out_q_stride_t,
                out_q_stride_h,
                out_q_stride_d,
            )
        else:
            heads = (head_group - q_head_groups) * heads_per_program + tl.arange(0, heads_per_program)
            mask = (heads[:, None] < num_kv_heads) & dim_mask[None, :]
            in_ptr = k_ptr
            weight_ptr = k_weight_ptr
            out_ptr = out_k_ptr
            in_stride_t, in_stride_h, in_stride_d = k_stride_t, k_stride_h, k_stride_d
            out_stride_t, out_stride_h, out_stride_d = (
                out_k_stride_t,
                out_k_stride_h,
                out_k_stride_d,
            )

        offsets = token * in_stride_t + heads[:, None] * in_stride_h + dims[None, :] * in_stride_d
        x = tl.load(in_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + dims, mask=dim_mask, other=0.0).to(tl.float32)
        # Padding lanes load 0 and do not perturb the sum; divide by the true
        # head_dim, not the padded width.
        inv_rms = tl.rsqrt(tl.sum(x * x, axis=1) / head_dim + eps)
        normalized = (x * inv_rms[:, None] * weight[None, :]).to(tl.bfloat16)

        # The two pairings differ only in three index expressions; the pair
        # reload, the table gather, the rotation formulas and the write-back
        # are shared (the same structure as _apply_rope_table).
        rotary_dim = rotary_half * 2
        if interleaved:
            # Adjacent pairs (2i, 2i+1) sharing theta_i.
            pair_dims = tl.where(dims < rotary_dim, dims ^ 1, dims)
            freq_dims = tl.where(dims < rotary_dim, dims // 2, 0)
            is_first = (dims % 2) == 0
        else:
            # Half-split pairs (d, d + rotary_half) sharing theta_d.
            pair_dims = tl.where(
                dims < rotary_half,
                dims + rotary_half,
                tl.where(dims < rotary_dim, dims - rotary_half, dims),
            )
            freq_dims = tl.where(
                dims < rotary_half,
                dims,
                tl.where(dims < rotary_dim, dims - rotary_half, 0),
            )
            is_first = dims < rotary_half
        pair_offsets = token * in_stride_t + heads[:, None] * in_stride_h + pair_dims[None, :] * in_stride_d
        pair_x = tl.load(in_ptr + pair_offsets, mask=mask, other=0.0).to(tl.float32)
        pair_weight = tl.load(weight_ptr + pair_dims, mask=pair_dims < head_dim, other=0.0).to(tl.float32)
        pair_normalized = (pair_x * inv_rms[:, None] * pair_weight[None, :]).to(tl.bfloat16)
        table_offsets = token * rope_stride_t + freq_dims
        cos = tl.load(rope_table_ptr + table_offsets).to(tl.float32)
        sin = tl.load(rope_table_ptr + table_offsets + rotary_half).to(tl.float32)
        first = normalized.to(tl.float32) * cos - pair_normalized.to(tl.float32) * sin
        second = normalized.to(tl.float32) * cos + pair_normalized.to(tl.float32) * sin
        output = tl.where(
            dims < rotary_dim,
            tl.where(is_first, first, second),
            normalized.to(tl.float32),
        )

        out_offsets = token * out_stride_t + heads[:, None] * out_stride_h + dims[None, :] * out_stride_d
        tl.store(out_ptr + out_offsets, output, mask=mask)

    @triton.jit
    def _norm_rope_tile(
        in_ptr,
        weight_ptr,
        rope_table_ptr,
        out_ptr,
        in_row,
        out_row,
        in_stride_t,
        in_stride_h,
        in_stride_d,
        rope_stride_t,
        out_stride_t,
        out_stride_h,
        out_stride_d,
        heads,
        mask,
        dims,
        dim_mask,
        head_dim: tl.constexpr,
        rotary_half: tl.constexpr,
        eps: tl.constexpr,
        interleaved: tl.constexpr,
    ):
        """One ``[heads_per_program, padded_dim]`` RMSNorm + RoPE tile.

        The body of ``_qk_norm_rope_kernel`` after its pointer selection,
        arithmetic unchanged, with the input row (``in_row``, into the
        stream's own projection) and the output/table row (``out_row``, into
        the joint sequence) passed separately so the joint kernel can gather
        from two streams.
        """
        offsets = in_row * in_stride_t + heads[:, None] * in_stride_h + dims[None, :] * in_stride_d
        x = tl.load(in_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + dims, mask=dim_mask, other=0.0).to(tl.float32)
        inv_rms = tl.rsqrt(tl.sum(x * x, axis=1) / head_dim + eps)
        normalized = (x * inv_rms[:, None] * weight[None, :]).to(tl.bfloat16)

        rotary_dim = rotary_half * 2
        if interleaved:
            pair_dims = tl.where(dims < rotary_dim, dims ^ 1, dims)
            freq_dims = tl.where(dims < rotary_dim, dims // 2, 0)
            is_first = (dims % 2) == 0
        else:
            pair_dims = tl.where(
                dims < rotary_half,
                dims + rotary_half,
                tl.where(dims < rotary_dim, dims - rotary_half, dims),
            )
            freq_dims = tl.where(
                dims < rotary_half,
                dims,
                tl.where(dims < rotary_dim, dims - rotary_half, 0),
            )
            is_first = dims < rotary_half
        pair_offsets = in_row * in_stride_t + heads[:, None] * in_stride_h + pair_dims[None, :] * in_stride_d
        pair_x = tl.load(in_ptr + pair_offsets, mask=mask, other=0.0).to(tl.float32)
        pair_weight = tl.load(weight_ptr + pair_dims, mask=pair_dims < head_dim, other=0.0).to(tl.float32)
        pair_normalized = (pair_x * inv_rms[:, None] * pair_weight[None, :]).to(tl.bfloat16)
        table_offsets = out_row * rope_stride_t + freq_dims
        cos = tl.load(rope_table_ptr + table_offsets).to(tl.float32)
        sin = tl.load(rope_table_ptr + table_offsets + rotary_half).to(tl.float32)
        first = normalized.to(tl.float32) * cos - pair_normalized.to(tl.float32) * sin
        second = normalized.to(tl.float32) * cos + pair_normalized.to(tl.float32) * sin
        output = tl.where(
            dims < rotary_dim,
            tl.where(is_first, first, second),
            normalized.to(tl.float32),
        )
        out_offsets = out_row * out_stride_t + heads[:, None] * out_stride_h + dims[None, :] * out_stride_d
        tl.store(out_ptr + out_offsets, output, mask=mask)

    @triton.jit
    def _joint_qkv_norm_rope_kernel(
        q0_ptr,
        k0_ptr,
        v0_ptr,
        q1_ptr,
        k1_ptr,
        v1_ptr,
        q0_weight_ptr,
        k0_weight_ptr,
        q1_weight_ptr,
        k1_weight_ptr,
        rope_table_ptr,
        out_q_ptr,
        out_k_ptr,
        out_v_ptr,
        q0_stride_t,
        q0_stride_h,
        q0_stride_d,
        k0_stride_t,
        k0_stride_h,
        k0_stride_d,
        v0_stride_t,
        v0_stride_h,
        v0_stride_d,
        q1_stride_t,
        q1_stride_h,
        q1_stride_d,
        k1_stride_t,
        k1_stride_h,
        k1_stride_d,
        v1_stride_t,
        v1_stride_h,
        v1_stride_d,
        rope_stride_t,
        out_q_stride_t,
        out_q_stride_h,
        out_q_stride_d,
        out_k_stride_t,
        out_k_stride_h,
        out_k_stride_d,
        out_v_stride_t,
        out_v_stride_h,
        out_v_stride_d,
        seq_len_0,
        seq_len_1,
        num_q_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        padded_dim: tl.constexpr,
        rotary_half: tl.constexpr,
        eps: tl.constexpr,
        heads_per_program: tl.constexpr,
        q_head_groups: tl.constexpr,
        kv_head_groups: tl.constexpr,
        interleaved: tl.constexpr,
    ):
        """Two-stream Q/K RMSNorm + RoPE and V gather, written in attention's
        joint input layout.

        Dual-stream DiT blocks (Flux.2, SD3, ...) normalize the text and the
        image Q/K with separate weights, concatenate the two streams along the
        sequence, apply RoPE and hand ``[B, S_0 + S_1, H, D]`` Q/K/V to
        attention. This kernel reads each stream straight from its own
        ``[B * S_i, H, D]`` projection view and writes the normalized, rotated
        Q/K rows and the V rows directly into the joint ``[B * (S_0 + S_1),
        H, D]`` outputs (stream 0 first) — the tensors attention consumes —
        so neither the normalized tensors nor the three ``cat`` copies are
        materialized.

        Grid axis 0 enumerates joint output rows; axis 1 assigns Q, then K,
        then V head groups (V programs only gather). Q/K use
        ``_norm_rope_tile`` — ``_qk_norm_rope_kernel``'s arithmetic unchanged;
        the row selects its stream's pointer, weights and strides.
        ``rope_table`` is indexed by the joint output row.
        """
        token = tl.program_id(0)
        head_group = tl.program_id(1)
        seq_total = seq_len_0 + seq_len_1
        batch = token // seq_total
        pos = token % seq_total
        in_stream_1 = pos >= seq_len_0
        if in_stream_1:
            in_row = batch * seq_len_1 + (pos - seq_len_0)
        else:
            in_row = batch * seq_len_0 + pos

        dims = tl.arange(0, padded_dim)
        dim_mask = dims < head_dim
        is_qk = head_group < q_head_groups + kv_head_groups
        if head_group < q_head_groups:
            heads = head_group * heads_per_program + tl.arange(0, heads_per_program)
            mask = (heads[:, None] < num_q_heads) & dim_mask[None, :]
            out_ptr = out_q_ptr
            out_stride_t, out_stride_h, out_stride_d = out_q_stride_t, out_q_stride_h, out_q_stride_d
            if in_stream_1:
                in_ptr, weight_ptr = q1_ptr, q1_weight_ptr
                in_stride_t, in_stride_h, in_stride_d = q1_stride_t, q1_stride_h, q1_stride_d
            else:
                in_ptr, weight_ptr = q0_ptr, q0_weight_ptr
                in_stride_t, in_stride_h, in_stride_d = q0_stride_t, q0_stride_h, q0_stride_d
        elif is_qk:
            heads = (head_group - q_head_groups) * heads_per_program + tl.arange(0, heads_per_program)
            mask = (heads[:, None] < num_kv_heads) & dim_mask[None, :]
            out_ptr = out_k_ptr
            out_stride_t, out_stride_h, out_stride_d = out_k_stride_t, out_k_stride_h, out_k_stride_d
            if in_stream_1:
                in_ptr, weight_ptr = k1_ptr, k1_weight_ptr
                in_stride_t, in_stride_h, in_stride_d = k1_stride_t, k1_stride_h, k1_stride_d
            else:
                in_ptr, weight_ptr = k0_ptr, k0_weight_ptr
                in_stride_t, in_stride_h, in_stride_d = k0_stride_t, k0_stride_h, k0_stride_d
        else:
            heads = (head_group - q_head_groups - kv_head_groups) * heads_per_program + tl.arange(0, heads_per_program)
            mask = (heads[:, None] < num_kv_heads) & dim_mask[None, :]
            out_ptr = out_v_ptr
            out_stride_t, out_stride_h, out_stride_d = out_v_stride_t, out_v_stride_h, out_v_stride_d
            # V is gathered, not normalized: the weight pointer is unused.
            if in_stream_1:
                in_ptr, weight_ptr = v1_ptr, q0_weight_ptr
                in_stride_t, in_stride_h, in_stride_d = v1_stride_t, v1_stride_h, v1_stride_d
            else:
                in_ptr, weight_ptr = v0_ptr, q0_weight_ptr
                in_stride_t, in_stride_h, in_stride_d = v0_stride_t, v0_stride_h, v0_stride_d

        if is_qk:
            _norm_rope_tile(
                in_ptr,
                weight_ptr,
                rope_table_ptr,
                out_ptr,
                in_row,
                token,
                in_stride_t,
                in_stride_h,
                in_stride_d,
                rope_stride_t,
                out_stride_t,
                out_stride_h,
                out_stride_d,
                heads,
                mask,
                dims,
                dim_mask,
                head_dim,
                rotary_half,
                eps,
                interleaved,
            )
        else:
            offsets = in_row * in_stride_t + heads[:, None] * in_stride_h + dims[None, :] * in_stride_d
            rows = tl.load(in_ptr + offsets, mask=mask, other=0.0)
            out_offsets = token * out_stride_t + heads[:, None] * out_stride_h + dims[None, :] * out_stride_d
            tl.store(out_ptr + out_offsets, rows, mask=mask)


def _eager_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    head_dim: int,
    rotary_dim: int,
    interleaved: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_norm = F.rms_norm(q, (head_dim,), q_weight, eps)
    k_norm = F.rms_norm(k, (head_dim,), k_weight, eps)
    # fp32 rotation for the interleaved (Boogu) semantics; the half-split
    # (H3) reference keeps its historical x-dtype arithmetic via the defaults.
    rope_dtype = torch.float32 if interleaved else None
    return (
        _apply_rope_table(q_norm, rope_table, rotary_dim, interleaved=interleaved, dtype=rope_dtype),
        _apply_rope_table(k_norm, rope_table, rotary_dim, interleaved=interleaved, dtype=rope_dtype),
    )


def _npu_apply_rope_table(
    x: torch.Tensor,
    rope_table: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    """Apply H3's rotary dimensions through Ascend's fused rotary kernel.

    ``mindiesd.rotary_position_embedding`` takes a 4-D BSND tensor, whereas
    MiniMax-H3 keeps packed activations as ``[tokens, heads, head_dim]``.
    The shared MindIE-SD wrapper normalizes this 3-D layout to BSND and
    restores it afterwards. It receives the half-width cos/sin values from
    the packed table; the wrapper expands them to H3's non-interleaved 96-D
    rotary layout. The 32 non-rotary head dimensions bypass the kernel.

    Some CANN environments package ``torch_npu`` without MindIE-SD. Keep
    those deployments on the same Ascend fused rotary primitive rather than
    silently falling back to eager elementwise RoPE.
    """
    import torch_npu

    half = rotary_dim // 2
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    cos = rope_table[..., :half]
    sin = rope_table[..., half:]

    if find_spec("mindiesd") is not None:
        from vllm_omni.diffusion.layers.rope import apply_rotary_emb_mindiesd

        x_rot = apply_rotary_emb_mindiesd(
            x_rot,
            cos,
            sin,
            interleaved=False,
            half_head_dim=True,
        )
    else:
        # npu_rotary_mul uses BSND and full rotary-width cos/sin. H3 uses
        # NeoX/rotated-half ordering, so duplicate rather than interleave the
        # half-width table along the final dimension.
        cos = cos.unsqueeze(0).unsqueeze(2).repeat(1, 1, 1, 2)
        sin = sin.unsqueeze(0).unsqueeze(2).repeat(1, 1, 1, 2)
        x_rot = torch_npu.npu_rotary_mul(x_rot.unsqueeze(0), cos, sin).squeeze(0)

    return torch.cat((x_rot, x_pass), dim=-1)


def _npu_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use Ascend RMSNorm and RoPE fused primitives for MiniMax-H3 DiT."""
    import torch_npu

    q_norm = torch_npu.npu_rms_norm(q, q_weight, epsilon=eps)[0]
    k_norm = torch_npu.npu_rms_norm(k, k_weight, epsilon=eps)[0]
    return (
        _npu_apply_rope_table(q_norm, rope_table, rotary_dim),
        _npu_apply_rope_table(k_norm, rope_table, rotary_dim),
    )


def _fused_cuda_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    head_dim: int,
    rotary_dim: int,
    interleaved: bool = False,
) -> bool:
    if not (
        HAS_TRITON
        and current_platform.is_cuda()
        and q.is_cuda
        and k.is_cuda
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
    ):
        return False
    if interleaved:
        return rotary_dim % 2 == 0 and 2 <= rotary_dim <= head_dim <= _FUSED_MAX_HEAD_DIM
    # Half-split traffic keeps the pre-existing per-tensor kernel and its
    # exact MiniMax-H3 geometry contract (that kernel is untouched by the
    # interleaved extension).
    return head_dim == _FUSED_HEAD_DIM and rotary_dim == _FUSED_ROTARY_DIM


def _fused_npu_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    head_dim: int,
    rotary_dim: int,
    interleaved: bool = False,
) -> bool:
    """Return whether the MiniMax-H3 Ascend fused-op contract is satisfied."""
    return (
        not interleaved
        and current_omni_platform.is_npu()
        and q.device.type == "npu"
        and k.device.type == "npu"
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and head_dim == _FUSED_HEAD_DIM
        and rotary_dim == _FUSED_ROTARY_DIM
    )


def _launch_fused_rms_norm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    tokens, heads, head_dim = x.shape
    rotary_half = rope_table.shape[-1] // 2
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if tokens == 0:
        return out
    grid = (tokens, triton.cdiv(heads, _HEADS_PER_PROGRAM))
    _rms_norm_rope_kernel[grid](
        x,
        weight,
        rope_table,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        rope_table.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_heads=heads,
        head_dim=head_dim,
        rotary_half=rotary_half,
        eps=eps,
        heads_per_program=_HEADS_PER_PROGRAM,
        num_warps=8,
    )
    return out


def _launch_fused_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, q_heads, head_dim = q.shape
    kv_heads = k.shape[1]
    rotary_half = rope_table.shape[-1] // 2
    out_q = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    out_k = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    if tokens == 0:
        return out_q, out_k
    if interleaved:
        heads_per_program = _INTERLEAVED_HEADS_PER_PROGRAM
        num_warps, num_stages = 1, 2
    else:
        # Match the historical per-tensor launch shape (8 heads per program,
        # 8 warps) so the half-split arithmetic keeps its reduction tree.
        heads_per_program = _HEADS_PER_PROGRAM
        num_warps, num_stages = 8, 3
    q_head_groups = triton.cdiv(q_heads, heads_per_program)
    kv_head_groups = triton.cdiv(kv_heads, heads_per_program)
    grid = (tokens, q_head_groups + kv_head_groups)
    _qk_norm_rope_kernel[grid](
        q,
        k,
        q_weight,
        k_weight,
        rope_table,
        out_q,
        out_k,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        rope_table.stride(0),
        out_q.stride(0),
        out_q.stride(1),
        out_q.stride(2),
        out_k.stride(0),
        out_k.stride(1),
        out_k.stride(2),
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        padded_dim=triton.next_power_of_2(head_dim),
        rotary_half=rotary_half,
        eps=eps,
        heads_per_program=heads_per_program,
        q_head_groups=q_head_groups,
        interleaved=interleaved,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out_q, out_k


def _fused_qk_norm_rope_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    head_dim: int,
    rotary_dim: int,
    interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _fused_npu_supported(q, k, head_dim, rotary_dim, interleaved):
        return _npu_qk_norm_rope(
            q,
            k,
            q_weight,
            k_weight,
            rope_table,
            eps,
            rotary_dim,
        )
    if not _fused_cuda_supported(q, k, head_dim, rotary_dim, interleaved):
        return _eager_qk_norm_rope(
            q,
            k,
            q_weight,
            k_weight,
            rope_table,
            eps,
            head_dim,
            rotary_dim,
            interleaved,
        )
    if interleaved:
        return _launch_fused_qk_norm_rope(q, k, q_weight, k_weight, rope_table, eps, interleaved=True)
    # Half-split production traffic keeps the historical per-tensor kernel;
    # _qk_norm_rope_kernel's half-split mode awaits the maintainers' call.
    return (
        _launch_fused_rms_norm_rope(q, q_weight, rope_table, eps),
        _launch_fused_rms_norm_rope(k, k_weight, rope_table, eps),
    )


def _fused_qk_norm_rope_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    head_dim: int,
    rotary_dim: int,
    interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    del q_weight, k_weight, rope_table, eps, head_dim, rotary_dim, interleaved
    return torch.empty_like(q), torch.empty_like(k)


_OMNI_OP_LIB = Library("vllm_omni", "FRAGMENT")
if not hasattr(torch.ops.vllm_omni, "fused_qk_norm_rope"):
    direct_register_custom_op(
        op_name="fused_qk_norm_rope",
        op_func=_fused_qk_norm_rope_impl,
        fake_impl=_fused_qk_norm_rope_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )


def _launch_fused_joint_qkv_norm_rope(
    q0: torch.Tensor,
    k0: torch.Tensor,
    v0: torch.Tensor,
    q1: torch.Tensor,
    k1: torch.Tensor,
    v1: torch.Tensor,
    q0_weight: torch.Tensor,
    k0_weight: torch.Tensor,
    q1_weight: torch.Tensor,
    k1_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the two-stream kernel; ``q_i``/``k_i``/``v_i`` are ``[B, S_i, H, D]``."""
    batch, seq_len_0, q_heads, head_dim = q0.shape
    seq_len_1 = q1.shape[1]
    kv_heads = k0.shape[2]
    seq_total = seq_len_0 + seq_len_1
    rotary_half = rope_table.shape[-1] // 2
    out_q = torch.empty((batch, seq_total, q_heads, head_dim), dtype=q0.dtype, device=q0.device)
    out_k = torch.empty((batch, seq_total, kv_heads, head_dim), dtype=k0.dtype, device=k0.device)
    out_v = torch.empty((batch, seq_total, kv_heads, head_dim), dtype=v0.dtype, device=v0.device)
    tokens = batch * seq_total
    if tokens == 0:
        return out_q, out_k, out_v
    # Each stream's [B, S_i, H, D] projection view merges (B, S_i) without a
    # copy when the batch stride is S_i times the sequence stride, which holds
    # for chunked-QKV projections; reshape copies otherwise.
    q0_v, k0_v, v0_v = (t.reshape(-1, t.shape[2], head_dim) for t in (q0, k0, v0))
    q1_v, k1_v, v1_v = (t.reshape(-1, t.shape[2], head_dim) for t in (q1, k1, v1))
    out_q_v = out_q.view(tokens, q_heads, head_dim)
    out_k_v = out_k.view(tokens, kv_heads, head_dim)
    out_v_v = out_v.view(tokens, kv_heads, head_dim)
    if interleaved:
        heads_per_program = _INTERLEAVED_HEADS_PER_PROGRAM
        num_warps, num_stages = 1, 2
    else:
        heads_per_program = _HEADS_PER_PROGRAM
        num_warps, num_stages = 8, 3
    q_head_groups = triton.cdiv(q_heads, heads_per_program)
    kv_head_groups = triton.cdiv(kv_heads, heads_per_program)
    grid = (tokens, q_head_groups + 2 * kv_head_groups)
    _joint_qkv_norm_rope_kernel[grid](
        q0_v,
        k0_v,
        v0_v,
        q1_v,
        k1_v,
        v1_v,
        q0_weight,
        k0_weight,
        q1_weight,
        k1_weight,
        rope_table,
        out_q_v,
        out_k_v,
        out_v_v,
        *q0_v.stride(),
        *k0_v.stride(),
        *v0_v.stride(),
        *q1_v.stride(),
        *k1_v.stride(),
        *v1_v.stride(),
        rope_table.stride(0),
        *out_q_v.stride(),
        *out_k_v.stride(),
        *out_v_v.stride(),
        seq_len_0,
        seq_len_1,
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        padded_dim=triton.next_power_of_2(head_dim),
        rotary_half=rotary_half,
        eps=eps,
        heads_per_program=heads_per_program,
        q_head_groups=q_head_groups,
        kv_head_groups=kv_head_groups,
        interleaved=interleaved,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out_q, out_k, out_v


def _eager_joint_qkv_norm_rope(
    q0: torch.Tensor,
    k0: torch.Tensor,
    v0: torch.Tensor,
    q1: torch.Tensor,
    k1: torch.Tensor,
    v1: torch.Tensor,
    q0_weight: torch.Tensor,
    k0_weight: torch.Tensor,
    q1_weight: torch.Tensor,
    k1_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    head_dim: int,
    rotary_dim: int,
    interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference: per-stream RMSNorm, concatenate along the sequence, RoPE."""
    q = torch.cat((F.rms_norm(q0, (head_dim,), q0_weight, eps), F.rms_norm(q1, (head_dim,), q1_weight, eps)), dim=1)
    k = torch.cat((F.rms_norm(k0, (head_dim,), k0_weight, eps), F.rms_norm(k1, (head_dim,), k1_weight, eps)), dim=1)
    v = torch.cat((v0, v1), dim=1)
    batch, seq_total, q_heads, _ = q.shape
    kv_heads = k.shape[2]
    rope_dtype = torch.float32 if interleaved else None
    q = _apply_rope_table(
        q.reshape(-1, q_heads, head_dim), rope_table, rotary_dim, interleaved=interleaved, dtype=rope_dtype
    )
    k = _apply_rope_table(
        k.reshape(-1, kv_heads, head_dim), rope_table, rotary_dim, interleaved=interleaved, dtype=rope_dtype
    )
    return q.view(batch, seq_total, q_heads, head_dim), k.view(batch, seq_total, kv_heads, head_dim), v


def _fused_joint_qkv_norm_rope_impl(
    q0: torch.Tensor,
    k0: torch.Tensor,
    v0: torch.Tensor,
    q1: torch.Tensor,
    k1: torch.Tensor,
    v1: torch.Tensor,
    q0_weight: torch.Tensor,
    k0_weight: torch.Tensor,
    q1_weight: torch.Tensor,
    k1_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    head_dim: int,
    rotary_dim: int,
    interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not _fused_cuda_supported(q0, k0, head_dim, rotary_dim, interleaved) or not interleaved:
        # The half-split geometry gate above is the MiniMax-H3 contract of the
        # per-tensor kernel; the joint kernel takes any even rotary width, but
        # its half-split mode is test-covered only, like the single-stream one.
        return _eager_joint_qkv_norm_rope(
            q0,
            k0,
            v0,
            q1,
            k1,
            v1,
            q0_weight,
            k0_weight,
            q1_weight,
            k1_weight,
            rope_table,
            eps,
            head_dim,
            rotary_dim,
            interleaved,
        )
    return _launch_fused_joint_qkv_norm_rope(
        q0, k0, v0, q1, k1, v1, q0_weight, k0_weight, q1_weight, k1_weight, rope_table, eps, interleaved=True
    )


def _fused_joint_qkv_norm_rope_fake(
    q0: torch.Tensor,
    k0: torch.Tensor,
    v0: torch.Tensor,
    q1: torch.Tensor,
    k1: torch.Tensor,
    v1: torch.Tensor,
    q0_weight: torch.Tensor,
    k0_weight: torch.Tensor,
    q1_weight: torch.Tensor,
    k1_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    head_dim: int,
    rotary_dim: int,
    interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del q0_weight, k0_weight, q1_weight, k1_weight, rope_table, eps, head_dim, rotary_dim, interleaved
    seq_total = q0.shape[1] + q1.shape[1]
    return tuple(t.new_empty((t.shape[0], seq_total, t.shape[2], t.shape[3])) for t in (q0, k0, v0))


if not hasattr(torch.ops.vllm_omni, "fused_joint_qkv_norm_rope"):
    direct_register_custom_op(
        op_name="fused_joint_qkv_norm_rope",
        op_func=_fused_joint_qkv_norm_rope_impl,
        fake_impl=_fused_joint_qkv_norm_rope_fake,
        mutates_args=[],
        target_lib=_OMNI_OP_LIB,
    )


def fused_joint_qkv_norm_rope(
    q0: torch.Tensor,
    k0: torch.Tensor,
    v0: torch.Tensor,
    q1: torch.Tensor,
    k1: torch.Tensor,
    v1: torch.Tensor,
    q0_weight: torch.Tensor,
    k0_weight: torch.Tensor,
    q1_weight: torch.Tensor,
    k1_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    *,
    rotary_dim: int | None = None,
    interleaved: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused two-stream Q/K RMSNorm + RoPE (and V gather) into the joint sequence.

    ``q_i``/``k_i``/``v_i`` are ``[B, S_i, H, D]`` (stream 0 = text, stream
    1 = image for Flux.2); each stream's Q/K is normalized with its own
    ``[D]`` weights, and the result is ``[B, S_0 + S_1, H, D]`` Q, K and V
    with stream 0 first — exactly the tensors the dual-stream block hands
    to attention after its norm / ``cat`` / RoPE chain, in one launch and
    with no intermediate copies. ``rope_table`` is
    ``[B * (S_0 + S_1), rotary_dim]`` in joint token order
    (``[cos(theta) | sin(theta)]``, activation dtype or float32), so the
    same per-forward table serves this op and ``fused_qk_norm_rope`` on the
    single-stream blocks that follow.
    """
    streams = (q0, k0, v0, q1, k1, v1)
    if any(t.ndim != 4 for t in streams):
        raise ValueError("q0, k0, v0, q1, k1, v1 must be [batch, seq, heads, head_dim]")
    batch, seq_len_0, q_heads, head_dim = q0.shape
    seq_len_1 = q1.shape[1]
    kv_heads = k0.shape[2]
    expected = (
        (batch, seq_len_0, kv_heads, head_dim),
        (batch, seq_len_0, kv_heads, head_dim),
        (batch, seq_len_1, q_heads, head_dim),
        (batch, seq_len_1, kv_heads, head_dim),
        (batch, seq_len_1, kv_heads, head_dim),
    )
    if any(t.shape != shape for t, shape in zip(streams[1:], expected)):
        raise ValueError(f"stream shapes are incompatible: {[tuple(t.shape) for t in streams]}")
    if any(t.dtype != q0.dtype or t.device != q0.device for t in streams):
        raise ValueError("q0, k0, v0, q1, k1, v1 must share dtype and device")
    if q0.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(f"Fused QK RMSNorm/RoPE requires floating inputs, got {q0.dtype}")
    rotary_dim = rope_table.shape[-1] if rotary_dim is None else rotary_dim
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError(f"rotary_dim must be even and in [2, {head_dim}], got {rotary_dim}")
    weights = (q0_weight, k0_weight, q1_weight, k1_weight)
    if any(w.shape != (head_dim,) for w in weights):
        raise ValueError(f"Expected norm weights [{head_dim}], got {[tuple(w.shape) for w in weights]}")
    if any(w.device != q0.device for w in weights):
        raise ValueError("Q/K norm weights must be on the activation device")
    if rope_table.device != q0.device or rope_table.dtype not in (q0.dtype, torch.float32):
        raise ValueError("rope_table must be on q/k's device with their dtype or float32")
    tokens = batch * (seq_len_0 + seq_len_1)
    if rope_table.shape != (tokens, rotary_dim):
        raise ValueError(f"Expected rope_table [{tokens}, {rotary_dim}], got {tuple(rope_table.shape)}")

    weights = tuple(w.contiguous() for w in weights)
    rope_table = rope_table.contiguous()
    if not _fused_cuda_supported(q0, k0, head_dim, rotary_dim, interleaved) or not interleaved:
        return _fused_joint_qkv_norm_rope_impl(*streams, *weights, rope_table, eps, head_dim, rotary_dim, interleaved)
    return torch.ops.vllm_omni.fused_joint_qkv_norm_rope(
        *streams, *weights, rope_table, eps, head_dim, rotary_dim, interleaved
    )


_MIN_TOKENS_ENV = "VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS"


def fused_qk_norm_rope_min_tokens(default: int) -> int:
    """Token span below which a consumer should keep its eager chain.

    Taking the fused path costs a fixed amount of host time per call (the
    custom-op dispatch into the Python impl and the Triton launcher), while the
    GPU time it saves grows with the number of tokens the call covers. Below a
    host- and stack-dependent crossover a call is host-bound and the fused path
    can cost more end-to-end than it saves; above it the fused path wins.
    Consumers pass the default they measured on their reference hardware
    (Boogu-Image: 2048 tokens on one H200) and a deployment on other hardware
    overrides it once, for every consumer, through
    ``VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS`` (``0`` = always fuse).

    Resolved at call time — consumers call this once per forward, not per
    attention site — so tests and operators can flip it without re-importing.
    Unset or blank means the caller's default; anything else that is not a
    non-negative integer raises ``ValueError``.
    """
    raw = envs.VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        value = -1
    if value < 0:
        raise ValueError(f"{_MIN_TOKENS_ENV} must be a non-negative integer, got {raw!r}")
    return value


def pack_qk_norm_rope_table(
    cos: torch.Tensor,
    sin: torch.Tensor,
    batch_size: int,
    *,
    dtype: torch.dtype,
    min_tokens: int,
) -> torch.Tensor | None:
    """Pack theta-width ``cos``/``sin`` ``[S, D/2]`` (shared by the batch) into
    the ``[B*S, D] = [cos | sin]`` table the fused ops index by flattened
    token, in ``dtype``; ``None`` when ``B*S`` is below the consumer's token
    gate (``fused_qk_norm_rope_min_tokens(min_tokens)``), which keeps every
    attention site on its eager chain for that forward."""
    tokens = batch_size * cos.shape[0]
    if tokens < fused_qk_norm_rope_min_tokens(min_tokens):
        return None
    table = torch.cat((cos, sin), dim=-1).to(dtype)
    if batch_size > 1:
        table = table.unsqueeze(0).expand(batch_size, -1, -1).reshape(tokens, -1)
    return table


def fused_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope_table: torch.Tensor,
    eps: float,
    *,
    head_dim: int | None = None,
    rotary_dim: int | None = None,
    interleaved: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Q/K RMSNorm and packed RoPE (half-split or adjacent pairs)."""
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError(f"q and k must be [tokens, heads, head_dim], got {q.shape} and {k.shape}")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError(f"q and k shapes are incompatible: {q.shape} and {k.shape}")
    if q.dtype != k.dtype or q.device != k.device:
        raise ValueError("q and k must have the same dtype and device")
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(f"Fused QK RMSNorm/RoPE requires floating inputs, got {q.dtype}")

    head_dim = q.shape[-1] if head_dim is None else head_dim
    rotary_dim = rope_table.shape[-1] if rotary_dim is None else rotary_dim
    if q.shape[-1] != head_dim:
        raise ValueError(f"Expected q/k head_dim={head_dim}, got {q.shape[-1]}")
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError(f"rotary_dim must be even and in [2, {head_dim}], got {rotary_dim}")
    if q_weight.shape != (head_dim,) or k_weight.shape != (head_dim,):
        raise ValueError(f"Expected norm weights [{head_dim}], got {tuple(q_weight.shape)} and {tuple(k_weight.shape)}")
    if q_weight.device != q.device or k_weight.device != q.device:
        raise ValueError("Q/K norm weights must be on the activation device")
    if rope_table.device != q.device or rope_table.dtype not in (
        q.dtype,
        torch.float32,
    ):
        raise ValueError("rope_table must be on q/k's device with their dtype or float32")
    if rope_table.shape != (q.shape[0], rotary_dim):
        raise ValueError(f"Expected rope_table [{q.shape[0]}, {rotary_dim}], got {tuple(rope_table.shape)}")

    q_weight = q_weight.contiguous()
    k_weight = k_weight.contiguous()
    rope_table = rope_table.contiguous()
    if _fused_npu_supported(q, k, head_dim, rotary_dim, interleaved):
        return _npu_qk_norm_rope(
            q,
            k,
            q_weight,
            k_weight,
            rope_table,
            eps,
            rotary_dim,
        )
    if not _fused_cuda_supported(q, k, head_dim, rotary_dim, interleaved):
        return _fused_qk_norm_rope_impl(
            q,
            k,
            q_weight,
            k_weight,
            rope_table,
            eps,
            head_dim,
            rotary_dim,
            interleaved,
        )
    return torch.ops.vllm_omni.fused_qk_norm_rope(
        q,
        k,
        q_weight,
        k_weight,
        rope_table,
        eps,
        head_dim,
        rotary_dim,
        interleaved,
    )


__all__ = [
    "fused_joint_qkv_norm_rope",
    "fused_qk_norm_rope",
    "fused_qk_norm_rope_min_tokens",
    "pack_qk_norm_rope_table",
]
