# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2025-2026 SandAI. All Rights Reserved.
# ruff: noqa: N803

"""BF16 routed GEMMs for the MAGI-2 fast path.

The BF16 grouped GEMM is adapted from SGLang's Apache-2.0 implementation at
``d533d3bf464d280f180e18d4d396e613a5998502``. This module implements only
MAGI-2's interleaved gate/up GEMM with SwiGLU7 and routed down GEMM.
The existing SandAI-derived reference kernel is retained below unchanged.
"""

from __future__ import annotations

import math

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_omni.platforms import current_omni_platform


def swiglu7_pair(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Reference SwiGLU7 activation used by the CPU correctness path."""

    dtype = gate.dtype
    gate = gate.float().clamp(max=7.0)
    up = up.float().clamp(min=-7.0, max=7.0)
    return (gate * torch.sigmoid(1.702 * gate) * (up + 1.0)).to(dtype)


@triton.jit
def _write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, accumulator, mask=token_mask[:, None] & (offs_cn[None, :] < N))


@triton.jit
def _fused_moe_bf16_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    even_Ks: tl.constexpr,
    FUSE_SWIGLU: tl.constexpr,
    SWIGLU_ALPHA: tl.constexpr,
    SWIGLU_LIMIT: tl.constexpr,
):
    # Group program IDs to preserve the original L2 reuse order.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    off_experts_i32 = tl.load(expert_ids_ptr + pid_m)
    off_experts = off_experts_i32.to(tl.int64)
    if off_experts == -1:
        if FUSE_SWIGLU:
            # The filtered down GEMM never reads these half-width rows.
            return
        _write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_SIZE_K):
        if even_Ks:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k_start),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if FUSE_SWIGLU:
        # Keep GEMM accumulators in FP32 through SwiGLU7.
        acc_pairs = tl.reshape(accumulator, (BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2))
        gate, up = tl.split(acc_pairs)
        gate_c = tl.minimum(gate, SWIGLU_LIMIT)
        up_c = tl.maximum(tl.minimum(up, SWIGLU_LIMIT), -SWIGLU_LIMIT)
        out_act = (gate_c * tl.sigmoid(SWIGLU_ALPHA * gate_c) * (up_c + 1.0)).to(compute_type)
        offs_half = pid_n * (BLOCK_SIZE_N // 2) + tl.arange(0, BLOCK_SIZE_N // 2)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_half[None, :]
        tl.store(c_ptrs, out_act, mask=token_mask[:, None] & (offs_half[None, :] < N // 2))
    else:
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        tl.store(c_ptrs, accumulator.to(compute_type), mask=token_mask[:, None] & (offs_cn[None, :] < N))


def invoke_fused_moe_bf16(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    top_k: int,
    config: dict[str, int],
    fuse_swiglu: bool,
) -> None:
    """Launch one BF16 MAGI-2 GEMM in the original route order.

    Gate/up calls use interleaved ``B=[experts, 2*intermediate, hidden]`` and
    retain FP32 accumulators through SwiGLU7 before writing BF16
    ``C=[routes, intermediate]``. Down calls use ``B=[experts, hidden,
    intermediate]`` and write ``C=[tokens, router_top_k, hidden]``. The latter
    multiplies routing weights in FP32 before the BF16 output cast.
    """
    if any(t.dtype != torch.bfloat16 for t in (A, B, C)):
        raise ValueError("MAGI-2 fused MoE requires BF16 activations, weights, and output")
    if A.ndim != 2 or B.ndim != 3 or topk_weights.ndim != 2:
        raise ValueError("MAGI-2 fused MoE expects A [tokens, K], B [experts, N, K], and 2D routing weights")
    if not isinstance(fuse_swiglu, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("fuse_swiglu must be a bool and top_k must be a positive integer")
    expected_top_k = topk_weights.shape[1] if fuse_swiglu else 1
    if top_k != expected_top_k or A.shape[0] * top_k != topk_weights.numel():
        raise ValueError("top_k and activation rows do not match the gate/up or down routing layout")
    if B.shape[1] == 0 or B.shape[2] == 0 or A.shape[1] != B.shape[2]:
        raise ValueError("MAGI-2 fused MoE requires matching nonempty GEMM dimensions")
    if fuse_swiglu and B.shape[1] % 2:
        raise ValueError("SwiGLU7 requires an even number of interleaved gate/up rows")
    expected_shape = (topk_weights.numel(), B.shape[1] // 2) if fuse_swiglu else (*topk_weights.shape, B.shape[1])
    if C.shape != expected_shape or not C.is_contiguous():
        raise ValueError("MAGI-2 fused MoE output must have the contiguous routed output layout")
    if not topk_weights.is_floating_point() or not topk_weights.is_contiguous():
        raise ValueError("routing weights must be a contiguous floating-point tensor")
    for name, tensor in (
        ("sorted_token_ids", sorted_token_ids),
        ("expert_ids", expert_ids),
    ):
        if tensor.ndim != 1 or not tensor.is_contiguous() or tensor.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must be a contiguous 1D integer tensor")
    if num_tokens_post_padded.numel() != 1 or num_tokens_post_padded.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("num_tokens_post_padded must contain one integer")
    tensors = (B, C, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded)
    if any(t.device != A.device for t in tensors):
        raise ValueError("MAGI-2 fused MoE tensors must share a device")

    supported_config = {
        "BLOCK_SIZE_M": (128,),
        "BLOCK_SIZE_N": (64, 128, 256, 512),
        "BLOCK_SIZE_K": (16, 32, 64, 128),
        "GROUP_SIZE_M": (16,),
        "num_warps": (4, 8, 16, 32),
        "num_stages": (1, 2, 3, 4),
    }
    if config.keys() != supported_config.keys() or any(
        type(config[key]) is not int or config[key] not in values for key, values in supported_config.items()
    ):
        raise ValueError("invalid MAGI-2 BF16 MoE launch config")
    if expert_ids.numel() < triton.cdiv(sorted_token_ids.numel(), config["BLOCK_SIZE_M"]):
        raise ValueError("expert_ids must contain one expert per padded token block")
    if sorted_token_ids.numel() % config["BLOCK_SIZE_M"]:
        raise ValueError("sorted_token_ids capacity must be block aligned")

    def grid(meta: dict[str, int]) -> tuple[int]:
        return (
            triton.cdiv(sorted_token_ids.shape[0], meta["BLOCK_SIZE_M"])
            * triton.cdiv(B.shape[1], meta["BLOCK_SIZE_N"]),
        )

    _fused_moe_bf16_kernel[grid](
        A,
        B,
        C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        B.shape[2],
        sorted_token_ids.shape[0],
        topk_weights.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(-2),
        C.stride(-1),
        MUL_ROUTED_WEIGHT=not fuse_swiglu,
        top_k=top_k,
        compute_type=tl.bfloat16,
        even_Ks=B.shape[2] % config["BLOCK_SIZE_K"] == 0,
        FUSE_SWIGLU=fuse_swiglu,
        SWIGLU_ALPHA=1.702 if fuse_swiglu else 0.0,
        SWIGLU_LIMIT=7.0 if fuse_swiglu else 0.0,
        **config,
    )


def global_sort_routes(
    topk_probs: torch.Tensor,
    topk_indices: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert per-head routes into a stable flattened-expert CSR layout."""

    if topk_probs.shape != topk_indices.shape or topk_indices.ndim != 3:
        raise ValueError("top-k probabilities and indices must have the same [H,S,K] shape")
    heads, sequence, top_k = topk_indices.shape
    device = topk_indices.device
    head_offset = torch.arange(heads, device=device).view(heads, 1, 1) * num_experts
    flattened_experts = (topk_indices + head_offset).reshape(-1)
    flat_probs = topk_probs.reshape(-1)
    flat_tokens = torch.arange(sequence, device=device).view(1, sequence, 1).expand(heads, sequence, top_k).reshape(-1)
    order = flattened_experts.argsort(stable=True)
    gather_ids = flat_tokens[order].to(torch.int32)
    sorted_probs = flat_probs[order].float()
    counts = torch.bincount(flattened_experts, minlength=heads * num_experts)
    offsets = torch.zeros(heads * num_experts + 1, device=device, dtype=torch.long)
    offsets[1:] = counts.cumsum(0)
    return gather_ids, sorted_probs, offsets


def torch_mh_moe_forward(
    x: torch.Tensor,
    gather_ids: torch.Tensor,
    probs: torch.Tensor,
    expert_offsets: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Small-shape correctness oracle for the fused expert kernel."""

    if x.ndim != 3:
        raise ValueError("multi-head MoE input must be [tokens,heads,head_dim]")
    output = torch.zeros_like(x)
    experts_per_head = (expert_offsets.numel() - 1) // x.shape[1]
    for flat_expert in range(expert_offsets.numel() - 1):
        begin = int(expert_offsets[flat_expert].item())
        end = int(expert_offsets[flat_expert + 1].item())
        if begin == end:
            continue
        head = flat_expert // experts_per_head
        token_ids = gather_ids[begin:end].long()
        expert_input = x.index_select(0, token_ids)[:, head]
        gate = expert_input @ w_gate[flat_expert]
        up = expert_input @ w_up[flat_expert]
        hidden = swiglu7_pair(gate, up)
        expert_output = hidden @ w_down[flat_expert]
        expert_output = expert_output * probs[begin:end, None].to(expert_output.dtype)
        output[:, head].index_add_(0, token_ids, expert_output)
    return output


# Triton 3.7 keeps ``tl.constexpr`` as an annotation/type marker, not a
# callable constant constructor.  Plain Python literals are folded into the
# JIT specialization and remain portable across Triton 3.2 (MUSA) and 3.7
# (CUDA/H20).
# vLLM's no-Triton placeholder exposes ``tl.constexpr`` as ``None``.
_SWIGLU7_ALPHA = tl.constexpr(1.702) if HAS_TRITON else 1.702
_SWIGLU7_LIMIT = tl.constexpr(7.0) if HAS_TRITON else 7.0
_SWIGLU7_BIAS = tl.constexpr(1.0) if HAS_TRITON else 1.0


@triton.jit
def _swiglu7_kernel(gate, up, out_dtype: tl.constexpr):
    gate_clamped = tl.minimum(gate, _SWIGLU7_LIMIT)
    up_clamped = tl.maximum(tl.minimum(up, _SWIGLU7_LIMIT), -_SWIGLU7_LIMIT)
    sigmoid = tl.sigmoid(_SWIGLU7_ALPHA * gate_clamped)
    swish = gate_clamped * sigmoid
    return (swish * (up_clamped + _SWIGLU7_BIAS)).to(out_dtype)


@triton.jit
def _binary_search_expert(
    cumulative_tiles,
    tile_id,
    num_experts: tl.constexpr,
    log2_num_experts: tl.constexpr,
):
    lo = 0
    hi = num_experts
    for _ in tl.static_range(0, log2_num_experts + 1):
        mid = (lo + hi + 1) // 2
        below = tl.load(cumulative_tiles + mid) <= tile_id
        lo = tl.where(below, mid, lo)
        hi = tl.where(below, hi, mid - 1)
    return lo


@triton.jit
def _mh_moe_kernel(
    x_ptr,
    wg_ptr,
    wu_ptr,
    wd_ptr,
    y_ptr,
    gather_ids_ptr,
    probs_ptr,
    expert_offsets_ptr,
    cumulative_tiles_ptr,
    stride_x_s,
    stride_x_h,
    stride_x_dh,
    stride_wg_e,
    stride_wg_dh,
    stride_wg_de,
    stride_wu_e,
    stride_wu_dh,
    stride_wu_de,
    stride_wd_e,
    stride_wd_de,
    stride_wd_dh,
    stride_y_s,
    stride_y_h,
    stride_y_dh,
    d_head: tl.constexpr,
    d_expert: tl.constexpr,
    num_heads: tl.constexpr,
    num_flat_experts: tl.constexpr,
    log2_num_experts: tl.constexpr,
    block_t: tl.constexpr,
    block_dh: tl.constexpr,
    block_de: tl.constexpr,
    acc_dtype: tl.constexpr,
    deterministic: tl.constexpr = False,
):
    tile_id = tl.program_id(0)
    total_tiles = tl.load(cumulative_tiles_ptr + num_flat_experts)
    if tile_id >= total_tiles:
        return

    expert = _binary_search_expert(cumulative_tiles_ptr, tile_id, num_flat_experts, log2_num_experts)
    expert_i64 = expert.to(tl.int64)
    head = expert // (num_flat_experts // num_heads)
    tile_in_expert = tile_id - tl.load(cumulative_tiles_ptr + expert)
    token_start = tl.load(expert_offsets_ptr + expert) + tile_in_expert * block_t
    expert_end = tl.load(expert_offsets_ptr + expert + 1)
    count = tl.minimum(token_start + block_t, expert_end) - token_start

    dh_block_offsets = tl.arange(0, block_dh)
    de_block_offsets = tl.arange(0, block_de)
    token_offsets = tl.arange(0, block_t)
    dh_offsets = tl.arange(0, d_head)

    token_positions = token_start + token_offsets
    token_mask = token_offsets < count
    # Token indices fit in int32, but multiplying a large packed-batch index by
    # the hidden-width stride does not. Promote before computing element offsets.
    gather_ids = tl.load(gather_ids_ptr + token_positions, mask=token_mask, other=0).to(tl.int64)
    probabilities = tl.load(probs_ptr + token_positions, mask=token_mask, other=0.0)
    x_base = gather_ids * stride_x_s + head * stride_x_h
    output_acc = tl.zeros([block_t, d_head], dtype=acc_dtype)

    for de_start in tl.range(0, d_expert, block_de):
        de_offsets = de_start + de_block_offsets
        gate_acc = tl.zeros([block_t, block_de], dtype=acc_dtype)
        up_acc = tl.zeros([block_t, block_de], dtype=acc_dtype)
        for dh_start in tl.static_range(0, d_head, block_dh):
            local_dh = dh_start + dh_block_offsets
            x_block = tl.load(
                x_ptr + x_base[:, None] + local_dh[None, :] * stride_x_dh,
                mask=token_mask[:, None],
                other=0.0,
            )
            wg = tl.load(
                wg_ptr
                + expert_i64 * stride_wg_e
                + local_dh[:, None] * stride_wg_dh
                + de_offsets[None, :] * stride_wg_de
            )
            wu = tl.load(
                wu_ptr
                + expert_i64 * stride_wu_e
                + local_dh[:, None] * stride_wu_dh
                + de_offsets[None, :] * stride_wu_de
            )
            gate_acc += tl.dot(x_block, wg)
            up_acc += tl.dot(x_block, wu)
        hidden = _swiglu7_kernel(gate_acc, up_acc, wd_ptr.dtype.element_ty)
        down = tl.load(
            wd_ptr + expert_i64 * stride_wd_e + de_offsets[:, None] * stride_wd_de + dh_offsets[None, :] * stride_wd_dh
        )
        output_acc += tl.dot(hidden, down)
    output_acc = output_acc * probabilities[:, None]

    if deterministic:
        output_ptrs = y_ptr + token_positions[:, None] * stride_y_s + dh_offsets[None, :] * stride_y_dh
        tl.store(output_ptrs, output_acc.to(y_ptr.dtype.element_ty), mask=token_mask[:, None])
    else:
        output_base = gather_ids * stride_y_s + head * stride_y_h
        output_ptrs = y_ptr + output_base[:, None] + dh_offsets[None, :] * stride_y_dh
        tl.atomic_add(output_ptrs, output_acc.to(y_ptr.dtype.element_ty), mask=token_mask[:, None])


def _deterministic_scatter(
    sorted_output: torch.Tensor,
    reference: torch.Tensor,
    gather_ids: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> torch.Tensor:
    num_flat_experts = expert_offsets.numel() - 1
    experts_per_head = num_flat_experts // reference.shape[1]
    expert_lengths = torch.diff(expert_offsets)
    head_values = torch.arange(num_flat_experts, device=gather_ids.device) // experts_per_head
    head_ids = torch.repeat_interleave(head_values, expert_lengths)
    scatter_ids = gather_ids.long() * reference.shape[1] + head_ids.long()
    output = torch.zeros_like(reference).view(-1, reference.shape[-1])
    output.scatter_add_(0, scatter_ids[:, None].expand_as(sorted_output), sorted_output.to(output.dtype))
    return output.view_as(reference)


def _select_block_config() -> tuple[int, int, int, int, int]:
    """Return the reference kernel config, capped for pre-Blackwell GPUs."""

    capability = current_omni_platform.get_device_capability()
    if capability is not None and capability.major >= 10:  # Blackwell
        return (128, 64, 32, 2, 8)
    # BLOCK_T=128 needs 122,880 bytes of shared memory and is not safe on the
    # qualified L20X path.  This is the reference kernel's portable config.
    return (64, 64, 32, 2, 4)


def triton_mh_moe_forward(
    x: torch.Tensor,
    gather_ids: torch.Tensor,
    probs: torch.Tensor,
    expert_offsets: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    *,
    deterministic: bool = False,
) -> torch.Tensor:
    """Fused gather/expert/scatter kernel for released MAGI dimensions."""

    routed_tokens = gather_ids.numel()
    if routed_tokens == 0:
        return torch.zeros_like(x)
    d_head, d_expert = x.shape[-1], w_down.shape[1]
    block_t, block_dh, block_de, num_stages, num_warps = _select_block_config()
    if d_head % block_dh or d_expert % block_de:
        return torch_mh_moe_forward(x, gather_ids, probs, expert_offsets, w_gate, w_up, w_down)

    if deterministic:
        output = torch.empty((routed_tokens, 1, d_head), device=x.device, dtype=x.dtype)
    else:
        output = torch.zeros_like(x)
    num_flat_experts = expert_offsets.numel() - 1
    expert_tiles = (torch.diff(expert_offsets) + block_t - 1) // block_t
    cumulative_tiles = torch.cat(
        (torch.zeros(1, dtype=torch.int32, device=x.device), expert_tiles.cumsum(0, dtype=torch.int32))
    )
    # Match the reference launch bound.  Empty/excess programs return after
    # comparing against ``cumulative_tiles[-1]`` inside the kernel.
    grid = ((routed_tokens + block_t - 1) // block_t + num_flat_experts,)
    log2_experts = max(1, math.ceil(math.log2(max(num_flat_experts, 1) + 1)))
    _mh_moe_kernel[grid](
        x,
        w_gate,
        w_up,
        w_down,
        output,
        gather_ids,
        probs,
        expert_offsets,
        cumulative_tiles,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        w_gate.stride(0),
        w_gate.stride(1),
        w_gate.stride(2),
        w_up.stride(0),
        w_up.stride(1),
        w_up.stride(2),
        w_down.stride(0),
        w_down.stride(1),
        w_down.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        d_head,
        d_expert,
        x.shape[1],
        num_flat_experts,
        log2_experts,
        block_t,
        block_dh,
        block_de,
        tl.float32,
        deterministic,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    if deterministic:
        return _deterministic_scatter(output.view(routed_tokens, d_head), x, gather_ids, expert_offsets)
    return output
