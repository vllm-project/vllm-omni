# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) 2025-2026 SandAI. All Rights Reserved.

"""Native multi-head MoE used by MAGI-2 Preview.

Adapted from SandAI's Apache-2.0 ``flash_mh_moe`` implementation and modified
to use vLLM's existing expert-parallel group.  MAGI's routing is unusual: each
of twelve 256-wide hidden-state heads independently selects experts from its
own 256-expert bank.  It is therefore not representable by vLLM's conventional
whole-token :class:`FusedMoE` primitive.
"""

from __future__ import annotations

import importlib
import logging
import math
import os
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.triton_utils import tl, triton

from vllm_omni.platforms import current_omni_platform

from .parallel import Magi2ParallelGroup, ep_dispatch, ep_undispatch, get_magi2_ep_group

RoutingScore = Literal["softmax", "sigmoid"]

logger = logging.getLogger(__name__)
_MATE_MOE_WARNED = False
_SGL_FUSED_MOE_MODULE = None
_SGL_FUSED_MOE_WORKSPACES: dict[tuple, torch.Tensor] = {}
_MAGI2_ALIGN_OP_AVAILABLE: bool | None = None


def _magi2_align_block_size_fixed_capacity(
    ids: torch.Tensor,
    num_experts: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build aligned route metadata without device-to-host scalar reads.

    The generic ``_moe_C`` align op is not present in the vLLM-MUSA image.
    Its eager fallback used a dynamic ``repeat_interleave`` result and called
    ``.item()`` twice per MoE layer to discover the published length.  Keep
    the buffers at their deterministic worst-case capacity instead: the
    fused Triton kernel already consumes ``num_tokens_post_pad`` on device
    and returns early for the unused tail.  ``searchsorted`` maps block
    indices to experts without a dynamic output allocation.
    """

    route_count = ids.numel()
    capacity = max(route_count + num_experts * (block_size - 1), block_size)
    max_blocks = (capacity + block_size - 1) // block_size
    sorted_ids = torch.full(
        (capacity,), route_count, device=ids.device, dtype=torch.int32
    )

    # ``torch.sort`` returns values and indices in one operation and preserves
    # the route order used by the previous argsort + gather implementation.
    sorted_experts, order = torch.sort(ids)
    counts = torch.zeros(num_experts, device=ids.device, dtype=torch.int32)
    if route_count:
        counts.scatter_add_(
            0, ids, torch.ones_like(ids, dtype=torch.int32)
        )
    padded_counts = ((counts + block_size - 1) // block_size) * block_size
    starts = torch.cumsum(padded_counts, 0) - padded_counts
    ends = torch.cumsum(counts, 0)
    begins = ends - counts
    positions = torch.arange(route_count, device=ids.device, dtype=torch.int64)
    experts = sorted_experts.to(torch.int64)
    destinations = starts[experts] + positions - begins[experts]
    sorted_ids[destinations] = order.to(torch.int32)

    block_counts = padded_counts // block_size
    cumulative_blocks = torch.cumsum(block_counts, 0)
    block_indices = torch.arange(max_blocks, device=ids.device, dtype=torch.int32)
    mapped_experts = torch.searchsorted(
        cumulative_blocks, block_indices, right=True
    ).to(torch.int32)
    expert_ids = torch.where(
        block_indices < cumulative_blocks[-1], mapped_experts, -1
    ).to(torch.int32)
    num_padded = padded_counts.sum(dtype=torch.int32).reshape(1)
    return sorted_ids, expert_ids, num_padded


def _magi2_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return SGLang-compatible sorted route metadata on MUSA.

    The vLLM CUDA extension exposes an in-place align op, but the MUSA image
    used for MAGI-2 does not load ``_moe_C``.  Keep a device-side fallback so
    the fused path remains usable while an AOT MUSA align op is upstreamed.
    """
    ids = topk_ids.reshape(-1).to(torch.int32)
    align_mode = os.environ.get("MAGI2_MOE_ALIGN_FIXED_CAPACITY")
    force_fixed_capacity = align_mode == "1"
    force_dynamic_fallback = align_mode == "0"
    # Keep the existing CUDA/CPU behavior untouched.  The fixed-capacity
    # implementation is a MUSA fallback for the image where the Python
    # wrapper is present but the CUDA ``_moe_C`` extension is absent; callers
    # can still force it on another backend for diagnostics.
    musa_device = ids.device.type in {"musa", "privateuseone"}
    if force_fixed_capacity:
        return _magi2_align_block_size_fixed_capacity(
            ids, num_experts, block_size
        )

    global _MAGI2_ALIGN_OP_AVAILABLE
    if not force_dynamic_fallback and _MAGI2_ALIGN_OP_AVAILABLE is not False:
        try:
            from vllm._custom_ops import moe_align_block_size

            capacity = ids.numel() + num_experts * block_size
            sorted_ids = torch.empty(
                (capacity,), device=ids.device, dtype=torch.int32
            )
            expert_ids = torch.empty(
                (capacity // block_size,), device=ids.device, dtype=torch.int32
            )
            num_padded = torch.empty((1,), device=ids.device, dtype=torch.int32)
            moe_align_block_size(
                ids,
                num_experts,
                block_size,
                sorted_ids,
                expert_ids,
                num_padded,
            )
            _MAGI2_ALIGN_OP_AVAILABLE = True
            n = int(num_padded.item())
            return sorted_ids[:n], expert_ids[: n // block_size], num_padded
        except Exception:
            # Some MUSA images expose the Python wrapper but not its CUDA
            # ``_moe_C`` symbol.  Cache that fact to avoid an exception and
            # failed custom-op dispatch on every layer.
            _MAGI2_ALIGN_OP_AVAILABLE = False

    if musa_device and not force_dynamic_fallback:
        # The MUSA image normally has the Python wrapper but not its CUDA
        # ``_moe_C`` implementation. Prefer the device-count/capacity path by
        # default; setting ``MAGI2_MOE_ALIGN_FIXED_CAPACITY=0`` retains the
        # original dynamic fallback as an emergency rollback.
        return _magi2_align_block_size_fixed_capacity(
            ids, num_experts, block_size
        )

    # MUSA builds may expose the Python wrapper without the CUDA `_moe_C`
    # extension.  This implementation intentionally stays on-device.
    order = torch.argsort(ids)
    sorted_experts = ids[order]
    counts = torch.bincount(ids, minlength=num_experts)
    padded_counts = ((counts + block_size - 1) // block_size) * block_size
    total = int(padded_counts.sum().item())
    sorted_ids = torch.full(
        (total,), ids.numel(), device=ids.device, dtype=torch.int32
    )
    starts = torch.cumsum(padded_counts, 0) - padded_counts
    ends = torch.cumsum(counts, 0)
    begins = ends - counts
    positions = torch.arange(ids.numel(), device=ids.device, dtype=torch.int64)
    experts = sorted_experts.to(torch.int64)
    destinations = starts[experts] + positions - begins[experts]
    sorted_ids[destinations] = order.to(torch.int32)
    expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, device=ids.device, dtype=torch.int32),
        padded_counts // block_size,
    )
    return (
        sorted_ids,
        expert_ids,
        torch.tensor(total, device=ids.device, dtype=torch.int32),
    )


def _magi2_sgl_fused_moe_module():
    global _SGL_FUSED_MOE_MODULE
    if _SGL_FUSED_MOE_MODULE is None:
        _SGL_FUSED_MOE_MODULE = importlib.import_module(
            "vllm_omni.diffusion.models.magi2.sgl_fused_moe_kernels"
        )
    return _SGL_FUSED_MOE_MODULE


def _magi2_sgl_fused_moe_forward(
    x_heads: torch.Tensor,
    probabilities: torch.Tensor,
    indices: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    packed_w13: torch.Tensor | None = None,
    packed_w2: torch.Tensor | None = None,
) -> torch.Tensor:
    """SGLang-compatible fused SwiGLU MoE for MAGI-2's head-routed layout."""
    if x_heads.ndim != 3 or probabilities.ndim != 3 or indices.ndim != 3:
        raise ValueError("unexpected MAGI-2 MoE tensor rank")
    num_tokens, num_heads, hidden_size = x_heads.shape
    top_k = probabilities.shape[-1]
    if top_k != 6:
        raise ValueError(f"SGLang MAGI-2 path expects top_k=6, got {top_k}")
    num_experts_per_head = w_gate.shape[0] // num_heads
    intermediate_size = w_gate.shape[-1]
    num_experts = w_gate.shape[0]
    # The generic fused kernel consumes one row per (head, token), with expert
    # IDs offset into the flattened head-local expert table.
    hidden = x_heads.permute(1, 0, 2).contiguous().reshape(
        num_heads * num_tokens, hidden_size
    )
    route_weights = probabilities.permute(0, 1, 2).contiguous().reshape(
        num_heads * num_tokens, top_k
    )
    route_ids = indices.to(torch.int32).contiguous()
    route_ids = (
        route_ids
        + torch.arange(num_heads, device=route_ids.device, dtype=torch.int32)
        .view(num_heads, 1, 1)
        .mul(num_experts_per_head)
    ).reshape(num_heads * num_tokens, top_k)
    sorted_ids, expert_ids, num_padded = _magi2_align_block_size(
        route_ids, num_experts, 128
    )

    # W13 is packed only into one per-process scratch buffer.  Keeping this
    # buffer out of each layer avoids a resident ~1 GB duplicate for all 36
    # MAGI-2 MoE layers; the model invokes layers serially on each rank.
    key = (
        str(w_gate.device),
        w_gate.dtype,
        num_experts,
        intermediate_size,
        hidden_size,
    )
    shape = (num_experts, 2 * intermediate_size, hidden_size)
    if packed_w13 is not None:
        if tuple(packed_w13.shape) != shape or not packed_w13.is_contiguous():
            raise ValueError("Invalid prepacked MAGI2 W13 layout")
        packed = packed_w13
    else:
        packed = _SGL_FUSED_MOE_WORKSPACES.get(key)
        if packed is None or tuple(packed.shape) != shape:
            packed = torch.empty(shape, device=w_gate.device, dtype=w_gate.dtype)
            _SGL_FUSED_MOE_WORKSPACES[key] = packed
        packed_view = packed.view(num_experts, intermediate_size, 2, hidden_size)
        packed_view[:, :, 0, :].copy_(w_gate.transpose(1, 2))
        packed_view[:, :, 1, :].copy_(w_up.transpose(1, 2))

    kernels = _magi2_sgl_fused_moe_module()
    import triton.language as tl

    block_n = int(os.environ.get("MAGI2_SGL_BLOCK_N", "128"))
    default_block_k = "32" if packed_w13 is not None else "64"
    block_k = int(os.environ.get("MAGI2_SGL_BLOCK_K", default_block_k))
    if block_n not in (64, 128, 256, 512) or block_k not in (16, 32, 64, 128):
        raise ValueError("invalid MAGI2 SGL tile config")
    num_warps = int(os.environ.get("MAGI2_SGL_NUM_WARPS", "16"))
    num_stages = int(os.environ.get("MAGI2_SGL_NUM_STAGES", "1"))
    if num_warps not in (4, 8, 16, 32) or num_stages not in (1, 2, 3, 4):
        raise ValueError("invalid MAGI2 SGL launch config")
    config = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": block_k,
        "GROUP_SIZE_M": 16,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }
    config_down = dict(config)
    if packed_w13 is not None and os.environ.get("MAGI2_SGL_BLOCK_K") is None:
        config_down["BLOCK_SIZE_K"] = 64
    # c_sorted=False uses original route IDs and masks padded rows with
    # num_valid_tokens; only real routed rows need intermediate storage.
    # Keep zero initialization for any filtered experts.
    intermediate_rows = route_ids.numel()
    intermediate = torch.zeros(
        (intermediate_rows, intermediate_size),
        device=x_heads.device,
        dtype=x_heads.dtype,
    )
    kernels.invoke_fused_moe_kernel(
        hidden,
        packed,
        None,
        intermediate,
        None,
        None,
        None,
        route_weights,
        route_ids,
        sorted_ids,
        expert_ids,
        num_padded,
        False,
        top_k,
        config,
        tl.bfloat16,
        False,
        False,
        False,
        False,
        False,
        no_combine=True,
        c_sorted=False,
        filter_expert=True,
        fuse_swiglu=True,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
    )
    route_output = torch.zeros(
        (hidden.shape[0], top_k, hidden_size),
        device=hidden.device,
        dtype=hidden.dtype,
    )
    kernels.invoke_fused_moe_kernel(
        intermediate,
        # The fused kernel follows SGLang's convention for W2: [E, K, N]
        # (output/hidden dimension first).  MAGI-2 stores W_down as
        # [E, N, K], so expose the transposed view here.
        packed_w2 if packed_w2 is not None else w_down.transpose(1, 2),
        None,
        route_output,
        None,
        None,
        None,
        route_weights,
        route_ids,
        sorted_ids,
        expert_ids,
        num_padded,
        True,
        1,
        config_down,
        tl.bfloat16,
        False,
        False,
        False,
        False,
        False,
        no_combine=False,
        c_sorted=False,
        filter_expert=True,
    )
    # SGLang's MUSA AOT ``moe_sum_reduce`` combines the six routed rows in
    # one custom kernel.  Use the equivalent reduction while we keep the
    # adapter self-contained; this is a single device reduction and leaves a
    # clean seam for replacing it with the AOT op when available in the image.
    output = torch.empty(
        (hidden.shape[0], hidden_size), device=hidden.device, dtype=hidden.dtype
    )
    used_fast_sum = False
    if os.environ.get("MAGI2_USE_MUSA_MOE_SUM", "1") == "1":
        try:
            from vllm_musa.jit_kernel.csrc.moe import maybe_fast_moe_sum

            used_fast_sum = maybe_fast_moe_sum(route_output, output)
        except Exception as exc:
            logger.debug("MUSA MoE sum-reduce unavailable: %s", exc)
    if not used_fast_sum:
        output.copy_(route_output.sum(dim=1))
    return output.view(num_heads, num_tokens, hidden_size).permute(1, 0, 2).contiguous()


def swiglu7_pair(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Released clamped SwiGLU7 expert activation, evaluated in fp32."""

    dtype = gate.dtype
    gate = gate.float().clamp(max=7.0)
    up = up.float().clamp(min=-7.0, max=7.0)
    return (gate * torch.sigmoid(1.702 * gate) * (up + 1.0)).to(dtype)


def compute_topk_probs_and_indices(
    router_logits: torch.Tensor,
    top_k: int,
    *,
    score_func: RoutingScore = "sigmoid",
    expert_bias: torch.Tensor | None = None,
    route_norm: bool = True,
    norm_eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route independently for every ``[head, token]`` pair.

    The auxiliary-free bias affects expert selection but deliberately does not
    affect the returned routing probability, matching the training recipe.
    """

    if router_logits.ndim != 3:
        raise ValueError("router_logits must be [heads,tokens,experts]")
    if not 0 < top_k <= router_logits.shape[-1]:
        raise ValueError("top_k must be in [1, num_experts]")
    if score_func == "sigmoid":
        router_scores = torch.sigmoid(router_logits)
    elif score_func == "softmax":
        router_scores = torch.softmax(router_logits, dim=-1)
    else:
        raise ValueError(f"unsupported routing score function {score_func!r}")
    selection_scores = router_scores
    if expert_bias is not None:
        selection_scores = selection_scores + expert_bias.view(router_logits.shape[0], 1, -1)
    # Keep the reference's default sorted=True behavior.  Besides defining the
    # route order for ties, this also fixes the reduction order used by the
    # following L1 normalization.
    topk_indices = torch.topk(selection_scores, top_k, dim=-1).indices
    topk_probs = router_scores.gather(-1, topk_indices)
    if route_norm:
        topk_probs = F.normalize(topk_probs, p=1, dim=-1, eps=norm_eps)
    return topk_probs, topk_indices


def _global_sort_routes_impl(
    topk_probs: torch.Tensor,
    topk_indices: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the stable flattened-expert route layout and retain sort order."""

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
    # ``torch.bincount`` on MUSA performs an internal min/max and scalar
    # synchronization before building its histogram.  Route IDs are already
    # bounded non-negative integers, so an int32 scatter accumulation is both
    # exact and substantially cheaper while preserving the same CSR counts.
    counts = torch.zeros(heads * num_experts, device=device, dtype=torch.int32)
    if flattened_experts.numel():
        counts.scatter_add_(
            0,
            flattened_experts,
            torch.ones_like(flattened_experts, dtype=torch.int32),
        )
    offsets = torch.zeros(heads * num_experts + 1, device=device, dtype=torch.long)
    offsets[1:] = counts.cumsum(0)
    return gather_ids, sorted_probs, offsets, order, counts


def global_sort_routes(
    topk_probs: torch.Tensor,
    topk_indices: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert per-head routes into a stable flattened-expert CSR layout.

    Keep this three-tensor return contract for existing callers.  The MATE
    path can use :func:`global_sort_routes_with_head_ids` when it also needs
    the source head for every sorted route.
    """

    gather_ids, sorted_probs, offsets, _, _ = _global_sort_routes_impl(
        topk_probs, topk_indices, num_experts
    )
    return gather_ids, sorted_probs, offsets


def global_sort_routes_with_head_ids(
    topk_probs: torch.Tensor,
    topk_indices: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sorted routes plus source-head IDs without expanding counts.

    ``flattened_experts`` is laid out as ``[head, sequence, top_k]`` before
    sorting.  Therefore the source head of a sorted route is recoverable from
    its stable-sort position with one integer division.  The optional metadata
    lets the MATE adapter avoid constructing an ``E``-element head table and
    ``repeat_interleave``-ing it to route length on every layer.
    """

    gather_ids, sorted_probs, offsets, order, _ = _global_sort_routes_impl(
        topk_probs, topk_indices, num_experts
    )
    _, sequence, top_k = topk_indices.shape
    if order.numel() == 0:
        sorted_head_ids = order
    else:
        sorted_head_ids = torch.div(order, sequence * top_k, rounding_mode="floor")
    return gather_ids, sorted_probs, offsets, sorted_head_ids


def global_sort_routes_with_head_ids_and_counts(
    topk_probs: torch.Tensor,
    topk_indices: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sorted routes, source heads, and the exact route counts.

    The count vector is the same int32 histogram used to build ``offsets``.
    MATE's per-expert grouped GEMM consumes it directly, so exposing it here
    avoids a second ``diff(offsets)`` plus dtype/layout conversion in the
    adapter while keeping the legacy three-/four-tensor APIs unchanged.
    """

    gather_ids, sorted_probs, offsets, order, counts = _global_sort_routes_impl(
        topk_probs, topk_indices, num_experts
    )
    _, sequence, top_k = topk_indices.shape
    if order.numel() == 0:
        sorted_head_ids = order
    else:
        sorted_head_ids = torch.div(order, sequence * top_k, rounding_mode="floor")
    return gather_ids, sorted_probs, offsets, sorted_head_ids, counts


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
    # Materialize the compact CSR offsets once per layer. Calling ``item``
    # for every expert forces a device-to-host synchronization for each
    # route; one bounded copy preserves the exact route order while avoiding
    # thousands of scalar syncs across the DiT stack.
    offsets = expert_offsets.detach().to(device="cpu").tolist()
    for flat_expert, (begin, end) in enumerate(zip(offsets, offsets[1:])):
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


def _mate_bf16_grouped_linear(
    input_a: torch.Tensor,
    weight: torch.Tensor,
    token_counts: torch.Tensor,
    *,
    major_b_mode: Literal["N"],
    backend: str,
) -> torch.Tensor:
    """Run one MAGI expert projection through MATE's ragged BF16 GEMM.

    MAGI stores gate/up/down weights as ``[expert, K, N]``.  The lower-level
    MATE entry point is used instead of materializing transposed gate/up
    copies: its ``major_b_mode="N"`` contract
    accepts the checkpoint's K-major layout directly.
    """

    if major_b_mode != "N":
        raise ValueError("MAGI grouped BF16 weights must use MATE major_b_mode='N'")
    if (
        not input_a.is_contiguous()
        or not weight.is_contiguous()
        or not token_counts.is_contiguous()
    ):
        raise ValueError("MATE grouped BF16 operands must be contiguous")
    # MAGI checkpoint tensors are all stored as ``[K, N]`` per expert.  The
    # MATE ``N`` major mode describes this physical layout and exposes the
    # trailing dimension as the output width.
    out_features = weight.shape[-1]
    output = torch.empty(
        (input_a.shape[0], out_features), device=input_a.device, dtype=input_a.dtype
    )
    from mate.gemm import ragged_m_moe_gemm_16bit

    ragged_m_moe_gemm_16bit(
        input_a,
        weight,
        token_counts,
        output,
        gemm_mode="per_expert",
        major_a_mode="K",
        major_b_mode=major_b_mode,
        backend=backend,
    )
    return output


def mate_bf16_mh_moe_forward(
    x: torch.Tensor,
    gather_ids: torch.Tensor,
    probs: torch.Tensor,
    expert_offsets: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    *,
    backend: str = "mubin",
    sorted_head_ids: torch.Tensor | None = None,
    token_counts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate the routed MAGI-2 MoE with MATE's grouped BF16 GEMMs.

    ``global_sort_routes`` already lays routes out as contiguous expert
    segments.  We only gather the corresponding head slices once, run the
    three projections as ragged grouped GEMMs, and scatter the weighted result
    once.  ``sorted_head_ids`` and ``token_counts`` are optional companions
    produced by :func:`global_sort_routes_with_head_ids_and_counts`; supplying
    them avoids rebuilding route metadata in the adapter. This is intentionally
    an opt-in experiment until the complete distributed/capture matrix has
    been validated.
    """

    if x.ndim != 3:
        raise ValueError("multi-head MoE input must be [tokens,heads,head_dim]")
    if x.dtype != torch.bfloat16 or any(
        weight.dtype != torch.bfloat16 for weight in (w_gate, w_up, w_down)
    ):
        raise ValueError("MATE BF16 grouped MoE requires BF16 activations and weights")
    num_flat_experts = expert_offsets.numel() - 1
    if num_flat_experts <= 0 or num_flat_experts % x.shape[1]:
        raise ValueError("expert count must be a positive multiple of the head count")
    if gather_ids.numel() == 0:
        return torch.zeros_like(x)

    if token_counts is None:
        token_counts = torch.diff(expert_offsets).to(dtype=torch.int32).contiguous()
    else:
        if token_counts.ndim != 1 or token_counts.numel() != num_flat_experts:
            raise ValueError("token_counts must have one entry per flattened expert")
        if token_counts.device != x.device:
            raise ValueError("token_counts must be on the MoE input device")
        if token_counts.dtype != torch.int32 or not token_counts.is_contiguous():
            token_counts = token_counts.to(dtype=torch.int32).contiguous()
    token_ids = gather_ids.to(dtype=torch.long)
    if sorted_head_ids is None:
        experts_per_head = num_flat_experts // x.shape[1]
        head_for_expert = torch.arange(
            num_flat_experts, device=x.device, dtype=torch.long
        ) // experts_per_head
        head_ids = torch.repeat_interleave(head_for_expert, token_counts.to(torch.long))
    else:
        if sorted_head_ids.ndim != 1 or sorted_head_ids.numel() != token_ids.numel():
            raise ValueError("sorted_head_ids must be one entry per sorted route")
        if sorted_head_ids.device != x.device:
            raise ValueError("sorted_head_ids must be on the MoE input device")
        head_ids = sorted_head_ids.to(dtype=torch.long)
    route_linear_ids = (token_ids * x.shape[1] + head_ids).contiguous()
    routed_input = x.contiguous().view(-1, x.shape[-1]).index_select(0, route_linear_ids)

    gate = _mate_bf16_grouped_linear(
        routed_input,
        w_gate,
        token_counts,
        major_b_mode="N",
        backend=backend,
    )
    up = _mate_bf16_grouped_linear(
        routed_input,
        w_up,
        token_counts,
        major_b_mode="N",
        backend=backend,
    )
    hidden = swiglu7_pair(gate, up)
    expert_output = _mate_bf16_grouped_linear(
        hidden,
        w_down,
        token_counts,
        major_b_mode="N",
        backend=backend,
    )
    expert_output = expert_output * probs[:, None].to(expert_output.dtype)
    output = torch.zeros(
        (x.shape[0] * x.shape[1], x.shape[-1]),
        device=x.device,
        dtype=x.dtype,
    )
    output.index_add_(0, route_linear_ids, expert_output)
    return output.view_as(x)


_SWIGLU7_ALPHA = tl.constexpr(1.702)
_SWIGLU7_LIMIT = tl.constexpr(7.0)
_SWIGLU7_BIAS = tl.constexpr(1.0)


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
    acc_dtype: tl.constexpr = tl.float32,
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


@dataclass(frozen=True)
class Magi2MultiHeadMoEConfig:
    hidden_size: int
    num_heads: int
    num_experts: int
    top_k: int
    expert_intermediate_size: int
    params_dtype: torch.dtype
    score_func: RoutingScore = "sigmoid"
    route_norm: bool = True
    route_scale: float = 1.0


class Magi2MultiHeadMoE(nn.Module):
    """Checkpoint-compatible MAGI-2 head-routed expert layer."""

    _EP_SHARDED_PARAMETER_NAMES = frozenset(
        {"gate", "W_gate", "W_up", "W_down", "router.expert_bias", "router.expert_bias_ema"}
    )

    def __init__(
        self,
        config: Magi2MultiHeadMoEConfig,
        *,
        ep_group: Magi2ParallelGroup | None = None,
    ) -> None:
        super().__init__()
        if config.hidden_size % config.num_heads:
            raise ValueError("hidden_size must be divisible by the number of MoE heads")
        self.config = config
        self.num_heads = config.num_heads
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.d_head = config.hidden_size // config.num_heads
        self.d_expert = config.expert_intermediate_size
        self.ep_group = ep_group or get_magi2_ep_group()
        self.padded_num_heads = math.ceil(self.num_heads / self.ep_group.world_size) * self.ep_group.world_size
        self.local_num_heads = self.padded_num_heads // self.ep_group.world_size
        self.local_flatten_num_experts = self.local_num_heads * self.num_experts
        self.ep_pad_heads = self.padded_num_heads - self.num_heads
        self.local_head_start = self.ep_group.rank * self.local_num_heads
        self.has_real_moe_heads = self.local_head_start < self.num_heads

        self.gate = nn.Parameter(torch.empty(self.local_flatten_num_experts, self.d_head, dtype=torch.float32))
        self.W_gate = nn.Parameter(
            torch.empty(self.local_flatten_num_experts, self.d_head, self.d_expert, dtype=config.params_dtype)
        )
        self.W_up = nn.Parameter(
            torch.empty(self.local_flatten_num_experts, self.d_head, self.d_expert, dtype=config.params_dtype)
        )
        self.W_down = nn.Parameter(
            torch.empty(self.local_flatten_num_experts, self.d_expert, self.d_head, dtype=config.params_dtype)
        )
        self.register_buffer("_owned_w13", None, persistent=False)
        self.register_buffer("_owned_w2", None, persistent=False)
        self.router = nn.Module()
        # Both tensors are released checkpoint entries.  Non-trainable
        # Parameters let the DLO mmap path bind them on a meta-constructed
        # model; persistent buffers are intentionally not mmap-loaded by the
        # generic backend.
        self.router.expert_bias = nn.Parameter(
            torch.zeros(self.local_flatten_num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        self.router.expert_bias_ema = nn.Parameter(
            torch.zeros(self.local_flatten_num_experts, dtype=torch.float32),
            requires_grad=False,
        )

        for name in self._EP_SHARDED_PARAMETER_NAMES:
            target: nn.Module | Magi2MultiHeadMoE = self
            parts = name.split(".")
            for part in parts[:-1]:
                target = getattr(target, part)
            parameter = getattr(target, parts[-1])
            parameter.mmap_weight_transform = self.ep_slice

    @torch.no_grad()
    def prepare_owned_w13(self) -> None:
        """Replace W_gate/W_up storage with SGL's interleaved W13 layout."""
        if self.W_gate.dtype != torch.bfloat16 or self.W_up.dtype != torch.bfloat16:
            raise ValueError("Owned W13 requires BF16 gate and up weights")
        if self._get_owned_w13() is not None:
            return
        packed = torch.empty(
            (self.local_flatten_num_experts, 2 * self.d_expert, self.d_head),
            dtype=self.W_gate.dtype,
            device=self.W_gate.device,
        )
        views = packed.view(self.local_flatten_num_experts, self.d_expert, 2, self.d_head)
        gate = views[:, :, 0, :].transpose(1, 2)
        up = views[:, :, 1, :].transpose(1, 2)
        gate.copy_(self.W_gate)
        up.copy_(self.W_up)
        self.W_gate.data = gate
        self.W_up.data = up
        self._owned_w13 = packed

    def _get_owned_w13(self) -> torch.Tensor | None:
        packed = self._owned_w13
        if packed is None:
            return None
        expected_stride = (2 * self.d_expert * self.d_head, 1, 2 * self.d_head)
        if (
            packed.device != self.W_gate.device
            or packed.dtype != self.W_gate.dtype
            or self.W_up.device != packed.device
            or self.W_up.dtype != packed.dtype
            or self.W_gate.data_ptr() != packed.data_ptr()
            or self.W_up.data_ptr() != packed.data_ptr() + self.d_head * packed.element_size()
            or self.W_gate.stride() != expected_stride
            or self.W_up.stride() != expected_stride
        ):
            return None
        return packed

    @torch.no_grad()
    def prepare_owned_w2(self) -> None:
        """Replace W2 storage with SGL's contiguous [E,H,I] layout."""
        if self.W_down.dtype != torch.bfloat16:
            raise ValueError("Owned W2 requires BF16 weights")
        if self._get_owned_w2() is not None:
            return
        packed = torch.empty(
            (self.local_flatten_num_experts, self.d_head, self.d_expert),
            dtype=self.W_down.dtype,
            device=self.W_down.device,
        )
        packed.copy_(self.W_down.transpose(1, 2))
        self.W_down.data = packed.transpose(1, 2)
        self._owned_w2 = packed

    def _get_owned_w2(self) -> torch.Tensor | None:
        packed = self._owned_w2
        if packed is None:
            return None
        if (
            packed.device != self.W_down.device
            or packed.dtype != self.W_down.dtype
            or self.W_down.data_ptr() != packed.data_ptr()
            or self.W_down.stride() != (packed.stride(0), packed.stride(2), packed.stride(1))
        ):
            return None
        return packed

    def ep_slice(self, checkpoint_tensor: torch.Tensor) -> torch.Tensor:
        """Slice flattened ``(head,expert)`` checkpoint rows for this rank."""

        if checkpoint_tensor.shape[0] == self.local_flatten_num_experts:
            return checkpoint_tensor
        start = self.local_head_start * self.num_experts
        end = min(start + self.local_flatten_num_experts, checkpoint_tensor.shape[0])
        if start >= checkpoint_tensor.shape[0]:
            return torch.zeros(
                (self.local_flatten_num_experts, *checkpoint_tensor.shape[1:]),
                dtype=checkpoint_tensor.dtype,
                device=checkpoint_tensor.device,
            )
        local = checkpoint_tensor[start:end]
        if local.shape[0] < self.local_flatten_num_experts:
            # Uneven EP/head partitions require materialized zero padding;
            # divisible production layouts keep the mmap-backed slice above.
            padding = torch.zeros(
                (self.local_flatten_num_experts - local.shape[0], *local.shape[1:]),
                dtype=local.dtype,
                device=local.device,
            )
            local = torch.cat((local, padding), dim=0)
        return local

    def _route(self, x_heads: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = self.gate.view(self.local_num_heads, self.num_experts, self.d_head).float()
        logits = torch.einsum("shd,hed->hse", x_heads.float(), gate)
        bias_source = (os.environ.get("MAGI2_ROUTER_BIAS_SOURCE") or "ema").strip().lower()
        bias_tensor = self.router.expert_bias if bias_source == "main" else self.router.expert_bias_ema
        bias = bias_tensor.view(self.local_num_heads, self.num_experts)
        probs, indices = compute_topk_probs_and_indices(
            logits,
            self.top_k,
            score_func=self.config.score_func,
            expert_bias=bias,
            route_norm=self.config.route_norm,
        )
        return probs * self.config.route_scale, indices

    def _local_forward(self, x_heads: torch.Tensor) -> torch.Tensor:
        global _MATE_MOE_WARNED
        probabilities, indices = self._route(x_heads)
        use_sgl_fused = (
            os.environ.get("MAGI2_USE_SGL_FUSED_MOE", "1") == "1"
            and x_heads.device.type in {"musa", "privateuseone"}
            and x_heads.dtype == torch.bfloat16
            and os.environ.get("MAGI2_DETERMINISTIC", "0") != "1"
        )
        if use_sgl_fused:
            try:
                packed_w13 = None
                # The memory-neutral owned layouts are the qualified MUSA
                # default; set either variable to 0 for an emergency rollback.
                if os.environ.get("MAGI2_SGL_OWNED_W13", "1") == "1":
                    if self._get_owned_w13() is None:
                        self.prepare_owned_w13()
                    packed_w13 = self._get_owned_w13()
                packed_w2 = None
                if os.environ.get("MAGI2_SGL_OWNED_W2", "1") == "1":
                    if self._get_owned_w2() is None:
                        self.prepare_owned_w2()
                    packed_w2 = self._get_owned_w2()
                return _magi2_sgl_fused_moe_forward(
                    x_heads,
                    probabilities,
                    indices,
                    self.W_gate,
                    self.W_up,
                    self.W_down,
                    packed_w13=packed_w13,
                    packed_w2=packed_w2,
                )
            except Exception as exc:
                logger.warning(
                    "SGLang-compatible fused MAGI-2 MoE path failed; "
                    "falling back to the configured route: %s",
                    exc,
                )
        use_mate = (
            os.environ.get("MAGI2_USE_MATE_MOE", "0") == "1"
            and x_heads.device.type == "musa"
            and x_heads.dtype == torch.bfloat16
        )
        sorted_head_ids: torch.Tensor | None = None
        route_counts: torch.Tensor | None = None
        if use_mate:
            (
                gather_ids,
                sorted_probs,
                offsets,
                sorted_head_ids,
                route_counts,
            ) = global_sort_routes_with_head_ids_and_counts(
                probabilities, indices, self.num_experts
            )
        else:
            gather_ids, sorted_probs, offsets = global_sort_routes(
                probabilities, indices, self.num_experts
            )
        if use_mate:
            backend = (os.environ.get("MAGI2_MATE_MOE_BACKEND") or "mubin").strip().lower()
            if backend not in {"auto", "mubin"}:
                raise ValueError(
                    "MAGI2_MATE_MOE_BACKEND must be auto or mubin; "
                    "mutlass does not accept per-expert count metadata"
                )
            if backend == "auto":
                # ``auto`` currently resolves to Mubin for this API. Keep the
                # effective choice explicit because the Mutlass branch treats
                # the same tensor as an MGroupedContiguous row-ID layout.
                backend = "mubin"
            try:
                return mate_bf16_mh_moe_forward(
                    x_heads,
                    gather_ids,
                    sorted_probs,
                    offsets,
                    self.W_gate,
                    self.W_up,
                    self.W_down,
                    backend=backend,
                    sorted_head_ids=sorted_head_ids,
                    token_counts=route_counts,
                )
            except Exception as exc:
                if not _MATE_MOE_WARNED:
                    logger.warning(
                        "MATE BF16 grouped MAGI-2 MoE path failed; falling back "
                        "to the Torch route: %s",
                        exc,
                    )
                    _MATE_MOE_WARNED = True
        # The Triton kernel is currently qualified only on CUDA. Keep MUSA on
        # the numerically equivalent Torch path until a MUSA Triton launch is
        # explicitly enabled and benchmarked.
        if current_omni_platform.is_cuda() and x_heads.device.type == "cuda":
            return triton_mh_moe_forward(
                x_heads,
                gather_ids,
                sorted_probs,
                offsets,
                self.W_gate,
                self.W_up,
                self.W_down,
                deterministic=os.environ.get("MAGI2_DETERMINISTIC", "0") == "1",
            )
        return torch_mh_moe_forward(x_heads, gather_ids, sorted_probs, offsets, self.W_gate, self.W_up, self.W_down)

    def forward(
        self,
        x: torch.Tensor,
        *,
        sequence_split_sizes: list[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.ep_group.world_size > 1 and self.ep_group.replicated_sequence:
            # TP column-parallel ``split_linear`` already emits exactly this
            # rank's contiguous MoE-head slice.  Compute it once and leave it
            # sharded for the row-parallel ``merge_linear``; no token dispatch
            # or head all-gather belongs on the true TP path.
            local_hidden_size = self.local_num_heads * self.d_head
            if x.shape[-1] != local_hidden_size:
                raise ValueError(f"TP-local MAGI MoE input has width {x.shape[-1]}, expected {local_hidden_size}")
            local = x.view(-1, self.local_num_heads, self.d_head)
            output = self._local_forward(local) if self.has_real_moe_heads else torch.zeros_like(local)
            return output.reshape(-1, local_hidden_size)

        x_heads = x.view(-1, self.num_heads, self.d_head)
        if self.ep_pad_heads:
            padding = x_heads.new_zeros((x_heads.shape[0], self.ep_pad_heads, self.d_head))
            x_heads = torch.cat((x_heads, padding), dim=1)
        resolved_split_sizes: list[int] | None = None
        if self.ep_group.world_size > 1:
            # ``sequence_split_sizes`` is normally produced by the SP group
            # (world=8), while the experimental EP group may contain only a
            # contiguous subset (world=4).  Select the entries belonging to
            # this EP group's global ranks before issuing all-to-all.
            global_ranks = getattr(self.ep_group, "global_ranks", None)
            if sequence_split_sizes is None:
                local_size = torch.tensor([x_heads.shape[0]], dtype=torch.int64, device=x_heads.device)
                gathered_sizes = [torch.empty_like(local_size) for _ in range(self.ep_group.world_size)]
                torch.distributed.all_gather(gathered_sizes, local_size, group=self.ep_group.group)
                resolved_split_sizes = [int(size.item()) for size in gathered_sizes]
            elif isinstance(sequence_split_sizes, torch.Tensor):
                if sequence_split_sizes.device.type != "cpu":
                    raise ValueError("sequence_split_sizes tensor must be CPU-resident")
                raw_split_sizes = [int(size) for size in sequence_split_sizes.tolist()]
                if global_ranks is not None and len(raw_split_sizes) != self.ep_group.world_size:
                    resolved_split_sizes = [raw_split_sizes[index] for index in global_ranks]
                else:
                    resolved_split_sizes = raw_split_sizes
            else:
                raw_split_sizes = [int(size) for size in sequence_split_sizes]
                if global_ranks is not None and len(raw_split_sizes) != self.ep_group.world_size:
                    resolved_split_sizes = [raw_split_sizes[index] for index in global_ranks]
                else:
                    resolved_split_sizes = raw_split_sizes
            if (
                len(resolved_split_sizes) != self.ep_group.world_size
                or any(size < 0 for size in resolved_split_sizes)
                or resolved_split_sizes[self.ep_group.rank] != x_heads.shape[0]
            ):
                raise ValueError(
                    "sequence_split_sizes must match the MoE group's local sequence shard"
                )
            x_heads = ep_dispatch(x_heads, self.ep_group, resolved_split_sizes)
        output = self._local_forward(x_heads) if self.has_real_moe_heads else torch.zeros_like(x_heads)
        if self.ep_group.world_size > 1:
            output = ep_undispatch(output, self.ep_group, resolved_split_sizes)
        if self.ep_pad_heads:
            output = output[:, : self.num_heads]
        return output.reshape(-1, self.num_heads * self.d_head)


__all__ = [
    "Magi2MultiHeadMoE",
    "Magi2MultiHeadMoEConfig",
    "compute_topk_probs_and_indices",
    "global_sort_routes",
    "global_sort_routes_with_head_ids",
    "global_sort_routes_with_head_ids_and_counts",
    "torch_mh_moe_forward",
    "mate_bf16_mh_moe_forward",
    "triton_mh_moe_forward",
]
