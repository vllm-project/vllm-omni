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

import math
import os
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.triton_utils import HAS_TRITON, tl, triton

from vllm_omni.platforms import current_omni_platform

from .fused_moe_kernels import (
    global_sort_routes,
    invoke_fused_moe_bf16,
    torch_mh_moe_forward,
    triton_mh_moe_forward,
)
from .parallel import Magi2ParallelGroup, ep_dispatch, ep_undispatch, get_magi2_ep_group

try:  # Triton's precise exp; ``tl.exp`` lowers to the 29-ULP ex2 approximation.
    from triton.language.extra import libdevice as _tl_libdevice
except ImportError:  # pragma: no cover - non-CUDA Triton builds
    _tl_libdevice = None

_HAS_PRECISE_EXP = _tl_libdevice is not None and hasattr(_tl_libdevice, "exp")

RoutingScore = Literal["softmax", "sigmoid"]
_BF16_MOE_CUDA_MIN_TOKENS = 4096


def _reference_topk_probs_and_indices(
    router_logits: torch.Tensor,
    top_k: int,
    *,
    score_func: RoutingScore = "sigmoid",
    expert_bias: torch.Tensor | None = None,
    route_norm: bool = True,
    norm_eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unfused reference routing.  Also the oracle the fused path is tested against."""

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


# Selection-score bounds: the finite fp32 range, so that -inf is reserved for
# lanes the top-k loop has already consumed and +inf for NaN lanes, which
# torch.topk ranks above everything.  Guarded like the SwiGLU7 constants in
# ``fused_moe_kernels``, since the placeholder's ``tl.constexpr`` is ``None``.
_MIN_FINITE_FP32 = tl.constexpr(-3.4028234663852886e38) if HAS_TRITON else -3.4028234663852886e38
_MAX_FINITE_FP32 = tl.constexpr(3.4028234663852886e38) if HAS_TRITON else 3.4028234663852886e38


@triton.jit
def _routing_topk_kernel(
    logits_ptr,
    bias_ptr,
    probs_ptr,
    indices_ptr,
    num_tokens,
    stride_logits_h,
    stride_logits_s,
    tiles_per_head,
    top_k: tl.constexpr,
    top_k_pad: tl.constexpr,
    num_experts: tl.constexpr,
    experts_pad: tl.constexpr,
    block_t: tl.constexpr,
    has_bias: tl.constexpr,
    route_norm: tl.constexpr,
    norm_eps: tl.constexpr,
    precise_exp: tl.constexpr,
):
    """Sigmoid, selection bias, top-k, probability gather and L1 norm in one pass.

    One program owns ``block_t`` ``[head, token]`` rows and keeps the whole
    expert bank in registers, so the ``[heads,tokens,experts]`` logits are read
    exactly once instead of once per routing stage.
    """

    tile = tl.program_id(0)
    head = tile // tiles_per_head
    token_tile = tile % tiles_per_head
    token_offsets = token_tile * block_t + tl.arange(0, block_t)
    expert_offsets = tl.arange(0, experts_pad)
    token_mask = token_offsets < num_tokens
    expert_mask = expert_offsets < num_experts
    live = token_mask[:, None] & expert_mask[None, :]

    # Head and token indices fit in int32, but scaling them by the per-head
    # logit stride does not once the packed sequence gets long.  Promote before
    # computing element offsets, as the expert kernel does.
    logits_base = head.to(tl.int64) * stride_logits_h + token_offsets.to(tl.int64) * stride_logits_s
    logits = tl.load(
        logits_ptr + logits_base[:, None] + expert_offsets[None, :],
        mask=live,
        other=0.0,
    )
    # tl.sigmoid and tl.exp lower to the ex2 approximation, which drifts up to
    # 29 ULP from torch.sigmoid and can reorder near-equal selection scores.
    # libdevice's exp keeps the fused route within one ULP of the reference.
    if precise_exp:
        router_scores = 1.0 / (1.0 + _tl_libdevice.exp(-logits))
    else:
        router_scores = 1.0 / (1.0 + tl.exp(-logits))
    if has_bias:
        bias = tl.load(bias_ptr + head * num_experts + expert_offsets, mask=expert_mask, other=0.0)
        # The loop below retires a winner by marking it -inf, and NaN lanes are
        # folded to +inf, so both infinities must stay out of reach for a
        # finite score.  A sigmoid is inside [0, 1], so the bias is the only way
        # in: clamp it to the finite range, over the [experts] bank rather than
        # the whole score tile.  The clamp would swallow a NaN bias, which the
        # reference ranks first, so that passes through untouched.
        clamped_bias = tl.minimum(tl.maximum(bias, _MIN_FINITE_FP32), _MAX_FINITE_FP32)
        selection_scores = router_scores + tl.where(bias == bias, clamped_bias, bias)[None, :]
    else:
        selection_scores = router_scores
    # tl.max does not propagate NaN, so a NaN row would match no live lane and
    # hand the padded sentinel id out of the kernel.  torch.topk ranks NaN above
    # every number; +inf is otherwise unreachable, so folding NaN onto it keeps
    # that order and the ids in range.  The route weight is still read from
    # ``router_scores``, so a NaN logit reaches the output as NaN, as it does in
    # the reference.  Dead and padded lanes keep -inf and stay unreachable for
    # good, which is sound because ``top_k <= num_experts`` leaves an unretired
    # live lane in every round.
    selection_scores = tl.where(selection_scores == selection_scores, selection_scores, float("inf"))
    selection_scores = tl.where(live, selection_scores, float("-inf"))

    route_offsets = tl.arange(0, top_k_pad)
    topk_probs = tl.zeros([block_t, top_k_pad], dtype=tl.float32)
    topk_indices = tl.zeros([block_t, top_k_pad], dtype=tl.int32)
    l1_norm = tl.zeros([block_t], dtype=tl.float32)
    for route in tl.static_range(top_k):
        best_score = tl.max(selection_scores, axis=1)
        # tl.argmax is several times more expensive than a plain max on this
        # shape, so recover the winner with a second reduction.  Ties resolve to
        # the lowest expert id, which torch.topk leaves unspecified.  A retired
        # expert sits at -inf, strictly below the clamp above, so it can never
        # match ``best_score`` and be routed to twice.  Scores are NaN-free here,
        # so ``best_score`` always matches a live lane.
        best_expert = tl.min(tl.where(selection_scores == best_score[:, None], expert_offsets, experts_pad), axis=1)
        selected = expert_offsets[None, :] == best_expert[:, None]
        # The bias steers selection only; the route weight is the unbiased score.
        probability = tl.sum(tl.where(selected, router_scores, 0.0), axis=1)
        is_route = route_offsets == route
        topk_probs += tl.where(is_route[None, :], probability[:, None], 0.0)
        topk_indices += tl.where(is_route[None, :], best_expert[:, None].to(tl.int32), 0)
        selection_scores = tl.where(selected, float("-inf"), selection_scores)
        l1_norm += tl.abs(probability)
    if route_norm:
        # F.normalize clamps with clamp_min, which propagates a NaN norm; the
        # default tl.maximum would replace it with eps and blow the finite
        # weights of a partly NaN row up to ~1e12 instead.
        topk_probs = topk_probs / tl.maximum(l1_norm, norm_eps, propagate_nan=tl.PropagateNan.ALL)[:, None]

    store_mask = token_mask[:, None] & (route_offsets[None, :] < top_k)
    store_base = (head.to(tl.int64) * num_tokens + token_offsets.to(tl.int64)) * top_k
    store_offsets = store_base[:, None] + route_offsets[None, :]
    tl.store(probs_ptr + store_offsets, topk_probs, mask=store_mask)
    tl.store(indices_ptr + store_offsets, topk_indices.to(tl.int64), mask=store_mask)


# Above this the [block_t, experts_pad] score tiles no longer fit in registers
# and the fused kernel loses to the unfused reference.
_MAX_FUSED_EXPERTS_PAD = 1024


def _fused_routing_config(experts_pad: int) -> tuple[int, int]:
    """Return ``(block_t, num_warps)`` for a padded expert-bank width."""

    # Two fp32 [block_t, experts_pad] tiles per program; one warp per 256-wide
    # slice keeps the six top-k reductions inside warp shuffles.
    num_warps = max(1, min(8, experts_pad // 256))
    block_t = max(1, 256 // experts_pad)
    return block_t, num_warps


def _fused_routing_supported(
    router_logits: torch.Tensor,
    score_func: RoutingScore,
    expert_bias: torch.Tensor | None,
) -> bool:
    if not router_logits.is_cuda or score_func != "sigmoid":
        return False
    # fp32 only: the reference evaluates sigmoid and the L1 norm in the logit
    # dtype, and reproducing narrow-dtype rounding is not worth a second kernel.
    if router_logits.dtype != torch.float32:
        return False
    if router_logits.stride(-1) != 1 or router_logits.shape[1] == 0:
        return False
    if triton.next_power_of_2(router_logits.shape[-1]) > _MAX_FUSED_EXPERTS_PAD:
        return False
    # Only contiguity is load bearing: the reference reshapes the bias with
    # ``view``, and so does :func:`_dense_expert_bias`.  Narrow dtypes and the
    # broadcast shapes the reference accepts are normalized there instead of
    # costing the whole fused path.
    if expert_bias is not None and not expert_bias.is_contiguous():
        return False
    return True


def _dense_expert_bias(expert_bias: torch.Tensor, heads: int, num_experts: int) -> torch.Tensor:
    """Return the dense fp32 ``[heads, num_experts]`` bank the kernel indexes.

    The reference broadcasts the bias with ``expert_bias.view(heads, 1, -1)``, so
    a per-head scalar is legal input, while the kernel reads ``heads *
    num_experts`` elements.  Materialize that broadcast rather than drop to the
    reference for a bank this small, and upcast narrow dtypes, which is exactly
    the promotion the reference gets from adding the bias to fp32 scores.

    ``view`` and ``expand`` reject the shapes the reference's own broadcast
    rejects, ``contiguous`` is what actually retires the zero stride ``expand``
    leaves behind, and every step is a no-op for a dense fp32 bias.
    """

    dense = expert_bias.view(heads, -1).expand(heads, num_experts)
    return dense.to(torch.float32).contiguous()


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

    Supported CUDA inputs run as a single fused kernel; other devices, logit
    dtypes, score functions and shapes fall back to
    :func:`_reference_topk_probs_and_indices`.  The two pick the same experts in
    the same order whenever the top ``top_k + 1`` selection scores of a row are
    separated by more than a few ULP, and the returned weights then agree to
    about 1e-6 relative: the fused sigmoid is within one ULP of
    ``torch.sigmoid`` and the folded L1 normalization sums in a different order.
    Under an exact tie the fused kernel selects the lowest expert id, an order
    ``torch.topk`` leaves unspecified.  NaN selection scores rank first, as in
    ``torch.topk``, and a NaN logit makes its row's weights NaN in both paths.
    """

    if router_logits.ndim != 3:
        raise ValueError("router_logits must be [heads,tokens,experts]")
    if not 0 < top_k <= router_logits.shape[-1]:
        raise ValueError("top_k must be in [1, num_experts]")
    if not _fused_routing_supported(router_logits, score_func, expert_bias):
        return _reference_topk_probs_and_indices(
            router_logits,
            top_k,
            score_func=score_func,
            expert_bias=expert_bias,
            route_norm=route_norm,
            norm_eps=norm_eps,
        )

    heads, num_tokens, num_experts = router_logits.shape
    if expert_bias is not None:
        expert_bias = _dense_expert_bias(expert_bias, heads, num_experts)
    experts_pad = triton.next_power_of_2(num_experts)
    block_t, num_warps = _fused_routing_config(experts_pad)
    tiles_per_head = triton.cdiv(num_tokens, block_t)
    topk_probs = torch.empty((heads, num_tokens, top_k), device=router_logits.device, dtype=torch.float32)
    topk_indices = torch.empty((heads, num_tokens, top_k), device=router_logits.device, dtype=torch.int64)
    _routing_topk_kernel[(heads * tiles_per_head,)](
        router_logits,
        expert_bias,
        topk_probs,
        topk_indices,
        num_tokens,
        router_logits.stride(0),
        router_logits.stride(1),
        tiles_per_head,
        top_k,
        triton.next_power_of_2(top_k),
        num_experts,
        experts_pad,
        block_t,
        expert_bias is not None,
        route_norm,
        norm_eps,
        _HAS_PRECISE_EXP,
        num_warps=num_warps,
        num_stages=1,
    )
    return topk_probs, topk_indices


def _align_bf16_routes(
    route_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build route metadata without reading the padded count back to the CPU."""

    flat_ids = route_ids.reshape(-1).to(torch.int32)
    route_count = flat_ids.numel()
    order = torch.argsort(flat_ids)
    sorted_experts = flat_ids[order]
    counts = torch.zeros(num_experts, device=flat_ids.device, dtype=torch.int64)
    counts.scatter_add_(0, flat_ids.long(), torch.ones_like(flat_ids, dtype=torch.int64))
    padded_counts = ((counts + block_size - 1) // block_size) * block_size
    starts = torch.cumsum(padded_counts, 0) - padded_counts
    ends = torch.cumsum(counts, 0)
    begins = ends - counts
    positions = torch.arange(route_count, device=flat_ids.device, dtype=torch.int64)
    destinations = starts[sorted_experts.long()] + positions - begins[sorted_experts.long()]
    if buffers is None:
        buffers = _allocate_bf16_route_buffers(route_count, num_experts, block_size, flat_ids.device)
    sorted_ids, expert_ids, num_padded = buffers
    sorted_ids.fill_(route_count)
    sorted_ids[destinations] = order.to(torch.int32)
    block_starts = torch.arange(expert_ids.numel(), device=flat_ids.device) * block_size
    expert_ids.copy_(torch.searchsorted(padded_counts.cumsum(0), block_starts, right=True))
    num_padded.copy_(padded_counts.sum().reshape(1))
    return sorted_ids, expert_ids, num_padded


def _allocate_bf16_route_buffers(
    route_count: int, num_experts: int, block_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Each expert adds at most block_size - 1 padding rows. The kernel uses
    # the device-side padded count to skip programs beyond the live prefix.
    capacity = math.ceil((route_count + num_experts * (block_size - 1)) / block_size) * block_size
    return (
        torch.empty(capacity, device=device, dtype=torch.int32),
        torch.empty(capacity // block_size, device=device, dtype=torch.int32),
        torch.empty(1, device=device, dtype=torch.int32),
    )


def _pack_bf16_w13(w_gate: torch.Tensor, w_up: torch.Tensor) -> torch.Tensor:
    """Pack gate/up weights into adjacent rows for the grouped W13 GEMM."""

    num_experts, hidden_size, intermediate_size = w_gate.shape
    return torch.stack((w_gate.transpose(1, 2), w_up.transpose(1, 2)), dim=2).reshape(
        num_experts, 2 * intermediate_size, hidden_size
    )


def _bf16_fused_moe_forward(
    x_heads: torch.Tensor,
    probabilities: torch.Tensor,
    indices: torch.Tensor,
    packed_w13: torch.Tensor,
    w_down: torch.Tensor,
    route_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Run the BF16 grouped GEMM path with head-local routing."""

    num_tokens, num_heads, hidden_size = x_heads.shape
    top_k = probabilities.shape[-1]
    num_experts = packed_w13.shape[0]
    if num_experts % num_heads:
        raise ValueError("expert bank must be divisible by local MoE heads")
    experts_per_head = num_experts // num_heads
    if num_tokens == 0:
        return torch.zeros_like(x_heads)
    head_offsets = (
        torch.arange(num_heads, device=x_heads.device, dtype=torch.int32).view(num_heads, 1, 1) * experts_per_head
    )
    route_ids = (indices.to(torch.int32) + head_offsets).reshape(num_heads * num_tokens, top_k)
    route_weights = probabilities.reshape(num_heads * num_tokens, top_k).contiguous()
    sorted_ids, expert_ids, num_padded = _align_bf16_routes(route_ids, num_experts, 128, route_buffers)

    hidden = x_heads.permute(1, 0, 2).contiguous().reshape(num_heads * num_tokens, hidden_size)
    intermediate_size = packed_w13.shape[1] // 2
    intermediate = torch.empty(
        (num_heads * num_tokens * top_k, intermediate_size), device=x_heads.device, dtype=x_heads.dtype
    )
    # Keep the qualified MUSA point unchanged.  CUDA/H20 benefits from the
    # pre-Blackwell tile found by the MAGI-2 BF16 sweep (smaller K/warp count
    # and deeper pipelining reduce register pressure).  The launch contract
    # remains the same; only legal, device-specific values are selected.
    config = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 16,
        "num_warps": 4 if not current_omni_platform.is_musa() else 16,
        "num_stages": 3 if not current_omni_platform.is_musa() else 1,
    }
    invoke_fused_moe_bf16(
        hidden,
        packed_w13,
        intermediate,
        route_weights,
        sorted_ids,
        expert_ids,
        num_padded,
        top_k=top_k,
        config=config,
        fuse_swiglu=True,
    )
    route_output = torch.empty((num_heads * num_tokens, top_k, hidden_size), device=x_heads.device, dtype=x_heads.dtype)
    invoke_fused_moe_bf16(
        intermediate,
        w_down.transpose(1, 2),
        route_output,
        route_weights,
        sorted_ids,
        expert_ids,
        num_padded,
        top_k=1,
        config=config,
        fuse_swiglu=False,
    )
    return route_output.sum(dim=1).reshape(num_heads, num_tokens, hidden_size).permute(1, 0, 2)


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

        # Derived, non-persistent weight storage.  The model loader populates
        # it after checkpoint loading; the lazy guard also supports direct
        # layer construction and later weight reloads in tests/tools.
        self.register_buffer("_bf16_packed_w13", None, persistent=False)
        self._bf16_packed_w13_key: tuple | None = None
        self._bf16_route_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def _apply(self, fn, recurse: bool = True):
        # Do not migrate an extra full-size derived weight bank on .to().
        self._bf16_packed_w13 = None
        self._bf16_packed_w13_key = None
        self._bf16_route_buffers = None
        return super()._apply(fn, recurse=recurse)

    def _get_bf16_packed_w13(self) -> torch.Tensor:
        # In-place loading increments _version; mmap/.data replacement may
        # not. Include object/storage identity and handle inference tensors,
        # which intentionally do not expose a version counter.
        key = tuple(
            (id(weight), weight.data_ptr(), None if weight.is_inference() else weight._version)
            for weight in (self.W_gate, self.W_up)
        )
        packed = self._bf16_packed_w13
        if (
            packed is None
            or self._bf16_packed_w13_key != key
            or packed.device != self.W_gate.device
            or packed.dtype != self.W_gate.dtype
        ):
            # [E, I, 2, D] -> [E, 2I, D]: adjacent rows are gate/up pairs.
            with torch.no_grad():
                packed = _pack_bf16_w13(self.W_gate, self.W_up)
            self._bf16_packed_w13 = packed
            self._bf16_packed_w13_key = key
        return packed

    def prepare_bf16_weights(self) -> None:
        """Materialize derived BF16 weights outside the inference hot path."""

        # Reload may mutate inference tensors without a version counter.
        self._bf16_packed_w13 = None
        self._bf16_packed_w13_key = None
        if self.W_gate.dtype == torch.bfloat16 and self.W_gate.device.type not in ("cpu", "meta"):
            self._get_bf16_packed_w13()

    def _get_bf16_route_buffers(self, route_count: int, device: torch.device):
        buffers = self._bf16_route_buffers
        capacity = math.ceil((route_count + self.local_flatten_num_experts * 127) / 128) * 128
        if buffers is None or buffers[0].numel() != capacity or buffers[0].device != device:
            buffers = _allocate_bf16_route_buffers(route_count, self.local_flatten_num_experts, 128, device)
            self._bf16_route_buffers = buffers
        return buffers

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
        probabilities, indices = self._route(x_heads)
        if (
            x_heads.device.type in ("cuda", "musa")
            and x_heads.dtype == torch.bfloat16
            and os.environ.get("MAGI2_DETERMINISTIC", "0") != "1"
            and (current_omni_platform.is_musa() or x_heads.shape[0] >= _BF16_MOE_CUDA_MIN_TOKENS)
            and not torch.compiler.is_compiling()
        ):
            return _bf16_fused_moe_forward(
                x_heads,
                probabilities,
                indices,
                self._get_bf16_packed_w13(),
                self.W_down,
                self._get_bf16_route_buffers(x_heads.shape[0] * self.local_num_heads * self.top_k, x_heads.device),
            )
        gather_ids, sorted_probs, offsets = global_sort_routes(probabilities, indices, self.num_experts)
        if x_heads.is_cuda:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        sequence_split_sizes: list[int] | None = None
        if self.ep_group.world_size > 1:
            local_size = torch.tensor([x_heads.shape[0]], dtype=torch.int64, device=x_heads.device)
            gathered_sizes = [torch.empty_like(local_size) for _ in range(self.ep_group.world_size)]
            torch.distributed.all_gather(gathered_sizes, local_size, group=self.ep_group.group)
            sequence_split_sizes = [int(size.item()) for size in gathered_sizes]
            x_heads = ep_dispatch(x_heads, self.ep_group, sequence_split_sizes)
        output = self._local_forward(x_heads) if self.has_real_moe_heads else torch.zeros_like(x_heads)
        if self.ep_group.world_size > 1:
            output = ep_undispatch(output, self.ep_group, sequence_split_sizes)
        if self.ep_pad_heads:
            output = output[:, : self.num_heads]
        return output.reshape(-1, self.num_heads * self.d_head)


__all__ = [
    "Magi2MultiHeadMoE",
    "Magi2MultiHeadMoEConfig",
    "compute_topk_probs_and_indices",
    "global_sort_routes",
    "torch_mh_moe_forward",
    "triton_mh_moe_forward",
]
