# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Routed-expert (FusedMoE) discovery checks and dispatch for Cosmos3 mixed precision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8MoEMethod

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput

    from .runtime import Cosmos3MixedPrecisionRuntime, PrecisionPath

SUPPORTED_BACKENDS: frozenset[Fp8MoeBackend] = frozenset({Fp8MoeBackend.TRITON, Fp8MoeBackend.FLASHINFER_CUTLASS})


@dataclass(frozen=True)
class RecordedMoE:
    """One routed-expert layer found by ``install``."""

    routed_experts: torch.nn.Module
    base_method: ModelOptFp8MoEMethod
    module_name: str
    path: PrecisionPath


def ensure_supported_moe_method(method: object, module_name: str) -> None:
    """Reject routed experts whose execution MAPS cannot reproduce in A16."""
    if type(method) is not ModelOptFp8MoEMethod:
        raise ValueError(
            f"Cosmos3 mixed precision supports only ModelOpt FP8 routed experts; "
            f"{module_name} uses {type(method).__name__}"
        )
    if method.fp8_backend not in SUPPORTED_BACKENDS:
        supported = ", ".join(sorted(backend.name for backend in SUPPORTED_BACKENDS))
        raise ValueError(
            f"Cosmos3 mixed precision does not support FP8 MoE backend "
            f"{method.fp8_backend.name} for {module_name}; supported: {supported}"
        )
    # Before weight loading this falls back to ``experts_cls.is_monolithic()``.
    if method.is_monolithic:
        raise ValueError(
            f"Cosmos3 mixed precision does not support monolithic MoE kernels ({module_name}); "
            "they route inside the kernel"
        )
    parallel = method.moe.moe_parallel_config
    if parallel.use_ep or parallel.ep_size != 1:
        raise ValueError(f"Cosmos3 mixed precision does not support expert parallel routed experts ({module_name})")
    if parallel.enable_eplb:
        raise ValueError(f"Cosmos3 mixed precision does not support EPLB for routed experts ({module_name})")
    if parallel.tp_size != 1 or parallel.dp_size != 1:
        raise ValueError(
            f"Cosmos3 mixed precision does not support tensor/data parallel routed experts "
            f"({module_name}: tp_size={parallel.tp_size}, dp_size={parallel.dp_size})"
        )


def _expert_scale(layer: torch.nn.Module, name: str, num_experts: int, device: torch.device) -> torch.Tensor:
    scale = getattr(layer, name)
    if scale.numel() != num_experts:
        raise ValueError(
            f"Cosmos3 mixed precision needs one {name} per expert ({num_experts}), got shape {tuple(scale.shape)}"
        )
    values = scale.detach().to(device=device, dtype=torch.float32)
    if not torch.isfinite(values).all() or not (values > 0).all():
        raise ValueError(f"Cosmos3 mixed precision found a non-finite or non-positive {name}")
    return values.reshape(num_experts, 1, 1)


def dequantize_fp8_experts(layer: torch.nn.Module, backend: Fp8MoeBackend) -> tuple[torch.Tensor, torch.Tensor]:
    """Rebuild BF16 ``w13`` ([E, 2I, H], [gate; up]) and ``w2`` ([E, H, I]) from the live FP8 experts."""
    # Read on every call: FSDP replaces these with freshly gathered tensors.
    w13 = layer.w13_weight
    w2 = layer.w2_weight
    num_experts = w13.shape[0]
    w13_scale = _expert_scale(layer, "w13_weight_scale", num_experts, w13.device)
    w2_scale = _expert_scale(layer, "w2_weight_scale", num_experts, w2.device)
    if backend == Fp8MoeBackend.FLASHINFER_CUTLASS:
        # vLLM stores FlashInfer's w13 as [up; gate] (swap_w13_to_w31).
        up, gate = w13.chunk(2, dim=1)
        w13 = torch.cat([gate, up], dim=1)
    # Same rounding as the linear materialize; in-place to save memory.
    return (
        w13.to(torch.float32, copy=True).mul_(w13_scale).to(torch.bfloat16),
        w2.to(torch.float32, copy=True).mul_(w2_scale).to(torch.bfloat16),
    )


def apply_high_moe(
    layer: torch.nn.Module,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    backend: Fp8MoeBackend,
) -> torch.Tensor:
    """Run the routed experts in A16 from the layer's live FP8 weights."""
    # vLLM defines fused_experts only when Triton is available.
    from vllm.model_executor.layers.fused_moe import fused_experts

    w13, w2 = dequantize_fp8_experts(layer, backend)
    return fused_experts(
        x,
        w13,
        w2,
        topk_weights,
        topk_ids,
        activation=layer.activation,
        apply_router_weight_on_input=layer.apply_router_weight_on_input,
        global_num_experts=layer.global_num_experts,
        expert_map=layer.expert_map,
        quant_config=None,
    )


def install_moe_dispatch(
    method: ModelOptFp8MoEMethod,
    runtime: Cosmos3MixedPrecisionRuntime,
    path: PrecisionPath,
    module_name: str,
) -> None:
    """Dispatch one routed-expert layer between vLLM's FP8 method and A16."""
    if "apply" in vars(method):
        raise RuntimeError(f"Cosmos3 mixed precision already overrides apply for {module_name}")

    # The MoE weight loader branches on the method's class name, so the method
    # cannot be wrapped like the linears; override `apply` on this one object instead.
    def apply(
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: torch.nn.Module | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | UnfinalizedMoEOutput:
        # Shared experts are OK as long as they don't run inside the kernel.
        if shared_experts is not None and method.mk_can_overlap_shared_experts:
            raise NotImplementedError(
                f"Cosmos3 mixed precision does not support shared experts run inside the MoE kernel ({module_name})"
            )
        if not runtime.use_high_precision(path):
            return type(method).apply(method, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input)
        return apply_high_moe(layer, x, topk_weights, topk_ids, method.fp8_backend)

    method.apply = apply
