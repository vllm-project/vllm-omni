# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in AITER tuning for the native MiniMax-H3 BF16 DiT projections."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

import torch
from vllm import envs as vllm_envs
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.platforms import current_platform


@lru_cache(maxsize=256)
def _has_tuned_gemm(m: int, n: int, k: int, device: torch.device) -> bool:
    # AITER owns the GPU/CU/dtype/shape keys and caches its CSV for the worker
    # lifetime. Keep its optional import and native-kernel initialization out
    # of disabled, unsupported, and untuned execution paths.
    from aiter.tuned_gemm import get_GEMM_A16W16_tuned_config

    return (
        get_GEMM_A16W16_tuned_config(m, n, k, False, str(torch.bfloat16), str(torch.bfloat16), False, False) is not None
    )


class MiniMaxH3AiterLinearMethod(UnquantizedLinearMethod):
    """Preserve the original linear method, using AITER only for tuned BF16.

    This is a linear-method adapter, so the layer's forward, tensor-parallel
    collectives, bias-return contract, and any surrounding LoRA wrapper remain
    in control. Weight creation/postprocessing retain the original method.
    """

    def __init__(self, original: UnquantizedLinearMethod) -> None:
        self._original = original

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: object,
    ) -> None:
        self._original.create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._original.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        if (
            torch.is_grad_enabled()
            or torch.compiler.is_compiling()
            or vllm_envs.VLLM_BATCH_INVARIANT
            or not current_platform.is_rocm()
            or x.device.type != "cuda"
            or x.ndim != 2
            or x.dtype != torch.bfloat16
            or not x.is_contiguous()
            or x.numel() == 0
            or bias is not None
            or layer.bias is not None
            or layer.tp_size != 1
            or weight.ndim != 2
            or weight.dtype != torch.bfloat16
            or weight.device != x.device
            or not weight.is_contiguous()
            or getattr(weight, "is_shuffled", False)
            or x.shape[1] != weight.shape[1]
        ):
            return self._original.apply(layer, x, bias)
        if not _has_tuned_gemm(x.shape[0], weight.shape[0], x.shape[1], x.device):
            return self._original.apply(layer, x, bias)

        from aiter.tuned_gemm import gemm_a16w16

        return gemm_a16w16(x, weight, otype=torch.bfloat16)


def enable_minimax_h3_aiter_gemm(blocks: Iterable[torch.nn.Module]) -> None:
    """Install the adapter on native main-DiT QKV/gate-up/down projections."""
    if not current_platform.is_rocm():
        return
    native_layers = {QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear}
    for block in blocks:
        for layer in (block.attn.qkv_proj, block.mlp.fc1, block.mlp.fc2):
            # Never replace quantization or a specialized linear/LoRA layer.
            # Normal LoRA wrappers may still surround the original forward.
            if type(layer) in native_layers and type(layer.quant_method) is UnquantizedLinearMethod:
                layer.quant_method = MiniMaxH3AiterLinearMethod(layer.quant_method)
