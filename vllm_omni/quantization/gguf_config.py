# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF quantization config for diffusion transformers.

Uses dequant+GEMM instead of the fused kernel path (which expects 2D inputs).
"""

from __future__ import annotations

import gguf
import torch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.gguf import (
    UNQUANTIZED_TYPES,
    GGUFConfig,
    GGUFLinearMethod,
    LinearBase,
    QuantizeMethodBase,
    UnquantizedLinearMethod,
    is_layer_skipped_gguf,
)


def dequant_gemm_gguf(x: torch.Tensor, qweight: torch.Tensor, qweight_type: int) -> torch.Tensor:
    if qweight_type in UNQUANTIZED_TYPES:
        return x @ qweight.T
    block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
    shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
    weight = ops.ggml_dequantize(qweight, qweight_type, *shape, x.dtype)
    return x @ weight.T


class DiffusionGGUFLinearMethod(GGUFLinearMethod):
    """GGUF linear method using dequant+GEMM for N-D diffusion tensors."""

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shard_id = getattr(layer.qweight, "shard_id", None)

        if shard_id:
            shard_id = ["q", "k", "v"] if "q" in shard_id else shard_id
            qweight = layer.qweight
            result = []
            for idx in shard_id:
                start, end, offset = layer.qweight.shard_offset_map[idx]
                qweight_type = layer.qweight_type.shard_weight_type[idx]
                result.append(dequant_gemm_gguf(x, qweight[start:end, :offset].contiguous(), qweight_type))
            out = torch.cat(result, axis=-1)
        else:
            qweight = layer.qweight
            qweight_type = layer.qweight_type.weight_type
            out = dequant_gemm_gguf(x, qweight, qweight_type)

        if bias is not None:
            out.add_(bias)
        return out


class DiffusionGGUFConfig(GGUFConfig):
    """GGUF config that carries gguf_model path and uses dequant+GEMM."""

    def __init__(
        self,
        gguf_model: str | None = None,
        unquantized_modules: list[str] | None = None,
    ) -> None:
        super().__init__(unquantized_modules=unquantized_modules or [])
        self.gguf_model = gguf_model

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            if is_layer_skipped_gguf(prefix, self.unquantized_modules, self.packed_modules_mapping):
                return UnquantizedLinearMethod()
            return DiffusionGGUFLinearMethod(self)
        return None
