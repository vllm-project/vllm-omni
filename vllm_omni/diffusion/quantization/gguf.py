# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF quantization config for diffusion transformers."""

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

from .base import DiffusionQuantizationConfig


def dequant_gemm_gguf(x: torch.Tensor, qweight: torch.Tensor, qweight_type: int) -> torch.Tensor:
    if qweight_type in UNQUANTIZED_TYPES:
        return x @ qweight.T
    block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
    shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
    weight = ops.ggml_dequantize(qweight, qweight_type, *shape, x.dtype)
    return x @ weight.T


class DiffusionGGUFLinearMethod(GGUFLinearMethod):
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Dequantize + GEMM path: torch.matmul multiplies over the last
        # dimension and broadcasts leading dimensions, so no 2D flattening
        # is required here.
        shard_id = getattr(layer.qweight, "shard_id", None)

        if shard_id:
            # dequantize shard weights respectively
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


class _GGUFConfig(GGUFConfig):
    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> "QuantizeMethodBase":
        if isinstance(layer, LinearBase):
            if is_layer_skipped_gguf(prefix, self.unquantized_modules, self.packed_modules_mapping):
                return UnquantizedLinearMethod()
            return DiffusionGGUFLinearMethod(self)
        return None


class DiffusionGgufConfig(DiffusionQuantizationConfig):
    """GGUF quantization config for diffusion transformers.

    This is a thin wrapper around vLLM's GGUFConfig and also carries
    the GGUF model reference for loader use.

    Args:
        gguf_model: GGUF model path or HF reference (repo/file or repo:quant_type)
        unquantized_modules: Optional list of module name patterns to skip GGUF
            quantization. Note: diffusion linear layers often use short prefixes
            (e.g., "to_qkv"), so these patterns are matched as substrings.
    """

    quant_config_cls = GGUFConfig

    def __init__(
        self,
        gguf_model: str | None = None,
        unquantized_modules: list[str] | None = None,
    ) -> None:
        self.gguf_model = gguf_model
        self.unquantized_modules = unquantized_modules or []

        self._vllm_config = _GGUFConfig(unquantized_modules=self.unquantized_modules)
