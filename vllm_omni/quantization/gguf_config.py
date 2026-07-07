# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF quantization config for diffusion transformers.

Uses dequant+GEMM instead of the fused kernel path (which expects 2D inputs).

The GGUF quantization module was migrated to an external plugin in upstream
vLLM (commit 6635279d8).  This file inlines the base classes that
DiffusionGGUFConfig / DiffusionGGUFLinearMethod depend on, reproduced from
the last upstream version before the migration.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import gguf
import torch
from gguf import GGMLQuantizationType as WeightType
from torch.nn.parameter import Parameter, UninitializedParameter
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# GGUF base types (inlined from upstream vllm before migration to plugin)
# ---------------------------------------------------------------------------

UNQUANTIZED_TYPES: set[WeightType] = {WeightType.F32, WeightType.F16, WeightType.BF16}

STANDARD_QUANT_TYPES: set[WeightType] = {
    WeightType.Q4_0,
    WeightType.Q4_1,
    WeightType.Q5_0,
    WeightType.Q5_1,
    WeightType.Q8_0,
    WeightType.Q8_1,
}
KQUANT_TYPES: set[WeightType] = {
    WeightType.Q2_K,
    WeightType.Q3_K,
    WeightType.Q4_K,
    WeightType.Q5_K,
    WeightType.Q6_K,
}
IMATRIX_QUANT_TYPES: set[WeightType] = {
    WeightType.IQ1_M,
    WeightType.IQ1_S,
    WeightType.IQ2_XXS,
    WeightType.IQ2_XS,
    WeightType.IQ2_S,
    WeightType.IQ3_XXS,
    WeightType.IQ3_S,
    WeightType.IQ4_XS,
    WeightType.IQ4_NL,
}
DEQUANT_TYPES: set[WeightType] = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES


def is_layer_skipped_gguf(
    prefix: str,
    unquantized_modules: list[str],
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({}),
) -> bool:
    # Fused layers like gate_up_proj or qkv_proj will not be fused
    # in the safetensors checkpoint. So, we convert the name
    # from the fused version to unfused + check to make sure that
    # each shard of the fused layer has the same scheme.
    proj_name = prefix.split(".")[-1]
    if proj_name in fused_mapping:
        shard_prefixes = [prefix.replace(proj_name, shard_proj_name) for shard_proj_name in fused_mapping[proj_name]]

        is_skipped = None
        for shard_prefix in shard_prefixes:
            is_shard_skipped = any(shard_prefix in module_name for module_name in unquantized_modules)

            if is_skipped is None:
                is_skipped = is_shard_skipped
            elif is_shard_skipped != is_skipped:
                raise ValueError(
                    f"Detected some but not all shards of {prefix} "
                    "are quantized. All shards of fused layers "
                    "to have the same precision."
                )
    else:
        is_skipped = any(module_name in prefix for module_name in unquantized_modules)

    assert is_skipped is not None
    return is_skipped


class _GGUFUninitializedParameter(UninitializedParameter):
    """UninitializedParameter subclass that stores GGUF weight data in a list.

    The actual tensor is materialized later via _create_padded_weight_param.
    """

    cls_to_become = Parameter
    data_container: list[torch.Tensor]


class GGUFConfig(QuantizationConfig):
    """Config class for GGUF (inlined from upstream vllm)."""

    def __init__(self, unquantized_modules: list[str] | None = None) -> None:
        super().__init__()
        self.unquantized_modules = unquantized_modules or []

    def __repr__(self) -> str:
        return "GGUFConfig()"

    def get_name(self) -> QuantizationMethods:
        return "gguf"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []  # no extra configs.

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> GGUFConfig:
        return cls()

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg: dict[str, Any], user_quant: str | None, hf_config=None
    ) -> QuantizationMethods | None:
        if user_quant == "gguf":
            return "gguf"
        return None

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            if is_layer_skipped_gguf(prefix, self.unquantized_modules, self.packed_modules_mapping):
                return UnquantizedLinearMethod()
            return GGUFLinearMethod(self)
        return None


class GGUFLinearMethod(LinearMethodBase):
    """Linear method for GGUF (inlined from upstream vllm)."""

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        qweight = _GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            },
        )
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        qweight_type = Parameter(
            torch.empty(len(output_partition_sizes), dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight_type,
            {
                "is_gguf_weight_type": True,
                "weight_type": 0,
                "shard_weight_type": {},
                "ignore_warning": True,
            },
        )
        set_weight_attrs(qweight_type, extra_weight_attrs)
        layer.register_parameter("qweight_type", qweight_type)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        qweight_type = layer.qweight_type.weight_type
        if not (qweight_type in UNQUANTIZED_TYPES or qweight_type in DEQUANT_TYPES):
            qweight_type = WeightType(qweight_type)
            raise ValueError(f"Unsupported GGUF quantization type {qweight_type} in layer {layer}.")
        self._create_padded_weight_param(layer)

    def _create_padded_weight_param(self, layer: torch.nn.Module):
        """Create padded weight parameter for GGUF MergedLinear layer."""
        qweight = layer.qweight
        shard_id_map = qweight.shard_id_map
        shard_id = qweight.shard_id
        if len(data_container := qweight.data_container) > 1:
            dtype = {data.dtype for data in data_container}
            assert len(dtype) == 1, ValueError(f"Data container has mixed dtypes: {dtype}")
            dtype = next(iter(dtype))
            padded_side = max(x.size(1) for x in data_container)
            concat_side = sum(x.size(0) for x in data_container)
            padded_data = torch.zeros((concat_side, padded_side), dtype=dtype, device=qweight.device)
            shard_offset_map = dict[str, tuple[int, int, int]]()
            for idx in shard_id:
                id_in_container = shard_id_map[idx]
                start = sum(x.size(0) for x in data_container[:id_in_container])
                end = start + data_container[id_in_container].size(0)
                size = data_container[id_in_container].size(1)
                padded_data[start:end, :size] = data_container[id_in_container]
                shard_offset_map[idx] = (start, end, size)
            qweight.data_container.clear()
            padded_param = Parameter(padded_data, requires_grad=False)
            set_weight_attrs(padded_param, vars(qweight))
            set_weight_attrs(padded_param, {"shard_offset_map": shard_offset_map})
            layer.register_parameter("qweight", padded_param)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Base apply: overridden by DiffusionGGUFLinearMethod.
        raise NotImplementedError("Use DiffusionGGUFLinearMethod for diffusion GGUF")


# ---------------------------------------------------------------------------
# Diffusion-specific GGUF (dequant+GEMM, replaces fused kernel path)
# ---------------------------------------------------------------------------


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
                result.append(
                    dequant_gemm_gguf(
                        x,
                        qweight[start:end, :offset].contiguous(),
                        qweight_type,
                    )
                )
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
