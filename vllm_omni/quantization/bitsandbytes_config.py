# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BitsAndBytes 4-bit quantization config for diffusion transformers.

Supports online (dynamic) NF4/FP4 weight-only quantization from BF16/FP16
checkpoints on CUDA GPUs.
"""

from typing import TYPE_CHECKING, Any, Optional

import torch
from torch.nn import Module
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.fp8 import _copy_missing_attrs
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.model_loader.weight_utils import initialize_single_dummy_weight
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import replace_parameter

from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization.int8_config import LazyWeightMixin

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper


class DiffusionBitsAndBytesConfig(QuantizationConfig):
    """BitsAndBytes 4-bit weight-only config for diffusion transformers.

    Supports online (dynamic) quantization from BF16/FP16 checkpoints.
    Works on CUDA GPUs with the optional ``bitsandbytes`` package installed.
    """

    def __init__(
        self,
        quant_type: str = "nf4",
        compress_statistics: bool = True,
        ignored_layers: list[str] | None = None,
    ) -> None:
        super().__init__()

        if quant_type not in ("nf4", "fp4"):
            raise ValueError(f"Unsupported quant_type {quant_type!r}; expected 'nf4' or 'fp4'")
        self.quant_type = quant_type
        self.compress_statistics = compress_statistics
        self.ignored_layers = ignored_layers or []

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "bitsandbytes"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.ignored_layers is not None:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DiffusionBitsAndBytesConfig":
        quant_type = cls.get_from_keys_or(config, ["quant_type"], "nf4")
        compress_statistics = cls.get_from_keys_or(config, ["compress_statistics"], True)
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)

        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(config, ["modules_to_not_convert"], None)
        return cls(
            quant_type=quant_type,
            compress_statistics=compress_statistics,
            ignored_layers=ignored_layers,
        )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            if current_omni_platform.is_cuda():
                return BnBOnlineLinearMethod(self)
            raise NotImplementedError("BitsAndBytes online quantization is only supported on CUDA.")
        return None


class BnBOnlineLinearMethod(LazyWeightMixin, LinearMethodBase):
    """Online BitsAndBytes 4-bit linear method.

    Loads BF16/FP16 checkpoint weights and quantizes them during loading.
    """

    def __init__(self, quant_config: DiffusionBitsAndBytesConfig):
        self.quant_config = quant_config

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        if layer.weight.device == torch.device("meta"):
            weight = ModelWeightParameter(
                data=torch.empty_like(layer.weight, device=layer._load_device),
                input_dim=1,
                output_dim=0,
                weight_loader=layer.weight.weight_loader,
            )
            _copy_missing_attrs(layer.weight, weight)
            layer.register_parameter("weight", weight)
            initialize_single_dummy_weight(layer.weight)

        import bitsandbytes.functional as bnb_F

        weight = layer.weight.data.contiguous()
        if not weight.is_cuda:
            weight = weight.cuda()

        original_shape = tuple(weight.shape)
        qweight, quant_state = bnb_F.quantize_4bit(
            weight,
            quant_type=self.quant_config.quant_type,
            compress_statistics=self.quant_config.compress_statistics,
        )

        replace_parameter(
            layer,
            "weight",
            torch.nn.Parameter(qweight, requires_grad=False),
        )
        layer.quant_state = quant_state
        layer.bnb_shape = original_shape

        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        import bitsandbytes as bnb

        ori_shape = x.shape
        ori_dtype = x.dtype
        x_2d = x.reshape(-1, ori_shape[-1])

        out = bnb.matmul_4bit(
            x_2d,
            layer.weight.t(),
            quant_state=layer.quant_state,
            bias=bias,
        )
        return out.reshape(*ori_shape[:-1], -1).to(ori_dtype)
