# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Initial TP=1 loader for the official Comfy-layout FLUX.2 NVFP4 weights."""

import torch
from torch import nn
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import set_weight_attrs


class ComfyNvfp4Config(QuantizationConfig):
    def __init__(self, quantized_layers: list[str]):
        super().__init__()
        self.quantized_layers = frozenset(quantized_layers)

    def get_name(self) -> str:
        return "comfy_nvfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 100

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, object]) -> "ComfyNvfp4Config":
        layers = config["quantized_layers"]
        if not isinstance(layers, list) or not all(isinstance(name, str) for name in layers):
            raise ValueError("quantized_layers must list the prepared checkpoint's NVFP4 layers")
        return cls(layers)

    def get_quant_method(self, layer: nn.Module, prefix: str) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        name = prefix.removeprefix("transformer.")
        return ComfyNvfp4LinearMethod() if name in self.quantized_layers else UnquantizedLinearMethod()


class ComfyNvfp4LinearMethod(LinearMethodBase):
    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: object,
    ) -> None:
        if input_size_per_partition != input_size or sum(output_partition_sizes) != output_size:
            raise ValueError("Initial Comfy NVFP4 support requires tensor parallel size 1")
        if input_size % 16:
            raise ValueError("NVFP4 input width must be divisible by 16")
        layer.nvfp4_shape = (output_size, input_size)
        layer.nvfp4_dtype = params_dtype
        for name, shape, dtype in (
            ("weight", (output_size, input_size // 2), torch.uint8),
            ("weight_scale", (output_size, input_size // 16), torch.float8_e4m3fn),
            ("weight_scale_2", (), torch.float32),
            ("input_scale", (), torch.float32),
        ):
            parameter = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
            set_weight_attrs(parameter, {"weight_loader": default_weight_loader})
            layer.register_parameter(name, parameter)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        from comfy_kitchen.tensor import QuantizedTensor, TensorCoreNVFP4Layout

        params = TensorCoreNVFP4Layout.Params(
            scale=layer.weight_scale_2,
            orig_dtype=layer.nvfp4_dtype,
            orig_shape=layer.nvfp4_shape,
            block_scale=layer.weight_scale,
        )
        packed = QuantizedTensor(layer.weight.detach(), "TensorCoreNVFP4Layout", params)
        layer.weight = nn.Parameter(packed, requires_grad=False)

    def apply(self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        from comfy_kitchen.tensor import QuantizedTensor

        quantized = QuantizedTensor.from_float(x, "TensorCoreNVFP4Layout", scale=layer.input_scale)
        return torch.nn.functional.linear(quantized, layer.weight, bias)
