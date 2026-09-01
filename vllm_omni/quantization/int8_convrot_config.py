# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""ComfyUI tensor-wise INT8 ConvRot checkpoint support.

The on-disk format stores an already rotated, row-wise quantized weight and a
per-output-row scale.  At runtime the activation must receive the matching
Hadamard rotation before dynamic INT8 quantization; treating these weights as
ordinary INT8 silently produces incorrect output.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import Module
from vllm.logger import init_logger
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
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.parameter import ChannelQuantScaleParameter

from vllm_omni.quantization.int8_config import create_weight_parameter

logger = init_logger(__name__)

_FORMAT = "int8_tensorwise"


@dataclass(frozen=True)
class Int8ConvRotLayerConfig:
    """Validated per-linear metadata decoded from ``.comfy_quant``."""

    convrot: bool
    convrot_groupsize: int = 256

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Int8ConvRotLayerConfig:
        fmt = value.get("format")
        if fmt != _FORMAT:
            raise ValueError(f"Unsupported Comfy quant format {fmt!r}; expected {_FORMAT!r}.")
        convrot = value.get("convrot", False)
        if not isinstance(convrot, bool):
            raise ValueError(f"Comfy ConvRot flag must be boolean, got {convrot!r}.")
        group_size = value.get("convrot_groupsize", 256)
        if not isinstance(group_size, int) or group_size < 4:
            raise ValueError(f"Comfy ConvRot group size must be an integer >= 4, got {group_size!r}.")
        # comfy-kitchen's regular Hadamard is defined for powers of four.
        value_left = group_size
        while value_left > 1 and value_left % 4 == 0:
            value_left //= 4
        if value_left != 1:
            raise ValueError(f"Comfy ConvRot group size must be a power of four, got {group_size}.")
        return cls(convrot=convrot, convrot_groupsize=group_size)


class DiffusionInt8ConvRotConfig(QuantizationConfig):
    """Offline ComfyUI W8A8 ConvRot configuration.

    Quantized layers are explicit because a Comfy checkpoint can mix INT8
    linears with BF16/FP16 linears.  The MiniMax-H3 pipeline fills this mapping
    from each layer's ``.comfy_quant`` tensor before constructing the model.
    """

    def __init__(
        self,
        layer_configs: Mapping[str, Mapping[str, Any] | Int8ConvRotLayerConfig] | None = None,
        quantized_layers: list[str] | None = None,
        convrot_groupsize: int = 256,
        ignored_layers: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.ignored_layers = ignored_layers or []
        self.layer_configs: dict[str, Int8ConvRotLayerConfig] = {}
        self.is_checkpoint_quantized = True
        self.is_checkpoint_int8_convrot_serialized = True

        if layer_configs:
            self.configure_layers(layer_configs)
        if quantized_layers:
            default = Int8ConvRotLayerConfig.from_mapping(
                {
                    "format": _FORMAT,
                    "convrot": True,
                    "convrot_groupsize": convrot_groupsize,
                }
            )
            for prefix in quantized_layers:
                self.layer_configs.setdefault(prefix, default)
        self._validate_checkpoint_layers_are_quantized(self.layer_configs)

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "int8_convrot"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        # comfy-kitchen supports INT8 tensor cores from Turing onward.
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DiffusionInt8ConvRotConfig:
        return cls(
            layer_configs=config.get("layer_configs"),
            quantized_layers=config.get("quantized_layers"),
            convrot_groupsize=config.get("convrot_groupsize", 256),
            ignored_layers=config.get("ignored_layers"),
        )

    def configure_layers(
        self,
        layer_configs: Mapping[str, Mapping[str, Any] | Int8ConvRotLayerConfig],
    ) -> None:
        """Install checkpoint-derived metadata before model construction."""
        parsed: dict[str, Int8ConvRotLayerConfig] = {}
        for prefix, value in layer_configs.items():
            if not prefix or prefix.endswith((".weight", ".weight_scale", ".comfy_quant")):
                raise ValueError(f"ConvRot layer metadata must use a module prefix, got {prefix!r}.")
            parsed[prefix] = (
                value if isinstance(value, Int8ConvRotLayerConfig) else Int8ConvRotLayerConfig.from_mapping(value)
            )
        self._validate_checkpoint_layers_are_quantized(parsed)
        if self.layer_configs and self.layer_configs != parsed:
            raise ValueError("ConvRot layer metadata was already configured with different checkpoint values.")
        self.layer_configs = parsed

    def _validate_checkpoint_layers_are_quantized(
        self,
        layer_configs: Mapping[str, Int8ConvRotLayerConfig],
    ) -> None:
        conflicts = sorted(prefix for prefix in layer_configs if is_layer_skipped(prefix, self.ignored_layers))
        if conflicts:
            raise ValueError(
                "Checkpoint-marked INT8 ConvRot layers cannot also be ignored: "
                f"{conflicts[:5]}. Remove them from ignored_layers."
            )

    def validate_model_bindings(self, model: Module) -> None:
        """Require every checkpoint marker to bind an executable ConvRot layer."""
        expected = set(self.layer_configs)
        bound = {
            method.prefix
            for module in model.modules()
            if isinstance(
                method := getattr(module, "quant_method", None),
                Int8ConvRotLinearMethod,
            )
            and method.quant_config is self
        }
        if bound != expected:
            missing = sorted(expected - bound)
            unexpected = sorted(bound - expected)
            raise ValueError(
                "MiniMax-H3 ConvRot checkpoint metadata does not match executable quantized layers: "
                f"unbound markers={missing[:5]}, unexpected bindings={unexpected[:5]}."
            )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None
        if is_layer_skipped(prefix, self.ignored_layers):
            return UnquantizedLinearMethod()
        layer_config = self.layer_configs.get(prefix)
        if layer_config is None:
            return UnquantizedLinearMethod()
        return Int8ConvRotLinearMethod(self, layer_config, prefix=prefix)


class Int8ConvRotLinearMethod(LinearMethodBase):
    """Execute a serialized Comfy INT8 weight without dequantizing it."""

    def __init__(
        self,
        quant_config: DiffusionInt8ConvRotConfig,
        layer_config: Int8ConvRotLayerConfig,
        *,
        prefix: str,
    ) -> None:
        self.quant_config = quant_config
        self.layer_config = layer_config
        self.prefix = prefix
        self._cuda_impl: Callable[..., torch.Tensor] | None = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        output_size_per_partition = sum(output_partition_sizes)
        group_size = self.layer_config.convrot_groupsize
        if self.layer_config.convrot and input_size_per_partition % group_size:
            raise ValueError(
                f"{self.prefix} has TP-local input width {input_size_per_partition}, "
                f"which is not aligned to its ConvRot group size {group_size}. "
                "Choose a tensor-parallel degree whose row-parallel shards preserve group boundaries."
            )

        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        weight = create_weight_parameter(
            output_size_per_partition=output_size_per_partition,
            input_size_per_partition=input_size_per_partition,
            weight_loader=weight_loader,
            params_dtype=torch.int8,
        )
        layer.register_parameter("weight", weight)
        scale = ChannelQuantScaleParameter(
            data=torch.empty((output_size_per_partition, 1), dtype=torch.float32),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", scale)

    def process_weights_after_loading(self, layer: Module) -> None:
        if layer.weight.dtype != torch.int8 or layer.weight.dim() != 2:
            raise ValueError(
                f"{self.prefix} expected a 2-D INT8 weight, got {tuple(layer.weight.shape)} {layer.weight.dtype}."
            )
        scale = layer.weight_scale
        if scale.dtype != torch.float32 or scale.numel() != layer.weight.shape[0]:
            raise ValueError(
                f"{self.prefix} expected one FP32 scale per output row; got "
                f"{tuple(scale.shape)} {scale.dtype} for weight {tuple(layer.weight.shape)}."
            )
        if not torch.isfinite(scale).all() or not torch.all(scale > 0):
            raise ValueError(f"{self.prefix} contains non-finite or non-positive INT8 weight scales.")
        layer.weight.data = layer.weight.data.contiguous()
        layer.weight_scale.data = scale.data.reshape(-1).contiguous()
        if layer.weight.is_cuda:
            probe = torch.empty(
                (1, layer.weight.shape[1]),
                dtype=layer.orig_dtype,
                device=layer.weight.device,
            )
            self._resolve_cuda_impl(layer, probe, bias=None)

    @staticmethod
    def _load_comfy_kitchen():
        try:
            import comfy_kitchen as ck
        except ImportError as exc:
            raise ImportError(
                "INT8 ConvRot requires comfy-kitchen with its CUDA backend. "
                'Install it with `pip install "vllm-omni[int8-convrot]"`.'
            ) from exc
        return ck

    @torch.compiler.disable
    def _resolve_cuda_impl(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        *,
        bias: torch.Tensor | None,
    ) -> Callable[..., torch.Tensor]:
        impl = self._cuda_impl
        if impl is not None:
            return impl
        ck = self._load_comfy_kitchen()
        kwargs = {
            "x": x,
            "weight": layer.weight,
            "weight_scale": layer.weight_scale,
            "bias": bias,
            "out_dtype": x.dtype,
            "convrot": self.layer_config.convrot,
            "convrot_groupsize": self.layer_config.convrot_groupsize,
            "input_act": None,
        }
        impl = ck.registry.get_implementation(
            "int8_linear",
            backend="cuda",
            kwargs=kwargs,
        )
        self._cuda_impl = impl
        return impl

    @staticmethod
    def _run_custom_op(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        output_dtype_code: int,
        convrot: bool,
        convrot_groupsize: int,
    ) -> torch.Tensor:
        # comfy-kitchen registers a fake implementation for this op, so Dynamo
        # keeps it opaque instead of tracing into the CUDA backend's DLPack calls.
        return torch.ops.comfy_kitchen.int8_linear(
            x,
            weight,
            weight_scale,
            bias,
            output_dtype_code,
            convrot,
            convrot_groupsize,
            None,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not x.is_cuda:
            raise RuntimeError(f"{self.prefix} INT8 ConvRot currently requires a CUDA activation tensor.")
        if x.dtype == torch.bfloat16:
            dtype_code = 2
        elif x.dtype == torch.float16:
            dtype_code = 1
        else:
            raise TypeError(f"{self.prefix} INT8 ConvRot supports only FP16/BF16 activations, got {x.dtype}.")
        # Normal model loading resolves this before regional torch.compile is
        # installed. Keep the lazy path for direct callers and unusual loaders.
        if self._cuda_impl is None:
            self._resolve_cuda_impl(layer, x, bias=bias)
        return self._run_custom_op(
            x.contiguous(),
            layer.weight,
            layer.weight_scale,
            bias,
            dtype_code,
            self.layer_config.convrot,
            self.layer_config.convrot_groupsize,
        )


__all__ = [
    "DiffusionInt8ConvRotConfig",
    "Int8ConvRotLayerConfig",
    "Int8ConvRotLinearMethod",
]
