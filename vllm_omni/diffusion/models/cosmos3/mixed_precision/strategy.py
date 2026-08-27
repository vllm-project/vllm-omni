# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Format-specific live-weight validation and reference A16 materialization."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptFp8LinearMethod,
    ModelOptNvFp4LinearMethod,
)
from vllm.model_executor.layers.quantization.utils import nvfp4_emulation_utils

_NVFP4_BLOCK_SIZE = 16


class Cosmos3PrecisionStrategy:
    """Quantization-format boundary behind the shared precision schedule."""

    def accepts(self, method: object | None) -> bool:
        """Return whether this strategy owns a native linear method."""
        raise NotImplementedError

    def validate_before_processing(
        self,
        method: object,
        layer: torch.nn.Module,
        module_name: str,
    ) -> None:
        """Reject native transforms that cannot support live dequantization."""

    def validate_after_processing(
        self,
        layer: torch.nn.Module,
        module_name: str,
    ) -> None:
        """Validate the live native weight representation."""
        raise NotImplementedError

    def materialize(self, layer: torch.nn.Module) -> torch.Tensor:
        """Materialize the live quantized weight as a dense BF16 matrix."""
        raise NotImplementedError

    def apply_high(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply the reference dense A16 path."""
        input_size = int(layer.input_size_per_partition)
        output_size = int(layer.output_size_per_partition)
        weight = self.materialize(layer)
        expected_shape = (output_size, input_size)
        if tuple(weight.shape) != expected_shape:
            raise RuntimeError(f"Dense weight shape {tuple(weight.shape)} does not match {expected_shape}")
        x_2d = x.reshape(-1, x.shape[-1])
        if x_2d.shape[1] != input_size:
            raise ValueError(f"Expected activation width {input_size}, got {x_2d.shape[1]}")
        weight = weight.to(dtype=x.dtype)
        if bias is not None and bias.dtype != x.dtype:
            bias = bias.to(dtype=x.dtype)
        output = F.linear(x_2d, weight, bias)
        return output.view(*x.shape[:-1], output_size)


class Fp8W8A8W8A16Strategy(Cosmos3PrecisionStrategy):
    """Use native ModelOpt W8A8 or reference dense W8A16."""

    def accepts(self, method: object | None) -> bool:
        return isinstance(method, ModelOptFp8LinearMethod)

    def validate_before_processing(
        self,
        method: object,
        layer: torch.nn.Module,
        module_name: str,
    ) -> None:
        if hasattr(layer, "pre_quant_scale"):
            raise ValueError(f"{module_name} uses unsupported SmoothQuant pre_quant_scale")
        weight_scale = getattr(layer, "weight_scale", None)
        if not isinstance(weight_scale, torch.Tensor) or weight_scale.numel() != 1:
            raise ValueError(f"{module_name} requires serialized tensorwise FP8 weights")
        fp8_kernel = getattr(method, "fp8_linear", None)
        from vllm.model_executor.kernels.linear.scaled_mm.marlin import (
            MarlinFP8ScaledMMLinearKernel,
        )

        if isinstance(fp8_kernel, MarlinFP8ScaledMMLinearKernel):
            raise ValueError(
                f"{module_name} selected Marlin FP8, which repacks the only weight copy; "
                "use a backend that retains canonical FP8 weights"
            )

    def validate_after_processing(
        self,
        layer: torch.nn.Module,
        module_name: str,
    ) -> None:
        weight = getattr(layer, "weight", None)
        scale = getattr(layer, "weight_scale", None)
        if not isinstance(weight, torch.Tensor) or weight.dtype != torch.float8_e4m3fn:
            dtype = getattr(weight, "dtype", None)
            raise TypeError(f"{module_name} requires a canonical FP8 weight, got {dtype}")
        if weight.ndim != 2:
            raise ValueError(f"{module_name} requires a rank-2 FP8 weight, got {tuple(weight.shape)}")
        if not isinstance(scale, torch.Tensor):
            raise ValueError(f"{module_name} has an FP8 weight but no weight_scale")
        if scale.numel() != 1:
            raise ValueError(f"{module_name} requires one live FP8 weight scale")
        input_size = int(layer.input_size_per_partition)
        output_size = int(layer.output_size_per_partition)
        if weight.shape[0] < input_size or weight.shape[1] < output_size:
            raise ValueError(
                f"{module_name} live FP8 weight shape {tuple(weight.shape)} does not cover "
                f"(K, N)=({input_size}, {output_size})"
            )
        _validate_positive_finite_scale(scale, module_name)

    def materialize(self, layer: torch.nn.Module) -> torch.Tensor:
        input_size = int(layer.input_size_per_partition)
        output_size = int(layer.output_size_per_partition)
        weight = layer.weight[:input_size, :output_size]
        scale = layer.weight_scale.reshape(1).to(device=weight.device, dtype=torch.float32)
        return (weight.to(torch.float32) * scale).t().to(torch.bfloat16)


class Nvfp4W4A4W4A16Strategy(Cosmos3PrecisionStrategy):
    """Use native ModelOpt W4A4 or reference dense W4A16."""

    def accepts(self, method: object | None) -> bool:
        return isinstance(method, ModelOptNvFp4LinearMethod)

    def validate_before_processing(
        self,
        method: object,
        layer: torch.nn.Module,
        module_name: str,
    ) -> None:
        kernel = getattr(method, "kernel", None)
        from vllm.model_executor.kernels.linear.nvfp4.cutlass import (
            CutlassNvFp4LinearKernel,
        )
        from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
            FlashInferCuteDslNvFp4LinearKernel,
            FlashInferCutlassNvFp4LinearKernel,
        )

        compatible_kernels = (
            CutlassNvFp4LinearKernel,
            FlashInferCuteDslNvFp4LinearKernel,
            FlashInferCutlassNvFp4LinearKernel,
        )
        if not isinstance(kernel, compatible_kernels):
            kernel_name = type(kernel).__name__ if kernel is not None else "none"
            raise ValueError(
                f"{module_name} selected unsupported NVFP4 backend {kernel_name}; "
                "live W4A16 dequantization requires a CUTLASS-compatible layout"
            )
        global_scale = _nvfp4_global_scale(layer)
        if global_scale is None or global_scale.numel() != 1:
            raise ValueError(f"{module_name} is missing ModelOpt NVFP4 scales")

    def validate_after_processing(
        self,
        layer: torch.nn.Module,
        module_name: str,
    ) -> None:
        packed = getattr(layer, "weight", None)
        scale = getattr(layer, "weight_scale", None)
        global_scale = getattr(layer, "weight_global_scale", None)
        if not isinstance(packed, torch.Tensor) or packed.dtype != torch.uint8:
            dtype = getattr(packed, "dtype", None)
            raise TypeError(f"{module_name} requires a live packed NVFP4 weight, got {dtype}")
        if not isinstance(scale, torch.Tensor) or not isinstance(global_scale, torch.Tensor):
            raise ValueError(f"{module_name} is missing live NVFP4 scales")
        output_size = int(layer.output_size_per_partition)
        input_size = int(layer.input_size_per_partition)
        if input_size % _NVFP4_BLOCK_SIZE != 0:
            raise ValueError(f"{module_name} input size {input_size} is not divisible by {_NVFP4_BLOCK_SIZE}")
        expected_weight = (output_size, input_size // 2)
        if packed.shape[0] < expected_weight[0] or packed.shape[1] < expected_weight[1]:
            raise ValueError(
                f"{module_name} live NVFP4 weight shape {tuple(packed.shape)} does not cover {expected_weight}"
            )
        _validate_positive_finite_scale(global_scale, module_name)
        lut_handle = nvfp4_emulation_utils.kE2M1ToFloat_handle
        if lut_handle.val.device != packed.device:
            lut_handle.val = lut_handle.val.to(packed.device)

    def materialize(self, layer: torch.nn.Module) -> torch.Tensor:
        output_size = int(layer.output_size_per_partition)
        input_size = int(layer.input_size_per_partition)
        packed = layer.weight[:output_size, : input_size // 2]
        return nvfp4_emulation_utils.dequantize_to_dtype(
            packed,
            layer.weight_scale,
            layer.weight_global_scale,
            torch.bfloat16,
            _NVFP4_BLOCK_SIZE,
            True,
        )


def _validate_positive_finite_scale(scale: torch.Tensor, module_name: str) -> None:
    values = scale.detach().float()
    if not torch.isfinite(values).all() or not (values > 0).all():
        raise ValueError(f"{module_name} has a non-finite or non-positive weight scale")


def _nvfp4_global_scale(layer: torch.nn.Module) -> torch.Tensor | None:
    for name in ("weight_scale_2", "weight_global_scale"):
        value = getattr(layer, name, None)
        if isinstance(value, torch.Tensor):
            return value
    return None
