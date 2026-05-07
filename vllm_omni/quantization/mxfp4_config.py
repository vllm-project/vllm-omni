# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 quantization config for diffusion transformers (W4A16).

Implements MXFP4 (Microscaling FP4) weight-only quantization for diffusion
models like Wan2.2. Reuses vLLM's Marlin kernel for efficient FP4 GEMM.

MXFP4 format:
- 4-bit float weights (E2M1) packed into uint8 (2 values per byte)
- Per-group E8M0 scales with group_size=32
- No global scale (unlike NVFP4)
- W4A16: 4-bit weights, 16-bit (BF16) activations

Online quantization flow:
1. create_weights() registers BF16 weight parameters
2. Weight loader loads BF16 checkpoints into these parameters
3. process_weights_after_loading() quantizes BF16 -> MXFP4 and repacks for Marlin
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter
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
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    prepare_fp4_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
)
from vllm.model_executor.utils import replace_parameter

from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)

# MXFP4 block size (OCP standard: 32 elements share one E8M0 scale)
MXFP4_BLOCK_SIZE = 32

# FP4 E2M1 representable values (absolute)
FP4_E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
FP4_E2M1_MAX = 6.0

# FP4 E2M1 encoding table: value -> 3-bit code
# 0.0=000, 0.5=001, 1.0=010, 1.5=011, 2.0=100, 3.0=101, 4.0=110, 6.0=111
FP4_ENCODE_LUT = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.uint8)


def _quantize_bf16_to_mxfp4(
    weight: torch.Tensor,
    block_size: int = MXFP4_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16/FP16 weight tensor to MXFP4 format.

    Args:
        weight: Input weight tensor of shape [out, in] with dtype bf16/fp16.
        block_size: Number of elements sharing one scale factor (default 32).

    Returns:
        qweight: Packed FP4 weights, shape [out, in//2], dtype uint8.
        weight_scale: Per-group E8M0 scales, shape [out, in//block_size], dtype uint8.
    """
    assert weight.ndim == 2, f"Expected 2D weight, got {weight.ndim}D"
    assert weight.dtype in (torch.bfloat16, torch.float16), f"Unsupported dtype: {weight.dtype}"

    out_features, in_features = weight.shape
    assert in_features % block_size == 0, (
        f"Input features ({in_features}) must be divisible by block_size ({block_size})"
    )

    device = weight.device
    # Reshape to [out, num_groups, block_size]
    weight_f32 = weight.to(torch.float32)
    weight_grouped = weight_f32.view(out_features, -1, block_size)

    # Compute per-group max absolute value
    abs_max = torch.max(torch.abs(weight_grouped), dim=-1, keepdim=True)[0]  # [out, num_groups, 1]

    # Compute E8M0 scale: scale = 2^round(log2(abs_max / FP4_MAX))
    # E8M0 is exponent-only: value = 2^(exp - 127), stored as uint8 exp
    # We clamp to avoid log(0)
    abs_max_clamped = torch.clamp(abs_max, min=1e-12)
    log2_scale = torch.log2(abs_max_clamped / FP4_E2M1_MAX)
    exp = torch.round(log2_scale + 127).clamp(0, 255).to(torch.uint8)  # E8M0 exponent

    # Decode E8M0 scale back to float for quantization
    scale_f32 = torch.pow(2.0, exp.to(torch.float32) - 127.0)  # [out, num_groups, 1]

    # Normalize and quantize to FP4
    normalized = weight_grouped / scale_f32  # [out, num_groups, block_size]
    normalized = torch.clamp(normalized, -FP4_E2M1_MAX, FP4_E2M1_MAX)

    # Round to nearest FP4 value using lookup
    abs_normalized = torch.abs(normalized)
    # Find nearest FP4 value index
    diffs = torch.abs(abs_normalized.unsqueeze(-1) - FP4_E2M1_VALUES.to(device))
    indices = torch.argmin(diffs, dim=-1).to(torch.uint8)  # [out, num_groups, block_size]

    # Apply sign
    signs = (normalized < 0).to(torch.uint8)
    fp4_codes = indices | (signs << 3)  # 4-bit code: [sign(1bit) | magnitude(3bit)]

    # Pack 2 FP4 values into 1 uint8 byte: [high_nibble | low_nibble]
    fp4_flat = fp4_codes.view(out_features, -1)  # [out, num_groups * block_size] = [out, in]
    low = fp4_flat[:, 0::2]  # [out, in//2]
    high = fp4_flat[:, 1::2]  # [out, in//2]
    qweight = (high << 4) | low  # Pack: high nibble | low nibble

    # Scales: remove the keepdim dimension -> [out, num_groups]
    weight_scale = exp.view(out_features, -1)  # [out, in//block_size]

    return qweight, weight_scale


class DiffusionMXFP4Config(QuantizationConfig):
    """MXFP4 quantization config for diffusion transformers.

    Supports W4A16 (4-bit weights, 16-bit activations) using Marlin kernel
    on CUDA GPUs (SM 75+).

    Args:
        ignored_layers: Layer name patterns to skip quantization.
        weight_block_size: Group size for per-group scaling (default 32).
    """

    def __init__(
        self,
        ignored_layers: list[str] | None = None,
        weight_block_size: list[int] | int = MXFP4_BLOCK_SIZE,
    ) -> None:
        super().__init__()
        self.ignored_layers = ignored_layers or []
        # vLLM expects weight_block_size to be a list (e.g., [32, 32])
        if isinstance(weight_block_size, int):
            self.weight_block_size = [weight_block_size, weight_block_size]
        else:
            self.weight_block_size = weight_block_size
        # Store scalar for internal quantization logic
        self._block_size_scalar = self.weight_block_size[0]

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Marlin requires SM 75+ (Turing or newer)
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.ignored_layers is not None:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DiffusionMXFP4Config":
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(config, ["modules_to_not_convert"], None)
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], MXFP4_BLOCK_SIZE)
        return cls(
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
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
                logger.info(f"[MXFP4] Skipping layer {prefix} (ignored)")
                return UnquantizedLinearMethod()

            if current_omni_platform.is_cuda():
                logger.info(f"[MXFP4] Applying quantization to layer: {prefix}")
                return DiffusionMXFP4LinearMethod(self)
            else:
                raise NotImplementedError(
                    f"MXFP4 is not supported on {current_omni_platform._omni_enum.value}. "
                    "Currently only CUDA (SM 75+) is supported."
                )
        return None


class DiffusionMXFP4LinearMethod(LinearMethodBase):
    """MXFP4 Linear method for diffusion models using Marlin kernel.

    This method implements online quantization:
    1. create_weights() registers BF16 weight parameters
    2. Weight loader loads BF16 checkpoints into these parameters
    3. process_weights_after_loading() quantizes BF16 -> MXFP4 and repacks for Marlin

    Applicable to:
    - ColumnParallelLinear
    - RowParallelLinear
    - QKVParallelLinear
    - Standard nn.Linear (via vLLM wrapper)
    """

    def __init__(self, quant_config: DiffusionMXFP4Config):
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
        """Create BF16 weight parameters for online MXFP4 quantization.

        The weight loader will load BF16 weights from the checkpoint into
        these parameters. Quantization happens in process_weights_after_loading().
        """
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.params_dtype = params_dtype

        # Register BF16 weight parameter (will be quantized after loading)
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

    def process_weights_after_loading(self, layer: Module) -> None:
        """Quantize BF16 weights to MXFP4 and repack for Marlin kernel."""
        layer_name = getattr(layer, "prefix", "unknown_layer")
        logger.info(f"[MXFP4] Starting online quantization for: {layer_name}")
        
        device = layer.weight.device
        torch.cuda.synchronize(device)
        mem_before = torch.cuda.memory_allocated(device)

        # Step 1: Quantize BF16 -> MXFP4
        qweight, weight_scale = _quantize_bf16_to_mxfp4(
            layer.weight.data,
            block_size=self.quant_config._block_size_scalar,
        )
        logger.info(f"[MXFP4] Quantized {layer_name}: weight shape={qweight.shape}, dtype={qweight.dtype}")

        # Step 2: Replace parameters with quantized versions
        replace_parameter(layer, "weight", Parameter(qweight, requires_grad=False))
        replace_parameter(layer, "weight_scale", Parameter(weight_scale, requires_grad=False))

        # Step 3: Repack for Marlin kernel
        prepare_fp4_layer_for_marlin(layer)

        # Force release old BF16 weights from PyTorch cache for accurate measurement
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        mem_after = torch.cuda.memory_allocated(device)
        
        saved_mb = (mem_before - mem_after) / (1024 * 1024)
        logger.info(f"[MXFP4] Quantization complete for {layer_name}. Memory saved: {saved_mb:.2f} MB")

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Execute MXFP4 W4A16 GEMM using Marlin kernel."""
        return apply_fp4_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=None,  # MXFP4 has no global scale
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )
