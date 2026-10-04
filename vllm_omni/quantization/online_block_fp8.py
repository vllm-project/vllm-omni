# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Explicit online block FP8 for ROCm diffusion linears."""

from typing import Any

from torch import nn
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerBlockOnlineLinearMethod,
    Fp8PerTensorOnlineLinearMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape, create_fp8_quant_key
from vllm.platforms import current_platform


class OnlineBlockFp8LinearMethod(Fp8PerBlockOnlineLinearMethod):
    """Use matching weight and activation groups before kernel selection."""

    def __init__(self, block_size: int) -> None:
        super().__init__()
        self.weight_block_size = [block_size, block_size]
        self.activation_quant_key = create_fp8_quant_key(static=False, group_shape=GroupShape(1, block_size))
        self.weight_quant_key = create_fp8_quant_key(static=True, group_shape=GroupShape(block_size, block_size))


class OnlineBlockFp8Config(Fp8Config):
    """Opt-in block scaling for non-serialized, dynamic FP8 linear weights.

    Delegate layer exclusion and fused-prefix matching to Fp8Config. Existing
    per-tensor and serialized FP8 configurations keep their original behavior.
    """

    def __init__(self, *, online_block_size: int, **kwargs: Any) -> None:
        if type(online_block_size) is not int or online_block_size not in (32, 64, 128):
            raise ValueError("online_block_size must be 32, 64 or 128")
        if kwargs.get("is_checkpoint_fp8_serialized", False) or kwargs.get("weight_block_size") is not None:
            raise ValueError("online_block_size requires non-serialized BF16/FP16 checkpoint weights")
        if kwargs.get("activation_scheme", "dynamic") != "dynamic":
            raise ValueError("online_block_size requires dynamic activation scaling")
        if not current_platform.is_rocm():
            raise ValueError("online_block_size is currently supported on ROCm")
        super().__init__(**kwargs)
        self.online_block_size = online_block_size

    def get_quant_method(self, layer: nn.Module, prefix: str):
        method = super().get_quant_method(layer, prefix)
        if isinstance(method, Fp8PerTensorOnlineLinearMethod):
            return OnlineBlockFp8LinearMethod(self.online_block_size)
        return method
