# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``bsgemm-moe``: block-FP8 routed experts; every other layer stays bf16."""

from __future__ import annotations

from typing import Any

import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

QUANTIZATION_METHOD = "bsgemm-moe"


@register_quantization_config(QUANTIZATION_METHOD)
class ExpertFp8Config(Fp8Config):
    """vLLM's block-FP8 MoE path for routed experts, unquantized linears."""

    def __init__(self, weight_block_size: list[int]) -> None:
        super().__init__(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=weight_block_size,
        )

    @classmethod
    def get_name(cls) -> str:
        return QUANTIZATION_METHOD

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ExpertFp8Config:
        if config.get("activation_scheme", "dynamic") != "dynamic":
            raise ValueError(f"{QUANTIZATION_METHOD} needs dynamic activation scales")
        block = config.get("weight_block_size")
        if not (isinstance(block, list) and len(block) == 2 and all(int(b) > 0 for b in block)):
            raise ValueError(f"{QUANTIZATION_METHOD} needs a 2-int weight_block_size, got {block!r}")
        return cls(weight_block_size=[int(b) for b in block])

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        return super().get_quant_method(layer, prefix)
