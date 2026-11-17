# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FP8 options backed by vLLM's quantization kernels."""

from typing import Any, Literal

import regex as re
from vllm.config.quantization import QuantizationConfigArgs, QuantSpec
from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig


class SerializedPerChannelFp8Config(ModelOptFp8Config):
    """E4M3 weights with one FP32 scale per output channel and dynamic tokens.

    FastVideo exports this same tensor layout under ``quant_method=fp8``.
    Reuse upstream's channel/token kernel and TP-aware parameter loaders.
    """

    def __init__(self, quantized_modules: str) -> None:
        super().__init__(
            quant_method="FP8_PER_CHANNEL_PER_TOKEN",
            is_checkpoint_fp8_serialized=True,
            kv_cache_quant_method=None,
            exclude_modules=[],
        )
        self.quantized_modules = re.compile(quantized_modules)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SerializedPerChannelFp8Config":
        for key, expected in (
            ("fmt", "e4m3"),
            ("weight_scheme", "per_channel"),
            ("activation_scheme", "dynamic_per_token"),
        ):
            if config.get(key) != expected:
                raise ValueError(f"Unsupported serialized per-channel FP8 {key}: {config.get(key)!r}")
        pattern = config.get("quantized_modules")
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("Serialized per-channel FP8 requires quantized_modules")
        return cls(pattern)

    def is_layer_excluded(self, prefix: str) -> bool:
        return self.quantized_modules.fullmatch(prefix + ".weight") is None

    def get_name(self) -> Literal["fp8"]:
        return "fp8"


class DiffusionFp8Config(OnlineQuantizationConfig):
    """Keep the diffusion ``fp8`` option for quantizing BF16/FP16 weights.

    Upstream Fp8Config now describes serialized checkpoints only. Keeping the
    Omni method name also lets checkpoint metadata replace this online config
    through the existing serialized-checkpoint reconciliation path.
    """

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: list[str] | None = None,
        weight_block_size: list[int] | None = None,
        store_dtype: str | None = None,
    ) -> None:
        if is_checkpoint_fp8_serialized:
            raise ValueError("Serialized FP8 checkpoints require upstream Fp8Config")
        if activation_scheme != "dynamic":
            raise ValueError("Online FP8 quantization requires activation_scheme='dynamic'")
        if weight_block_size is not None:
            raise ValueError("Block-wise FP8 quantization requires a serialized checkpoint or fp8_per_block")
        if store_dtype is not None:
            raise ValueError("Online FP8 quantization does not support store_dtype")
        super().__init__(
            QuantizationConfigArgs(
                linear=QuantSpec(weight="fp8_per_tensor_static"),
                moe=QuantSpec(weight="fp8_per_tensor_static"),
                ignore=list(ignored_layers or []),
            )
        )
        self.is_checkpoint_fp8_serialized = False
        self.activation_scheme = activation_scheme
        self.weight_block_size = None
        self.store_dtype = None

    @classmethod
    def get_name(cls) -> str:
        return "fp8"
