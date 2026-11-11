# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FP8 options backed by vLLM's quantization kernels."""

from __future__ import annotations

from typing import Any, Literal

import regex as re
from torch import nn
from vllm.config.quantization import QuantizationConfigArgs, QuantSpec
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerBlockOnlineLinearMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape, QuantKey, create_fp8_quant_key
from vllm.platforms import current_platform


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
    def from_config(cls, config: dict[str, Any]) -> SerializedPerChannelFp8Config:
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


class OnlineBlockFp8LinearMethod(Fp8PerBlockOnlineLinearMethod):
    """Online FP8 with matching configurable weight and activation groups."""

    def __init__(self, block_size: int) -> None:
        super().__init__()
        self.weight_block_size = [block_size, block_size]
        self.activation_quant_key = create_fp8_quant_key(
            static=False,
            group_shape=GroupShape(1, block_size),
        )
        self.weight_quant_key = create_fp8_quant_key(
            static=True,
            group_shape=GroupShape(block_size, block_size),
        )


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
        online_block_size: int | None = None,
    ) -> None:
        if is_checkpoint_fp8_serialized:
            raise ValueError("Serialized FP8 checkpoints require upstream Fp8Config")
        if activation_scheme != "dynamic":
            raise ValueError("Online FP8 quantization requires activation_scheme='dynamic'")
        if weight_block_size is not None:
            raise ValueError("Block-wise FP8 quantization requires a serialized checkpoint or fp8_per_block")
        if store_dtype is not None:
            raise ValueError("Online FP8 quantization does not support store_dtype")

        block_weight_key: QuantKey | None = None
        block_activation_key: QuantKey | None = None
        if online_block_size is not None:
            if type(online_block_size) is not int or online_block_size not in (32, 64, 128):
                raise ValueError("online_block_size must be 32, 64 or 128")
            if not current_platform.is_rocm():
                raise ValueError("online_block_size is currently supported on ROCm only")
            block_weight_key = create_fp8_quant_key(
                static=True,
                group_shape=GroupShape(online_block_size, online_block_size),
            )
            block_activation_key = create_fp8_quant_key(
                static=False,
                group_shape=GroupShape(1, online_block_size),
            )

        linear_spec = (
            QuantSpec(weight=block_weight_key, activation=block_activation_key)
            if block_weight_key is not None
            else QuantSpec(weight="fp8_per_tensor_static")
        )
        super().__init__(
            QuantizationConfigArgs(
                linear=linear_spec,
                moe=QuantSpec(weight="fp8_per_tensor_static"),
                ignore=list(ignored_layers or []),
            )
        )
        self.is_checkpoint_fp8_serialized = False
        self.activation_scheme = activation_scheme
        self.weight_block_size = None
        self.store_dtype = None
        self.online_block_size = online_block_size
        self._online_block_weight_key = block_weight_key
        self._online_block_activation_key = block_activation_key

    def _get_method_cls(
        self,
        spec: QuantSpec | None,
        table: dict[QuantKey, type],
        layer_cls: type[nn.Module],
    ) -> type | None:
        if (
            spec is not None
            and self._online_block_weight_key is not None
            and spec.weight == self._online_block_weight_key
        ):
            if spec.activation != self._online_block_activation_key:
                raise ValueError(
                    f"online_block_size requires matching dynamic activation groups; got activation={spec.activation}"
                )
            return OnlineBlockFp8LinearMethod
        return super()._get_method_cls(spec, table, layer_cls)

    def get_quant_method(self, layer: nn.Module, prefix: str) -> QuantizeMethodBase | None:
        if self.online_block_size is not None:
            resolved = self.resolve_quant_method_cls(type(layer), prefix)
            if resolved is not None:
                source, quant_key_str, target_pattern, _, quant_method_cls = resolved
                if quant_method_cls is OnlineBlockFp8LinearMethod:
                    self.quantized_layers[prefix] = (
                        source.value,
                        quant_key_str,
                        target_pattern,
                    )
                    return OnlineBlockFp8LinearMethod(self.online_block_size)
        return super().get_quant_method(layer, prefix)

    @classmethod
    def get_name(cls) -> str:
        return "fp8"
