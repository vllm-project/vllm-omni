# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FP8 config covering both serialized checkpoints and online quantization."""

from torch import nn
from vllm.config.quantization import resolve_quantization_config
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig


@register_quantization_config("fp8")
class OmniFp8Config(Fp8Config):
    """Fp8Config that also supports online FP8 when is_checkpoint_fp8_serialized is False."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: list[str] | None = None,
        weight_block_size: list[int] | None = None,
        store_dtype: str | None = None,
    ) -> None:
        # Upstream rejects non-serialized FP8, so initialize as serialized and then set the real mode.
        super().__init__(
            is_checkpoint_fp8_serialized=True,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
            store_dtype=store_dtype,
        )
        if not is_checkpoint_fp8_serialized:
            if activation_scheme != "dynamic":
                raise ValueError("Online FP8 quantization requires activation_scheme='dynamic'")
            if weight_block_size is not None:
                raise ValueError("Block-wise FP8 quantization requires a serialized checkpoint or fp8_per_block")
            if store_dtype is not None:
                raise ValueError("Online FP8 quantization does not support store_dtype")
            self.is_checkpoint_fp8_serialized = False

    def get_quant_method(self, layer: nn.Module, prefix: str) -> QuantizeMethodBase | None:
        if self.is_checkpoint_fp8_serialized:
            return super().get_quant_method(layer, prefix)
        # https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/model_executor/model_loader/weight_utils.py#L347
        args = resolve_quantization_config("fp8_per_tensor", {"ignore": self.ignored_layers})
        # fp8_per_tensor has a built-in preset and a config dict is passed, so we should never get None here
        if args is None:
            raise RuntimeError("vLLM did not resolve the fp8_per_tensor online quantization preset")
        online = OnlineQuantizationConfig(args)
        # Ensure packed module mapping is forwarded through properly
        online.packed_modules_mapping = self.packed_modules_mapping
        return online.get_quant_method(layer, prefix)
