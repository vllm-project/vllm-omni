# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""TorchAO quantization config registrations."""

from __future__ import annotations

from typing import Any

from typing_extensions import Self
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.torchao import TorchAOConfig


@register_quantization_config("torchao")
class OmniTorchAOConfig(TorchAOConfig):
    """TorchAO config registered for Omni's shared quantization lookup."""

    def __init__(self, **kwargs: Any) -> None:
        if "quant_type" in kwargs:
            config = TorchAOConfig.from_config({**kwargs, "quant_method": "torchao"})

            super().__init__(
                config.torchao_config,
                config.skip_modules,
                config.is_checkpoint_torchao_serialized,
            )
        else:
            super().__init__(**kwargs)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        return cls(**config)


@register_quantization_config("torchao_float8_weight_only")
class OmniTorchAOFloat8WeightOnlyConfig(OmniTorchAOConfig):
    """TorchAO serialized Float8 weight-only checkpoint shorthand."""

    def __init__(self, **kwargs: Any) -> None:
        from torchao.quantization import Float8WeightOnlyConfig

        super().__init__(
            torchao_config=Float8WeightOnlyConfig(set_inductor_config=False),
            is_checkpoint_torchao_serialized=True,
        )
