# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-component quantization routing for multi-stage models.

Routes get_quant_method() to different configs based on longest-prefix match:
    {"transformer": fp8_config, "vae": None}
    "transformer.blocks.0.attn.to_q" -> fp8_config
    "vae.encoder.conv_in"            -> None
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizeMethodBase,
    )


# Pre-quantized checkpoints (modelopt FP8/FP4/MXFP8) only quantize the
# Thinker LM.  Vision and audio encoder weights remain in BF16 with no
# corresponding scale tensors in the checkpoint.
PRE_QUANTIZED_METHODS: frozenset[str] = frozenset({"modelopt", "modelopt_fp4", "modelopt_mxfp8"})


def resolve_encoder_quant_config(
    quant_config: QuantizationConfig | None,
) -> QuantizationConfig | None:
    """Resolve quantization config for vision / audio encoders.

    Returns *None* for pre-quantized methods so that FP8 kernels are never
    applied to BF16 encoder weights (which lack scale tensors).  All other
    configs — including ``ComponentQuantizationConfig`` and ``None`` — are
    returned as-is so the caller can handle them.
    """
    if (
        quant_config is not None
        and not isinstance(quant_config, ComponentQuantizationConfig)
        and quant_config.get_name() in PRE_QUANTIZED_METHODS
    ):
        return None
    return quant_config


class ComponentQuantizationConfig(QuantizationConfig):
    """Routes quantization to different configs by layer prefix."""

    def __init__(
        self,
        component_configs: dict[str, QuantizationConfig | None],
        default_config: QuantizationConfig | None = None,
    ) -> None:
        self._components = component_configs
        self._default = default_config
        self._sorted_prefixes = sorted(self._components.keys(), key=len, reverse=True)

    def resolve(self, prefix: str) -> QuantizationConfig | None:
        """Find the config for a given layer prefix (longest-prefix match).

        Note: vLLM may remap quantization prefixes vs model definition
        prefixes (e.g. via WeightsMapper). If prefixes don't match after
        remapping, layers may fall through to the default config.
        """
        for comp_prefix in self._sorted_prefixes:
            if prefix.startswith(comp_prefix):
                return self._components[comp_prefix]
        return self._default

    def get_name(self) -> str:
        return "component"

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        config = self.resolve(prefix)
        if config is None:
            return None
        return config.get_quant_method(layer, prefix)

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    def get_min_capability(self) -> int:
        """Return the minimum capability across all component configs."""
        caps = [c.get_min_capability() for c in self._components.values() if c is not None]
        if self._default is not None:
            caps.append(self._default.get_min_capability())
        return min(caps) if caps else 0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ComponentQuantizationConfig:
        raise NotImplementedError("Use build_quant_config() instead")

    def get_config_filenames(self) -> list[str]:
        return []

    @property
    def component_configs(self) -> dict[str, QuantizationConfig | None]:
        return self._components

    @property
    def default_config(self) -> QuantizationConfig | None:
        return self._default
