# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Quantization mapping helpers for nested Qwen3-Omni stages."""

from vllm.model_executor.models.interfaces import SupportsQuant

_OUTER_MAPPING_APPLIED = "_qwen3_omni_outer_mapping_applied"


def apply_outer_quant_config_mapping(model: SupportsQuant) -> None:
    """Map checkpoint paths once at the unified model boundary."""
    SupportsQuant._maybe_apply_model_mapping(model)
    if model.quant_config is not None:
        setattr(model.quant_config, _OUTER_MAPPING_APPLIED, True)


def apply_nested_quant_config_mapping(model: SupportsQuant) -> None:
    """Avoid applying a stage weight mapper twice to a mapped config."""
    quant_config = model.quant_config
    if quant_config is None:
        return
    if not getattr(quant_config, _OUTER_MAPPING_APPLIED, False):
        SupportsQuant._maybe_apply_model_mapping(model)
        return

    # The outer model maps names, while the nested model still contributes
    # fused-module metadata needed by quantization scheme matching.
    if packed_modules_mapping := getattr(model, "packed_modules_mapping", None):
        quant_config.packed_modules_mapping.update(packed_modules_mapping)


class Qwen3OmniNestedSupportsQuant(SupportsQuant):
    """SupportsQuant variant for models initialized inside the stage wrapper."""

    def _maybe_apply_model_mapping(self) -> None:
        apply_nested_quant_config_mapping(self)
