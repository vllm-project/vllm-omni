# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import math

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.config.lora import LoRAConfig

from vllm_omni.diffusion.lora.layers import (
    DiffusionColumnParallelLinearWithLoRA,
    DiffusionMergedColumnParallelLinearWithLoRA,
    DiffusionMergedQKVParallelLinearWithLoRA,
    DiffusionQKVParallelLinearWithLoRA,
    DiffusionReplicatedLinearWithLoRA,
    DiffusionRowParallelLinearWithLoRA,
    DiffusionTorchLinearWithLoRA,
)


def _get_submodule(root: nn.Module, module_path: str) -> nn.Module | None:
    try:
        return root.get_submodule(module_path)
    except AttributeError:
        return None


def fold_diffusers_lora_alpha(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fold per-module Diffusers alpha values into their B matrices."""

    converted = dict(state_dict)
    for alpha_key in (key for key in state_dict if key.endswith(".alpha")):
        base_key = alpha_key.removesuffix(".alpha")
        lora_a_key = f"{base_key}.lora_A.weight"
        lora_b_key = f"{base_key}.lora_B.weight"
        if lora_a_key not in state_dict or lora_b_key not in state_dict:
            raise ValueError(f"LoRA alpha key {alpha_key!r} does not have matching A/B weights")
        lora_a = state_dict[lora_a_key]
        lora_b = state_dict[lora_b_key]
        if lora_a.ndim != 2 or lora_b.ndim != 2:
            raise ValueError(f"LoRA alpha key {alpha_key!r} requires two-dimensional A/B weights")
        rank = lora_a.shape[0]
        if rank <= 0:
            raise ValueError(f"LoRA alpha key {alpha_key!r} requires a positive rank")
        alpha = state_dict[alpha_key]
        if alpha.numel() != 1:
            raise ValueError(f"LoRA alpha key {alpha_key!r} must contain one scalar value")
        alpha_value = float(alpha.item())
        if not math.isfinite(alpha_value):
            raise ValueError(f"LoRA alpha key {alpha_key!r} must be finite")
        converted[lora_b_key] = lora_b * (alpha_value / rank)
        converted.pop(alpha_key)
    return converted


def _match_target_modules(module_name: str, target_modules: list[str]) -> bool:
    """from vllm/lora/model_manager.py _match_target_modules, helper function"""
    import regex as re

    return any(
        re.match(rf".*\.{target_module}$", module_name) or target_module == module_name
        for target_module in target_modules
    )


def _expand_expected_modules_for_packed_layers(
    supported_modules: set[str],
    packed_modules_mapping: dict[str, list[str]] | None,
) -> set[str]:
    """Expand expected LoRA module suffixes for packed (fused) projections.

    Some diffusion models use packed projections like `to_qkv` or `w13`, while
    LoRA checkpoints are typically saved against the logical sub-projections
    (e.g. `to_q`/`to_k`/`to_v`, `w1`/`w3`). The packed layer name is present in
    `supported_modules`, but the sublayer names are not. Expanding the set
    ensures these sublayer keys are not dropped when loading a LoRA checkpoint.

    The packed→sublayer mapping is model-specific and is derived from each
    diffusion model's `stacked_params_mapping` (used by `load_weights()`), so
    new packed layers are added alongside the model implementation rather than
    hard-coded in the LoRA framework.
    """
    expanded = set(supported_modules)
    if not packed_modules_mapping:
        return expanded

    for packed_name, sub_names in packed_modules_mapping.items():
        if packed_name in supported_modules:
            expanded.update(sub_names)

    return expanded


def from_layer_diffusion(
    layer: nn.Module,
    max_loras: int,
    lora_config: LoRAConfig,
    packed_modules_list: list[str],
    model_config: PretrainedConfig | None = None,
) -> nn.Module:
    """
    Diffusion-specific layer replacement. similar to vLLM's `from_layer`
    """
    diffusion_lora_classes = [
        DiffusionMergedQKVParallelLinearWithLoRA,
        DiffusionQKVParallelLinearWithLoRA,
        DiffusionMergedColumnParallelLinearWithLoRA,
        DiffusionColumnParallelLinearWithLoRA,
        DiffusionRowParallelLinearWithLoRA,
        DiffusionReplicatedLinearWithLoRA,
        DiffusionTorchLinearWithLoRA,
    ]

    for lora_cls in diffusion_lora_classes:
        if lora_cls.can_replace_layer(
            source_layer=layer,
            lora_config=lora_config,
            packed_modules_list=packed_modules_list,
            model_config=model_config,
        ):
            instance = lora_cls(layer)  # type: ignore[arg-type]
            instance.create_lora_weights(max_loras, lora_config, model_config)
            return instance

    return layer
