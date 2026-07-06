# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from vllm.logger import init_logger

from vllm_omni.config.lora import LoRAConfig
from vllm_omni.diffusion.lora.layers import (
    DiffusionColumnParallelLinearWithLoRA,
    DiffusionMergedColumnParallelLinearWithLoRA,
    DiffusionMergedQKVParallelLinearWithLoRA,
    DiffusionQKVParallelLinearWithLoRA,
    DiffusionReplicatedLinearWithLoRA,
    DiffusionRowParallelLinearWithLoRA,
)

logger = init_logger(__name__)


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


# Key suffixes used by single-file LoRA checkpoints. Kohya-style exports
# (used by e.g. lightx2v/Qwen-Image-Lightning and most community diffusion
# LoRAs) use `lora_down`/`lora_up` plus per-module `alpha` scalars; some
# exporters use PEFT-style `lora_A`/`lora_B` names but still ship a bare
# safetensors file without an `adapter_config.json`.
_LORA_A_SUFFIXES = (".lora_down.weight", ".lora_A.weight")
_LORA_B_SUFFIXES = (".lora_up.weight", ".lora_B.weight")
_ALPHA_SUFFIX = ".alpha"
# Root prefixes some exporters prepend to module paths.
_ROOT_PREFIXES = ("base_model.model.", "diffusion_model.", "transformer.")


def find_single_file_lora(lora_path: str) -> str | None:
    """Return the safetensors file if ``lora_path`` is a single-file LoRA.

    Accepts either a direct path to a ``.safetensors`` file or a directory
    that contains exactly one ``.safetensors`` file and no
    ``adapter_config.json`` (i.e. not a PEFT adapter directory).
    Returns None when the path should be handled by the PEFT loader.
    """
    if os.path.isfile(lora_path) and lora_path.lower().endswith(".safetensors"):
        return lora_path
    if os.path.isdir(lora_path):
        if os.path.isfile(os.path.join(lora_path, "adapter_config.json")):
            return None
        candidates = [f for f in os.listdir(lora_path) if f.lower().endswith(".safetensors")]
        if len(candidates) == 1:
            return os.path.join(lora_path, candidates[0])
    return None


def _resolve_module_path(path: str, expected_lora_modules: set[str]) -> str | None:
    """Map a checkpoint module path onto the pipeline's module tree.

    diffusers wraps some projections in a ModuleList (e.g. ``attn.to_out.0``)
    while vllm-omni implementations expose the projection directly
    (``attn.to_out``), so a trailing index is folded into its parent when the
    parent is a supported LoRA module. Returns None if the path cannot be
    matched to any supported module.
    """
    leaf = path.rsplit(".", 1)[-1]
    if leaf in expected_lora_modules:
        return path
    if leaf.isdigit():
        parent = path.rsplit(".", 1)[0]
        if parent.rsplit(".", 1)[-1] in expected_lora_modules:
            return parent
    return None


def _target_module_suffix(path: str) -> str:
    """Strip the leading block path (up to the first index) for readability,
    e.g. ``transformer_blocks.31.attn.to_q`` -> ``attn.to_q``."""
    parts = path.split(".")
    for i, part in enumerate(parts):
        if part.isdigit():
            return ".".join(parts[i + 1 :]) or path
    return path


def convert_single_file_lora(
    tensors: dict[str, torch.Tensor],
    expected_lora_modules: set[str],
) -> tuple[dict, dict[str, torch.Tensor]]:
    """Convert a single-file (Kohya/diffusers) LoRA state dict to PEFT layout.

    - renames ``{module}.lora_down/lora_up.weight`` (or ``lora_A/lora_B``) to
      ``base_model.model.{module}.lora_A/lora_B.weight``
    - folds per-module ``{module}.alpha`` scalars into ``lora_B`` and returns
      a config with ``lora_alpha == r``, so vLLM's global scaling stays 1.0
      and the effective per-module scale remains ``alpha / rank``
    - strips root prefixes some exporters add (``transformer.`` etc.)
    - folds diffusers ModuleList indirections (``attn.to_out.0`` ->
      ``attn.to_out``) via :func:`_resolve_module_path`

    Returns ``(peft_config_dict, converted_tensors)`` consumable by
    ``PEFTHelper.from_dict`` and ``LoRAModel.from_lora_tensors``.
    """
    modules: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in tensors.items():
        name = key
        for prefix in _ROOT_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
        matched = False
        for suffix in _LORA_A_SUFFIXES:
            if name.endswith(suffix):
                modules.setdefault(name[: -len(suffix)], {})["lora_A"] = tensor
                matched = True
                break
        if matched:
            continue
        for suffix in _LORA_B_SUFFIXES:
            if name.endswith(suffix):
                modules.setdefault(name[: -len(suffix)], {})["lora_B"] = tensor
                matched = True
                break
        if matched:
            continue
        if name.endswith(_ALPHA_SUFFIX):
            modules.setdefault(name[: -len(_ALPHA_SUFFIX)], {})["alpha"] = tensor
        else:
            raise ValueError(f"Unrecognized key in single-file LoRA checkpoint: {key}")

    converted: dict[str, torch.Tensor] = {}
    target_modules: set[str] = set()
    unmatched: list[str] = []
    ranks: set[int] = set()
    max_rank = 0
    for path, parts in sorted(modules.items()):
        if "lora_A" not in parts or "lora_B" not in parts:
            raise ValueError(f"LoRA module {path} is missing its lora_down/lora_up weights")
        resolved = _resolve_module_path(path, expected_lora_modules)
        if resolved is None:
            unmatched.append(path)
            continue
        rank = parts["lora_A"].shape[0]
        ranks.add(rank)
        max_rank = max(max_rank, rank)
        lora_b = parts["lora_B"]
        alpha = parts.get("alpha")
        if alpha is not None:
            lora_b = lora_b * (float(alpha) / rank)
        converted[f"base_model.model.{resolved}.lora_A.weight"] = parts["lora_A"].contiguous()
        converted[f"base_model.model.{resolved}.lora_B.weight"] = lora_b.contiguous()
        target_modules.add(_target_module_suffix(resolved))

    if unmatched:
        raise ValueError(
            f"While converting a single-file LoRA, {len(unmatched)} module(s) do not match "
            f"any supported LoRA module of this pipeline (supported: "
            f"{sorted(expected_lora_modules)}): {unmatched[:8]}"
        )
    if not converted:
        raise ValueError("Single-file LoRA checkpoint contains no LoRA modules.")
    if len(ranks) > 1:
        logger.warning(
            "Single-file LoRA has mixed ranks %s; using r=%d (max) in the synthesized "
            "PEFT config. Per-module scaling stays exact because alpha is folded per module.",
            sorted(ranks),
            max_rank,
        )

    peft_config = {
        "r": max_rank,
        "lora_alpha": max_rank,
        "target_modules": sorted(target_modules),
    }
    return peft_config, converted


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
