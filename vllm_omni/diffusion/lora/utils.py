# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from typing import Any

import torch.nn as nn
from transformers import PretrainedConfig
from vllm.config.lora import LoRAConfig
from vllm.logger import init_logger

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


_MODEL_LOADER_HINT = (
    "Provide a model-specific loader via the pipeline's "
    "`_load_diffusion_lora_adapter` hook, or supply a canonical "
    "single-rank checkpoint."
)


def read_peft_rank_alpha_patterns(lora_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Best-effort read of ``rank_pattern``/``alpha_pattern`` from a PEFT config.

    ``vllm.lora.peft_helper.PEFTHelper`` has no ``rank_pattern``/``alpha_pattern``
    fields and silently drops them (see ``PEFTHelper.from_dict``), so a checkpoint
    that declares per-module ranks/alphas loses that information before the
    generic diffusion loader ever sees it. We re-read the raw ``adapter_config.json``
    only to detect (not honor) those declarations, so the loader can fail loudly
    instead of applying one global scale.

    Returns empty dicts when the config is absent or unreadable (e.g. tensorizer
    or remote paths); the tensor-shape based check below still runs in that case.
    """
    config_path = os.path.join(lora_path, "adapter_config.json")
    if not os.path.isfile(config_path):
        return {}, {}
    try:
        with open(config_path) as f:
            config = json.load(f)
    except (OSError, ValueError) as exc:  # unreadable / not JSON
        logger.debug("Could not read %s for rank_pattern detection: %s", config_path, exc)
        return {}, {}
    rank_pattern = config.get("rank_pattern") or {}
    alpha_pattern = config.get("alpha_pattern") or {}
    rank_pattern = rank_pattern if isinstance(rank_pattern, Mapping) else {}
    alpha_pattern = alpha_pattern if isinstance(alpha_pattern, Mapping) else {}
    return dict(rank_pattern), dict(alpha_pattern)


def _iter_module_ranks(loras: Mapping[str, Any]) -> Iterator[tuple[str, int, int]]:
    """Yield ``(module_name, a_rank, b_rank)`` for every loaded LoRA slice.

    ``a_rank`` is the row count of ``lora_A`` and ``b_rank`` the column count of
    ``lora_B`` (LoRA stores ``A`` as ``[rank, in]`` and ``B`` as ``[out, rank]``).
    ``PackedLoRALayerWeights`` exposes ``lora_a``/``lora_b`` as lists of slices,
    so each slice is reported separately.
    """
    for name, weights in loras.items():
        lora_a = getattr(weights, "lora_a", None)
        lora_b = getattr(weights, "lora_b", None)
        a_slices = list(lora_a) if isinstance(lora_a, (list, tuple)) else [lora_a]
        b_slices = list(lora_b) if isinstance(lora_b, (list, tuple)) else [lora_b]
        for a, b in zip(a_slices, b_slices):
            if a is None or b is None:
                continue
            yield name, int(a.shape[0]), int(b.shape[1])


def validate_generic_peft_ranks(
    loras: Mapping[str, Any],
    global_rank: int,
    *,
    rank_pattern: Mapping[str, Any] | None = None,
    alpha_pattern: Mapping[str, Any] | None = None,
) -> None:
    """Guard the generic PEFT fallback against incorrect per-module rank scaling.

    The generic path in ``DiffusionLoRAManager._load_adapter`` builds every module
    from the single global ``r``/``lora_alpha`` in ``adapter_config.json`` (via
    ``vllm.lora.peft_helper.PEFTHelper`` and ``LoRALayerWeights.from_config``) and
    applies one uniform ``scaling = lora_alpha / r``. It never re-derives the
    per-module rank from the actual ``A``/``B`` tensors. As a result a checkpoint
    whose physical modules do not all share ``r`` — e.g. a fused-QKV adapter whose
    packed projection has rank ``3 * r`` while the remaining modules keep ``r`` —
    is loaded at the wrong rank and silently scaled incorrectly.

    This function keeps the generic path honest: it accepts only uniform-rank
    adapters with no per-module ``rank_pattern``/``alpha_pattern``. Anything else
    raises an actionable ``ValueError`` pointing at a model-specific loader.

    It deliberately does **not** attempt to recover a per-module scale (e.g.
    ``lora_alpha / actual_rank``): some distilled checkpoints bake the effective
    scale into the ``A``/``B`` tensors themselves, so silently reinterpreting the
    scale here could double-apply or drop it. Rejecting is the safe generic
    behavior; a model that understands its own contract can supply a loader.

    Args:
        loras: Mapping of module name to loaded LoRA weights (``lora_model.loras``).
        global_rank: The single ``r`` from the PEFT config (``peft_helper.r``).
        rank_pattern: Raw ``rank_pattern`` declared in ``adapter_config.json``, if any.
        alpha_pattern: Raw ``alpha_pattern`` declared in ``adapter_config.json``, if any.

    Raises:
        ValueError: If the adapter declares a per-module pattern the generic loader
            cannot honor, or if any module's tensor rank disagrees with ``global_rank``,
            or if a module's ``lora_A``/``lora_B`` ranks are mutually inconsistent.
    """
    if rank_pattern:
        raise ValueError(
            f"adapter_config.json declares a non-empty 'rank_pattern' "
            f"({sorted(rank_pattern)}), but the generic diffusion PEFT loader applies "
            f"a single global rank/scale and does not honor per-module rank_pattern "
            f"(vllm.lora.peft_helper.PEFTHelper drops it). " + _MODEL_LOADER_HINT
        )
    if alpha_pattern:
        raise ValueError(
            f"adapter_config.json declares a non-empty 'alpha_pattern' "
            f"({sorted(alpha_pattern)}), but the generic diffusion PEFT loader applies "
            f"a single global lora_alpha and does not honor per-module alpha_pattern. " + _MODEL_LOADER_HINT
        )

    inconsistent: list[tuple[str, int, int]] = []
    mismatched: list[tuple[str, int]] = []
    for name, a_rank, b_rank in _iter_module_ranks(loras):
        if a_rank != b_rank:
            inconsistent.append((name, a_rank, b_rank))
        elif a_rank != global_rank:
            mismatched.append((name, a_rank))

    if inconsistent:
        detail = ", ".join(f"{n} (lora_A rank={a}, lora_B rank={b})" for n, a, b in inconsistent[:8])
        raise ValueError(
            f"LoRA tensors have inconsistent internal rank between lora_A and lora_B "
            f"for {len(inconsistent)} module(s): {detail}. The checkpoint may be corrupt "
            f"or use a packed/fused layout the generic diffusion loader cannot interpret. " + _MODEL_LOADER_HINT
        )

    if mismatched:
        observed = sorted({rank for _, rank in mismatched})
        detail = ", ".join(f"{n} (rank={r})" for n, r in mismatched[:8])
        more = "" if len(mismatched) <= 8 else f" (+{len(mismatched) - 8} more)"
        raise ValueError(
            f"LoRA tensor rank does not match the global PEFT config r={global_rank} for "
            f"{len(mismatched)} module(s): {detail}{more}. Observed ranks: {observed}. "
            f"The checkpoint likely contains packed/fused or mixed-rank modules; the generic "
            f"diffusion loader applies a single global rank/scale and would silently scale "
            f"these modules. " + _MODEL_LOADER_HINT
        )
