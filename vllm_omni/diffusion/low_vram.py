# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in diffusion memory preset expansion, after configuration sources merge."""

import json
from collections.abc import Mapping
from typing import Any

LOW_VRAM_PRESETS = ("fp8",)
_PIPELINES = {"ZImagePipeline"}


def expand_low_vram_preset(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Fill unset memory knobs and reject combinations outside the preset contract.

    This is deliberately independent of torch, model downloads and GPU probing.
    Call only after source merging: inserting defaults into an ingress fragment
    would incorrectly override explicit values from other sources.
    """
    result = dict(kwargs)
    preset = result.get("low_vram_preset")
    if preset is None:
        return result
    if preset not in LOW_VRAM_PRESETS:
        raise ValueError(f"Unknown low_vram_preset {preset!r}; choose from {LOW_VRAM_PRESETS}")

    defaults = {
        "quantization_config": "fp8",
        "diffusion_offload_config": {"mode": "layer", "components": ["dit", "text_encoder"]},
        "cache_backend": "none",
        "vae_use_tiling": True,
    }
    for key, value in defaults.items():
        if result.get(key) is None:
            result[key] = value

    quant = result["quantization_config"]
    if isinstance(quant, Mapping):
        method = quant.get("method", quant.get("quant_method"))
    elif isinstance(quant, str):
        method = quant
    else:
        method = quant.get_name() if hasattr(quant, "get_name") else None
    if method != "fp8":
        raise ValueError("low_vram_preset='fp8' requires FP8 quantization; remove the preset to use another method")

    from vllm_omni.diffusion.offloader.config import OffloadMode, parse_diffusion_offload_config

    offload = parse_diffusion_offload_config(result["diffusion_offload_config"])
    if offload is None or offload.components != {"dit", "text_encoder"}:
        raise ValueError("low_vram_preset requires diffusion_offload_config.components=['dit', 'text_encoder']")
    if offload.mode is OffloadMode.MODULE:
        raise ValueError(
            "low_vram_preset requires layer offload; FP8 with module offload produced invalid images "
            "during qualification. Use mode='layer' or remove the preset. "
            "Removing the preset does not resolve the underlying module-offload issue."
        )
    if offload.mode is OffloadMode.LAYER and any(
        options.weight_transfer is not None and options.weight_transfer.value != "rank-local"
        for options in offload.layer_options.values()
    ):
        raise ValueError("low_vram_preset supports only rank-local layer offload on a single GPU")
    # Do not let the legacy priority rule silently discard a conflicting mode.
    if (
        result.get("enable_distributed_layerwise_offload")
        or (result.get("enable_cpu_offload") and offload.mode is not OffloadMode.MODULE)
        or (result.get("enable_layerwise_offload") and offload.mode is not OffloadMode.LAYER)
    ):
        raise ValueError("low_vram_preset conflicts with legacy enable_*_offload flags; use diffusion_offload_config")
    if result["vae_use_tiling"] is not True:
        raise ValueError("low_vram_preset requires vae_use_tiling=True")
    if result["cache_backend"] not in ("none", "tea_cache"):
        raise ValueError("low_vram_preset supports cache_backend='none' or 'tea_cache'")
    cache_config = result.get("cache_config")
    if isinstance(cache_config, str):
        try:
            cache_config = json.loads(cache_config)
        except json.JSONDecodeError as exc:
            raise ValueError("low_vram_preset requires valid cache_config JSON") from exc
        if not isinstance(cache_config, dict):
            raise ValueError("low_vram_preset requires cache_config to be a JSON object")
        result["cache_config"] = cache_config
    if result.get("diffusion_compile_granularity") == "full":
        raise ValueError("low_vram_preset uses CPU offload and requires regional compilation")
    if result.get("diffusion_load_format") not in (None, "default"):
        raise ValueError("low_vram_preset requires the native diffusion loader (diffusion_load_format='default')")
    if result.get("enable_sleep_mode"):
        raise ValueError("low_vram_preset with sleep mode is not qualified; disable sleep mode or remove the preset")

    validate_low_vram_pipeline(result.get("model_class_name"))
    return result


def validate_low_vram_pipeline(model_class_name: str | None) -> None:
    """Check architecture once metadata resolution has supplied its class name."""
    if model_class_name is None:
        return
    if model_class_name not in _PIPELINES:
        raise ValueError(
            f"low_vram_preset currently supports only ZImagePipeline, got {model_class_name}; "
            "remove the preset to configure offload and quantization for this pipeline"
        )
