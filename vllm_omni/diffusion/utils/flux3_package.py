# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Flux3 package paths and package configuration, without model imports."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _resolve_flux3_component_path(shared_root: str, path: str) -> tuple[str, str]:
    """Resolve a package-relative path or hf://owner/repo/component reference."""
    if not path.startswith("hf://"):
        return shared_root, path
    parts = path.removeprefix("hf://").split("/")
    if len(parts) < 3 or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Invalid Flux3 component reference {path!r}; expected hf://owner/repo/component")
    return "/".join(parts[:2]), "/".join(parts[2:])


@dataclass(frozen=True)
class _Flux3PackageSelection:
    shared_root: str
    transformer_root: str
    package_config: dict[str, Any]
    quantization: str | None


def _requested_flux3_variant(additional_config: Mapping[str, Any]) -> str | None:
    value = additional_config.get("variant")
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"additional_config['variant'] must be a string or None, got {type(value).__name__}")
    variant = value.strip()
    if not variant:
        raise ValueError("additional_config['variant'] must name an available Flux3 variant")
    return variant


def _is_flux3_package_config(config: Mapping[str, Any]) -> bool:
    """Recognize Flux3 by its name or the supported package architecture."""
    dit = config.get("dit")
    if not isinstance(dit, dict) or not isinstance(config.get("text_encoder"), dict):
        return False
    if config.get("model") == "flux-3-image-klein":
        return True
    return (
        dit.get("hidden") == 3072
        and dit.get("heads") == 24
        and dit.get("depth") == 5
        and dit.get("depth_single") == 28
        and dit.get("in_channels") == {"image": 128, "image_cond": 128, "text": 20480}
    )


def _read_flux3_package_config(model_path: str) -> dict[str, Any] | None:
    """Read the flux3 package config when a local package is supplied."""
    if not model_path or not os.path.isdir(model_path):
        return None
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path) as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict) or not _is_flux3_package_config(raw):
        return None
    raw["model"] = "flux3"
    return raw


def _available_flux3_variants(model_path: str) -> list[str]:
    variants_dir = os.path.join(model_path, "variants")
    if not os.path.isdir(variants_dir):
        return []
    return sorted(
        name for name in os.listdir(variants_dir) if os.path.isfile(os.path.join(variants_dir, name, "config.json"))
    )


def _resolve_flux3_package_selection(
    model_path: str,
    additional_config: Mapping[str, Any],
) -> _Flux3PackageSelection:
    variant = _requested_flux3_variant(additional_config)
    direct_config = _read_flux3_package_config(model_path)
    if direct_config is None:
        raise ValueError("Flux3 model path must point at a flux3 package root")
    if direct_config.get("quantization") is not None:
        raise ValueError(
            "Flux3 model path must point at the shared package root; "
            "select quantized packages with additional_config['variant'] instead"
        )

    if variant is not None:
        available = _available_flux3_variants(model_path)
        if variant not in available:
            suffix = f"; available variants: {', '.join(available)}" if available else ""
            raise ValueError(f"Flux3 package variant {variant!r} not found under {model_path!r}{suffix}")
        shared_root = model_path
        transformer_root = os.path.join(model_path, "variants", variant)
        package_config = _read_flux3_package_config(transformer_root)
        if package_config is None:
            raise ValueError(f"Flux3 package variant {variant!r} must contain a flux3 config.json")
        quantization = package_config.get("quantization")
    else:
        shared_root = model_path
        transformer_root = model_path
        package_config = direct_config
        quantization = package_config.get("quantization")
    return _Flux3PackageSelection(
        shared_root=shared_root,
        transformer_root=transformer_root,
        package_config=package_config,
        quantization=quantization,
    )
