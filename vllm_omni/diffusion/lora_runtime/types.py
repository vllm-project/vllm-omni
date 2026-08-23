# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch


@dataclass(frozen=True)
class DiffusionLoRADeployment:
    """One immutable startup-time adapter definition."""

    name: str
    path: str


@dataclass(frozen=True)
class DiffusionLoRASelection:
    """One request-time selection of a deployment-owned adapter."""

    name: str
    scale: float = 1.0


DiffusionLoRAComposition: TypeAlias = tuple[DiffusionLoRASelection, ...]
DiffusionLoRACompositionKey: TypeAlias = tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class LowRankUpdate:
    """Model-normalized low-rank update for one logical module."""

    component: str
    logical_target: str
    lora_a: torch.Tensor
    lora_b: torch.Tensor
    intrinsic_scale: float = 1.0

    @property
    def rank(self) -> int:
        return int(self.lora_a.shape[0])


@dataclass(frozen=True)
class LoadedDiffusionLoRA:
    """Canonical output of a model-owned loader."""

    name: str
    updates: tuple[LowRankUpdate, ...]

    def update_map(self) -> dict[tuple[str, str], LowRankUpdate]:
        result: dict[tuple[str, str], LowRankUpdate] = {}
        for update in self.updates:
            key = (update.component, update.logical_target)
            if key in result:
                raise ValueError(f"LoRA {self.name!r} contains duplicate update {key!r}")
            result[key] = update
        return result


_DEPLOYMENT_FIELDS = frozenset({"name", "path"})
_SELECTION_FIELDS = frozenset({"name", "scale"})


def _parse_json_object(value: str, *, label: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be a JSON object: {exc.msg}") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def parse_diffusion_lora_deployments(
    values: Sequence[str | Mapping[str, Any]] | None,
) -> tuple[DiffusionLoRADeployment, ...]:
    """Parse repeatable startup definitions without assigning hidden IDs."""

    if not values:
        return ()
    deployments: list[DiffusionLoRADeployment] = []
    seen: set[str] = set()
    for raw_value in values:
        value = _parse_json_object(raw_value, label="--diffusion-lora") if isinstance(raw_value, str) else raw_value
        unknown = set(value) - _DEPLOYMENT_FIELDS
        if unknown:
            raise ValueError(f"--diffusion-lora contains unknown fields: {sorted(unknown)}")
        name = value.get("name")
        path = value.get("path")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("--diffusion-lora requires a non-empty string name")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("--diffusion-lora requires a non-empty string path")
        name = name.strip()
        if name in seen:
            raise ValueError(f"Duplicate diffusion LoRA deployment name: {name!r}")
        seen.add(name)
        deployments.append(DiffusionLoRADeployment(name=name, path=path.strip()))
    return tuple(sorted(deployments, key=lambda item: item.name))


def normalize_diffusion_lora_composition(
    values: Sequence[DiffusionLoRASelection | Mapping[str, Any]] | None,
) -> DiffusionLoRAComposition:
    """Return a deterministic name/scale composition for scheduling."""

    if not values:
        return ()
    combined: dict[str, float] = {}
    for raw_value in values:
        if isinstance(raw_value, DiffusionLoRASelection):
            selection = raw_value
        elif isinstance(raw_value, Mapping):
            unknown = set(raw_value) - _SELECTION_FIELDS
            if unknown:
                raise ValueError(f"Diffusion LoRA selection contains unknown fields: {sorted(unknown)}")
            name = raw_value.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Diffusion LoRA selection requires a non-empty string name")
            selection = DiffusionLoRASelection(name=name.strip(), scale=float(raw_value.get("scale", 1.0)))
        else:
            raise TypeError(f"Expected a diffusion LoRA selection object, got {type(raw_value)!r}")

        name = selection.name.strip()
        scale = float(selection.scale)
        if not name:
            raise ValueError("Diffusion LoRA selection name must not be empty")
        if not math.isfinite(scale):
            raise ValueError(f"Diffusion LoRA scale must be finite, got {scale!r}")
        combined[name] = combined.get(name, 0.0) + scale

    return tuple(
        DiffusionLoRASelection(name=name, scale=scale) for name, scale in sorted(combined.items()) if scale != 0.0
    )


def diffusion_lora_composition_key(
    composition: Sequence[DiffusionLoRASelection | Mapping[str, Any]] | None,
) -> DiffusionLoRACompositionKey:
    return tuple((item.name, item.scale) for item in normalize_diffusion_lora_composition(composition))
