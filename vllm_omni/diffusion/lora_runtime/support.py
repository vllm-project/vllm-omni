# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypeAlias

import torch
import torch.nn as nn

from .types import (
    DiffusionLoRAComposition,
    DiffusionLoRADeployment,
    LoadedDiffusionLoRA,
)


@dataclass(frozen=True)
class DiffusionLoRABindingPlan:
    """Model-owned allowlist for logical-to-physical LoRA binding."""

    component_names: tuple[str, ...]
    target_modules: tuple[str, ...]
    packed_modules_mapping: Mapping[str, Sequence[str]] = field(default_factory=dict)


class DiffusionLoRALoader(Protocol):
    def load(
        self,
        deployment: DiffusionLoRADeployment,
        artifact_path: Path,
    ) -> LoadedDiffusionLoRA: ...


class DiffusionLoRAExecutor(Protocol):
    def install(
        self,
        loras: Mapping[str, LoadedDiffusionLoRA],
        bindings: Sequence[object],
    ) -> None: ...

    def activate(self, composition: DiffusionLoRAComposition) -> None: ...

    def finalize(self) -> None: ...


LoRALoaderFactory: TypeAlias = Callable[[nn.Module], DiffusionLoRALoader]
LoRAExecutorFactory: TypeAlias = Callable[[nn.Module, torch.device, torch.dtype], DiffusionLoRAExecutor]


@dataclass(frozen=True)
class DiffusionLoRASupport:
    """Complete model-owned contract consumed by the common runtime."""

    loader_factory: LoRALoaderFactory
    binding_plan: DiffusionLoRABindingPlan
    executor_factory: LoRAExecutorFactory
    supports_composition: bool = False
