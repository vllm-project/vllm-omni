# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias

import torch

LoRAStateDict = dict[str, torch.Tensor]


@dataclass(frozen=True)
class DiffusionAdapterUpdate:
    """Canonical non-low-rank update produced by a model converter."""

    module_name: str


@dataclass(frozen=True)
class AdditiveBiasUpdate(DiffusionAdapterUpdate):
    """Add a tensor to a linear module's output bias."""

    tensor: torch.Tensor


@dataclass(frozen=True)
class ConvertedLoRAState:
    """Canonical output of a model-owned raw checkpoint converter."""

    lora_tensors: LoRAStateDict
    auxiliary_updates: tuple[DiffusionAdapterUpdate, ...] = ()


LoRAStateDictConverter: TypeAlias = Callable[[LoRAStateDict], LoRAStateDict | ConvertedLoRAState]


@dataclass(frozen=True)
class DiffusionLoRALoadPlan:
    """Model-owned description of how to interpret a raw LoRA checkpoint."""

    peft_config: Mapping[str, Any]
    weights_mapper: Any = None
    state_dict_converter: LoRAStateDictConverter | None = None


@dataclass(frozen=True)
class DiffusionLoRAApplyPlan:
    """Model-owned description of where normalized LoRA tensors are applied."""

    component_names: tuple[str, ...] | None = None
    target_modules: tuple[str, ...] | None = None
    packed_modules_mapping: Mapping[str, Sequence[str]] = field(default_factory=dict)


class SupportsDiffusionLoRAPlan(Protocol):
    """Optional model/pipeline extension point consumed by the shared backend.

    The model describes checkpoint normalization and logical-to-physical layer
    binding. The backend remains the sole owner of registration, composition,
    dynamic execution, and prefusion mathematics.
    """

    def get_lora_load_plan(
        self,
        adapter_path: str,
        tensor_keys: tuple[str, ...],
    ) -> DiffusionLoRALoadPlan | None: ...

    def get_lora_apply_plan(self) -> DiffusionLoRAApplyPlan: ...
