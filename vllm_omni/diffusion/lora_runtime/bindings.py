# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import torch.nn as nn

from .support import DiffusionLoRABindingPlan
from .types import LoadedDiffusionLoRA, LowRankUpdate


@dataclass(frozen=True)
class ResolvedLoRABinding:
    component_name: str
    module_name: str
    module: nn.Module
    logical_targets: tuple[str, ...]
    updates: Mapping[str, tuple[LowRankUpdate | None, ...]]

    @property
    def full_module_name(self) -> str:
        return f"{self.component_name}.{self.module_name}"


def _resolve_update_target(
    component: nn.Module,
    update: LowRankUpdate,
    plan: DiffusionLoRABindingPlan,
) -> tuple[str, tuple[str, ...]]:
    leaf = update.logical_target.rsplit(".", 1)[-1]
    if leaf not in plan.target_modules:
        raise ValueError(f"LoRA target {update.component}.{update.logical_target} is not allowed by the model plan")

    try:
        component.get_submodule(update.logical_target)
    except AttributeError:
        pass
    else:
        return update.logical_target, (update.logical_target,)

    prefix, separator, logical_leaf = update.logical_target.rpartition(".")
    for packed_leaf, logical_leaves in plan.packed_modules_mapping.items():
        logical_tuple = tuple(logical_leaves)
        if logical_leaf not in logical_tuple:
            continue
        physical_target = f"{prefix}.{packed_leaf}" if separator else packed_leaf
        try:
            component.get_submodule(physical_target)
        except AttributeError:
            continue
        targets = tuple(f"{prefix}.{name}" if separator else name for name in logical_tuple)
        return physical_target, targets

    raise ValueError(f"LoRA target {update.component}.{update.logical_target} does not resolve to a model module")


def resolve_lora_bindings(
    pipeline: nn.Module,
    plan: DiffusionLoRABindingPlan,
    loras: Mapping[str, LoadedDiffusionLoRA],
) -> tuple[ResolvedLoRABinding, ...]:
    """Resolve every normalized update exactly once against model modules."""

    allowed_components = set(plan.component_names)
    components: dict[str, nn.Module] = {}
    for component_name in plan.component_names:
        component = getattr(pipeline, component_name, None)
        if not isinstance(component, nn.Module):
            raise ValueError(f"Diffusion LoRA component {component_name!r} is not present on the pipeline")
        components[component_name] = component

    grouped: dict[tuple[str, str], dict[str, dict[str, LowRankUpdate]]] = defaultdict(lambda: defaultdict(dict))
    binding_targets: dict[tuple[str, str], tuple[str, ...]] = {}
    for lora_name, loaded in loras.items():
        seen: set[tuple[str, str]] = set()
        for update in loaded.updates:
            if update.component not in allowed_components:
                raise ValueError(
                    f"LoRA {lora_name!r} targets undeclared component {update.component!r}; "
                    f"allowed={sorted(allowed_components)}"
                )
            identity = (update.component, update.logical_target)
            if identity in seen:
                raise ValueError(f"LoRA {lora_name!r} contains duplicate target {identity!r}")
            seen.add(identity)
            physical_name, logical_targets = _resolve_update_target(
                components[update.component],
                update,
                plan,
            )
            binding_key = (update.component, physical_name)
            previous_targets = binding_targets.setdefault(binding_key, logical_targets)
            if previous_targets != logical_targets:
                raise ValueError(
                    f"LoRA target {binding_key!r} mixes direct and packed update layouts: "
                    f"{previous_targets!r} != {logical_targets!r}"
                )
            grouped[binding_key][lora_name][update.logical_target] = update

    bindings: list[ResolvedLoRABinding] = []
    for (component_name, module_name), adapter_updates in sorted(grouped.items()):
        logical_targets = binding_targets[(component_name, module_name)]
        updates: dict[str, tuple[LowRankUpdate | None, ...]] = {}
        for lora_name in sorted(loras):
            by_target = adapter_updates.get(lora_name, {})
            updates[lora_name] = tuple(by_target.get(target) for target in logical_targets)
        bindings.append(
            ResolvedLoRABinding(
                component_name=component_name,
                module_name=module_name,
                module=components[component_name].get_submodule(module_name),
                logical_targets=logical_targets,
                updates=updates,
            )
        )
    if not bindings:
        raise ValueError("The deployed diffusion LoRAs did not resolve to any model modules")
    return tuple(bindings)
