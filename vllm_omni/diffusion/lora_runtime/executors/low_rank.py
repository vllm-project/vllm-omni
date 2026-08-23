# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from ..bindings import ResolvedLoRABinding
from ..layers import DynamicLoRALinear
from ..types import DiffusionLoRAComposition, LoadedDiffusionLoRA


def _replace_submodule(root: nn.Module, path: str, module: nn.Module) -> None:
    parent_path, separator, child_name = path.rpartition(".")
    parent = root.get_submodule(parent_path) if separator else root
    if isinstance(parent, (nn.ModuleList, nn.Sequential)) and child_name.isdigit():
        parent[int(child_name)] = module
    else:
        setattr(parent, child_name, module)


class LowRankLinearExecutor:
    """Default ``Wx + B(Ax)`` executor for diffusion linear layers."""

    def __init__(self, pipeline: nn.Module, device: torch.device, dtype: torch.dtype) -> None:
        self.pipeline = pipeline
        self.device = device
        self.dtype = dtype
        self._layers: dict[str, DynamicLoRALinear] = {}
        self._finalized = False

    def install(
        self,
        loras: Mapping[str, LoadedDiffusionLoRA],
        bindings: Sequence[object],
    ) -> None:
        if self._finalized:
            raise RuntimeError("Diffusion LoRA executor is already finalized")
        for untyped_binding in bindings:
            if not isinstance(untyped_binding, ResolvedLoRABinding):
                raise TypeError(f"Expected ResolvedLoRABinding, got {type(untyped_binding)!r}")
            binding = untyped_binding
            if binding.full_module_name in self._layers:
                raise ValueError(f"Duplicate diffusion LoRA binding {binding.full_module_name!r}")
            wrapper = DynamicLoRALinear(
                binding,
                loras,
                device=self.device,
                dtype=self.dtype,
            )
            component = getattr(self.pipeline, binding.component_name)
            _replace_submodule(component, binding.module_name, wrapper)
            self._layers[binding.full_module_name] = wrapper

    def activate(self, composition: DiffusionLoRAComposition) -> None:
        if not self._finalized:
            raise RuntimeError("Diffusion LoRA executor must be finalized before activation")
        for layer in self._layers.values():
            layer.activate(composition)

    def finalize(self) -> None:
        if not self._layers:
            raise ValueError("Diffusion LoRA executor installed no layers")
        self._finalized = True


def create_low_rank_executor(
    pipeline: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> LowRankLinearExecutor:
    return LowRankLinearExecutor(pipeline, device, dtype)
