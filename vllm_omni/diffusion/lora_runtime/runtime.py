# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from vllm.logger import init_logger

from .bindings import resolve_lora_bindings
from .loader import resolve_diffusion_lora_artifact
from .support import DiffusionLoRAExecutor, DiffusionLoRASupport
from .types import (
    DiffusionLoRAComposition,
    DiffusionLoRADeployment,
    DiffusionLoRASelection,
    LoadedDiffusionLoRA,
    normalize_diffusion_lora_composition,
)

logger = init_logger(__name__)


class DiffusionLoRARuntime:
    """Immutable startup registry plus request-scoped LoRA activation."""

    def __init__(
        self,
        pipeline: nn.Module,
        deployments: Sequence[DiffusionLoRADeployment],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        support = getattr(pipeline, "diffusion_lora_support", None)
        if not isinstance(support, DiffusionLoRASupport):
            raise ValueError(f"{type(pipeline).__name__} does not declare Diffusion LoRA Runtime support")
        if not deployments:
            raise ValueError("Diffusion LoRA Runtime requires at least one --diffusion-lora deployment")

        by_name: dict[str, LoadedDiffusionLoRA] = {}
        loader = support.loader_factory(pipeline)
        for deployment in sorted(deployments, key=lambda item: item.name):
            if deployment.name in by_name:
                raise ValueError(f"Duplicate diffusion LoRA deployment name: {deployment.name!r}")
            artifact_path = resolve_diffusion_lora_artifact(deployment)
            logger.info("Loading diffusion LoRA %s from %s", deployment.name, artifact_path)
            loaded = loader.load(deployment, artifact_path)
            if loaded.name != deployment.name:
                raise ValueError(f"Model loader renamed diffusion LoRA {deployment.name!r} to {loaded.name!r}")
            if not loaded.updates:
                raise ValueError(f"Diffusion LoRA {deployment.name!r} contains no low-rank updates")
            loaded.update_map()
            by_name[deployment.name] = loaded

        bindings = resolve_lora_bindings(pipeline, support.binding_plan, by_name)
        executor = support.executor_factory(pipeline, device, dtype)
        executor.install(by_name, bindings)
        executor.finalize()

        self._support = support
        self._registered_names = tuple(by_name)
        self._executor: DiffusionLoRAExecutor = executor
        self._active: DiffusionLoRAComposition = ()
        self.activate(())

    @property
    def registered_names(self) -> tuple[str, ...]:
        return self._registered_names

    @property
    def active_composition(self) -> DiffusionLoRAComposition:
        return self._active

    def activate(
        self,
        composition: Sequence[DiffusionLoRASelection | dict] | None,
    ) -> None:
        normalized = normalize_diffusion_lora_composition(composition)
        unknown = [selection.name for selection in normalized if selection.name not in self._registered_names]
        if unknown:
            raise ValueError(f"Unknown diffusion LoRA selection(s) {unknown}; deployed={list(self._registered_names)}")
        if len(normalized) > 1 and not self._support.supports_composition:
            raise ValueError(f"{type(self._executor).__name__} does not support multi-LoRA composition")
        if normalized == self._active:
            return
        self._executor.activate(normalized)
        self._active = normalized
