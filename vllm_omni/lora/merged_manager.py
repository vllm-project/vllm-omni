# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Autoregressive adapters using the shared merged-LoRA manager."""

from collections.abc import Iterator

from torch import nn

from vllm_omni.lora.capability import MergedLoRASpec, resolve_merged_lora_spec
from vllm_omni.lora.manager import LoRAManager


class ARMergedLoRAManager(LoRAManager):
    """Bind checkpoint LoRA factors without installing runtime AR wrappers.

    The shared manager keeps binding wrappers outside the forward module tree;
    it needs neither ``enable_lora`` nor a Punica runtime context.
    """

    def _resolve_merged_spec(self) -> MergedLoRASpec:
        return resolve_merged_lora_spec(self.pipeline, stage="ar")

    def _iter_lora_components(self) -> Iterator[tuple[str, nn.Module]]:
        if self._merged_spec is not None:
            yield from super()._iter_lora_components()
            return
        yield "", self.pipeline

    def _compute_packed_modules_mapping(self) -> dict[str, list[str]]:
        if self._merged_spec is not None:
            return super()._compute_packed_modules_mapping()
        spec = self._resolve_merged_spec()
        return {name: list(parts) for name, parts in spec.packed_modules_mapping.items()}
