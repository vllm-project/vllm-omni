# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Autoregressive adapters using the shared merged-LoRA manager."""

from collections.abc import Iterator

from torch import nn

from vllm_omni.lora.manager import LoRAManager


class ARMergedLoRAManager(LoRAManager):
    """Bind checkpoint LoRA factors without installing runtime AR wrappers.

    The shared manager keeps binding wrappers outside the forward module tree;
    it needs neither ``enable_lora`` nor a Punica runtime context.
    """

    def _iter_lora_components(self) -> Iterator[tuple[str, nn.Module]]:
        yield "", self.pipeline

    def _compute_packed_modules_mapping(self) -> dict[str, list[str]]:
        return {
            name.rsplit(".", 1)[-1]: [part.rsplit(".", 1)[-1] for part in parts]
            for name, parts in self.pipeline.packed_modules_mapping.items()
        }
