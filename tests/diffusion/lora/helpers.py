# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared test helpers for diffusion LoRA tests."""

from __future__ import annotations

import torch
from vllm.model_executor.layers.linear import LinearBase


class FakeLinearBase(LinearBase):
    """Minimal LinearBase stub for LoRA layer discovery."""

    def __init__(self):
        torch.nn.Module.__init__(self)


class DummyBaseLayerWithLoRA(torch.nn.Module):
    """Fake LoRA wrapper that records set/reset/create calls per slot."""

    def __init__(self, base_layer: torch.nn.Module):
        super().__init__()
        self.base_layer = base_layer

        self.set_calls: list[
            tuple[int, list[torch.Tensor | None] | torch.Tensor, list[torch.Tensor | None] | torch.Tensor]
        ] = []
        self.reset_calls: list[int] = []
        self.create_calls: int = 0
        self._n_active_adapters: int = 0

    def set_lora(self, index: int, lora_a, lora_b):
        self.set_calls.append((index, lora_a, lora_b))

    def reset_lora(self, index: int):
        self.reset_calls.append(index)

    def create_lora_weights(self, max_loras, lora_config, model_config):
        self.create_calls += 1


def fake_replace_submodule(
    root: torch.nn.Module,
    module_name: str,
    submodule: torch.nn.Module,
    replace_calls: list[str] | None = None,
) -> None:
    """Replace a submodule by traversing dotted paths correctly."""
    if replace_calls is not None:
        replace_calls.append(module_name)
    parts = module_name.split(".")
    parent = root
    for attr in parts[:-1]:
        parent = getattr(parent, attr)
    setattr(parent, parts[-1], submodule)
