# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared test helpers for diffusion LoRA tests."""

from __future__ import annotations

import torch
from vllm.model_executor.layers.linear import LinearBase


class FakeLinearBase(LinearBase):
    """Minimal LinearBase stub for LoRA layer discovery."""

    def __init__(self):
        torch.nn.Module.__init__(self)


class DummyBaseLayerWithLoRA(torch.nn.Module):
    """Fake LoRA wrapper that records set/reset/create calls."""

    def __init__(self, base_layer: torch.nn.Module):
        super().__init__()
        self.base_layer = base_layer

        self.set_calls: list[
            tuple[list[torch.Tensor | None] | torch.Tensor, list[torch.Tensor | None] | torch.Tensor]
        ] = []
        self.reset_calls: int = 0
        self.suspend_calls: int = 0
        self.resume_calls: int = 0
        self.active_slices: tuple[bool, ...] = ()
        self.suspended_slices: tuple[bool, ...] | None = None
        self.create_calls: int = 0

    def set_lora(self, index: int, lora_a, lora_b):
        assert index == 0
        self.set_calls.append((lora_a, lora_b))
        if isinstance(lora_b, list):
            self.active_slices = tuple(b is not None for b in lora_b)
        else:
            self.active_slices = (True,)
        self.suspended_slices = None

    def reset_lora(self, index: int):
        assert index == 0
        self.reset_calls += 1
        self.active_slices = ()
        self.suspended_slices = None

    def suspend_lora(self) -> None:
        if self.suspended_slices is not None:
            return
        self.suspended_slices = self.active_slices
        self.active_slices = (False,) * len(self.active_slices)
        self.suspend_calls += 1

    def resume_lora(self) -> None:
        if self.suspended_slices is not None:
            self.active_slices = self.suspended_slices
            self.suspended_slices = None
        self.resume_calls += 1

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
