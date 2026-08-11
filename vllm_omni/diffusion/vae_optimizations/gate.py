# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-scoped gates for optional, non-bit-exact VAE fast paths."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from weakref import WeakKeyDictionary

from torch import nn

from vllm_omni.diffusion.offloader.module_collector import ModuleDiscovery


class VaeFastPathGate:
    """Mutable flag shared by wrappers installed on one VAE instance."""

    __slots__ = ("enabled",)

    def __init__(self) -> None:
        self.enabled = False


_VAE_FAST_PATH_GATES: WeakKeyDictionary[nn.Module, VaeFastPathGate] = WeakKeyDictionary()


def register_vae_fast_path_gate(vae: nn.Module, gate: VaeFastPathGate) -> None:
    """Associate an installed fast path with its owning VAE."""

    _VAE_FAST_PATH_GATES[vae] = gate


@contextmanager
def use_vae_fast_path(vae: nn.Module, enabled: bool) -> Iterator[None]:
    """Enable an installed fast path for one decode and always restore it."""

    gate = _VAE_FAST_PATH_GATES.get(vae)
    if gate is None:
        yield
        return

    previous_enabled = gate.enabled
    gate.enabled = enabled
    try:
        yield
    finally:
        gate.enabled = previous_enabled


@contextmanager
def use_pipeline_vae_fast_path(pipeline: nn.Module, enabled: bool) -> Iterator[None]:
    """Apply one request's quality choice to every declared VAE component."""

    vaes = ModuleDiscovery.discover(pipeline).vaes
    with ExitStack() as stack:
        for vae in vaes:
            stack.enter_context(use_vae_fast_path(vae, enabled))
        yield


__all__ = [
    "VaeFastPathGate",
    "register_vae_fast_path_gate",
    "use_pipeline_vae_fast_path",
    "use_vae_fast_path",
]
