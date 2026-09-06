# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for AR-Diffusion runner device resolution (platform-neutral KV sizing)."""

from __future__ import annotations

import pytest
import torch

from vllm_omni.experimental.ar_diffusion import runner as runner_mod
from vllm_omni.experimental.ar_diffusion.runner import ARDiffusionModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

FREE = 7 << 30


def make_runner(device: str | None) -> ARDiffusionModelRunner:
    runner = object.__new__(ARDiffusionModelRunner)
    runner.device = None if device is None else torch.device(device)
    return runner


@pytest.fixture
def platform_free_memory(monkeypatch):
    """Record the device the platform layer is asked about, and answer with FREE."""
    seen: list[torch.device] = []

    class FakePlatform:
        @staticmethod
        def get_free_memory(device=None):
            seen.append(device)
            return FREE

    monkeypatch.setattr(runner_mod, "current_omni_platform", FakePlatform)
    return seen


@pytest.mark.parametrize("device", ["xpu:0", "cuda:0"])
def test_available_memory_delegates_to_the_platform_layer(platform_free_memory, device: str) -> None:
    assert make_runner(device)._available_memory_bytes() == FREE
    assert platform_free_memory == [torch.device(device)]


def test_available_memory_requires_a_device(platform_free_memory) -> None:
    with pytest.raises(RuntimeError, match="requires a device"):
        make_runner(None)._available_memory_bytes()

    assert platform_free_memory == []


def test_available_memory_rejects_cpu(platform_free_memory) -> None:
    """Widening the gate to accelerators must not widen it to cpu.

    ``get_free_memory`` is ``NotImplementedError`` on the platform base, so without
    this guard a cpu device surfaces that instead of a message naming the problem.
    """
    with pytest.raises(RuntimeError, match="got cpu"):
        make_runner("cpu")._available_memory_bytes()

    assert platform_free_memory == []


@pytest.mark.parametrize("kind", ["xpu", "cuda"])
def test_accelerator_device_matches_the_active_accelerator(monkeypatch, kind: str) -> None:
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: torch.device(kind))

    assert make_runner(f"{kind}:0")._accelerator_device() == torch.device(f"{kind}:0")
    assert make_runner("cpu")._accelerator_device() is None


def test_accelerator_device_is_none_without_an_accelerator(monkeypatch) -> None:
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: None)

    assert make_runner("xpu:0")._accelerator_device() is None
