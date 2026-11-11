# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GraphedForward against real device graphs, with the class the platform picks."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tests.helpers.mark import hardware_test
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256))

    def forward(self, *, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _graphed(max_entries: int):
    device = torch.device(current_omni_platform.device_type, 0)
    torch.manual_seed(0)
    model = _MLP().to(device, torch.bfloat16)
    graph = current_omni_platform.get_graphed_forward_cls()(model, device=device, max_entries=max_entries)
    assert graph.enabled
    return model, graph, device


@hardware_test(res={"cuda": "L4", "xpu": "B60"})
@torch.inference_mode()
def test_replay_matches_eager_across_eviction() -> None:
    model, graph, device = _graphed(max_entries=2)
    # Three shapes through a two-graph LRU and back to the first: later
    # captures happen after an earlier graph was freed.
    for rows in (64, 96, 128, 64, 64):
        x = torch.randn(rows, 256, device=device, dtype=torch.bfloat16)
        torch.testing.assert_close(graph(x=x), model(x=x), rtol=0, atol=0)

    assert graph.stats()["captures"] == 4
    assert graph.stats()["evictions"] == 2
    assert graph.stats()["eager_calls"] == 0


class _Deep(nn.Module):
    """Large enough for a graph's driver-side state to show in device memory."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(2048, 2048) for _ in range(32))

    def forward(self, *, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.nn.functional.gelu(layer(x))
        return x


@hardware_test(res={"xpu": "B60"})
@torch.inference_mode()
def test_recapturing_evicted_shapes_keeps_device_memory_bounded() -> None:
    """Shapes that keep changing (varied prompt lengths) recapture on every call.

    Each round recaptures all eight shapes through the two-graph LRU. On a B70,
    device memory rose over the first four rounds (about 130 MiB above eager
    for this model) and then stayed flat however long recapturing went on, while
    ``memory_reserved()`` never moved, so the check reads device-wide used
    memory. A per-capture leak would keep it rising after the warm rounds.
    """
    device = torch.device(current_omni_platform.device_type, 0)
    model = _Deep().to(device, torch.bfloat16)
    graph = current_omni_platform.get_graphed_forward_cls()(model, device=device, max_entries=2)
    shapes = range(1024, 1024 + 8 * 64, 64)

    def one_round() -> int:
        for rows in shapes:
            graph(x=torch.randn(rows, 2048, device=device, dtype=torch.bfloat16))
        current_omni_platform.synchronize()
        free, total = current_omni_platform.get_device_memory(device)
        return total - free

    for _ in range(4):
        one_round()
    settled = one_round()
    for _ in range(4):
        last = one_round()

    assert graph.stats()["captures"] == 9 * len(shapes)
    assert last <= settled + 8 * 2**20, f"device memory grew from {settled} to {last} bytes over 32 recaptures"
