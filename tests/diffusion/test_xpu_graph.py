# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""XPUGraphedForward on CPU, with the XPU primitives stubbed.

Keying, capture bookkeeping, LRU eviction and the eager fallback run
unchanged; real XPU graphs are covered by ``test_graphed_forward_device.py``.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import vllm_omni.diffusion.cuda_graph as cuda_graph_mod
import vllm_omni.diffusion.profiler.diffusion_pipeline_profiler as profiler_mod
import vllm_omni.diffusion.xpu_graph as xpu_graph_mod
from vllm_omni.diffusion.cuda_graph import GraphedForward
from vllm_omni.diffusion.xpu_graph import XPUGraphedForward

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _Doubler(nn.Module):
    def forward(self, *, x: torch.Tensor) -> torch.Tensor:
        return x * 2


def _no_cuda(*args, **kwargs):
    raise AssertionError("the XPU capture must not touch torch.cuda")


@pytest.fixture
def platform(monkeypatch) -> SimpleNamespace:
    stream = SimpleNamespace(wait_stream=lambda other: None)
    monkeypatch.setattr(torch.xpu, "Stream", lambda **kwargs: stream, raising=False)
    monkeypatch.setattr(torch.xpu, "current_stream", lambda device: stream, raising=False)
    monkeypatch.setattr(torch.xpu, "stream", lambda side: nullcontext(), raising=False)
    for name in ("get_sync_debug_mode", "set_sync_debug_mode", "Stream", "graph", "CUDAGraph"):
        monkeypatch.setattr(torch.cuda, name, _no_cuda)
    fake = SimpleNamespace(captures=0, outer_capture=False)
    fake.is_graph_capturing = lambda: fake.outer_capture

    def capture_graph(capture, generators=()):
        capture()
        fake.captures += 1
        return SimpleNamespace(replay=lambda: None)

    fake.capture_graph = capture_graph
    monkeypatch.setattr(xpu_graph_mod, "current_omni_platform", fake)
    monkeypatch.setattr(cuda_graph_mod, "current_omni_platform", fake)
    return fake


def _forced_on(**kwargs) -> XPUGraphedForward:
    graph = XPUGraphedForward(_Doubler(), device=torch.device("cpu"), warmup_iters=1, **kwargs)
    assert not graph.enabled, "CPU is not an XPU device"
    graph.enabled = True
    return graph


def test_captures_through_the_platform_with_lru_eviction(platform) -> None:
    graph = _forced_on(max_entries=2)
    for rows in (2, 3, 4):
        torch.testing.assert_close(graph(x=torch.ones(rows)), torch.full((rows,), 2.0))

    assert platform.captures == 3
    stats = graph.stats()
    assert (stats["graphs"], stats["evictions"], stats["eager_calls"]) == (2, 1, 0)


def test_a_platform_that_cannot_capture_turns_graphs_off(platform) -> None:
    platform.capture_graph = lambda capture, generators=(): None
    graph = _forced_on()

    torch.testing.assert_close(graph(x=torch.ones(2)), torch.full((2,), 2.0))

    assert not graph.enabled
    assert graph.stats()["captures"] == 0
    assert graph.stats()["eager_calls"] == 1


def test_base_platform_returns_the_cuda_graphed_forward() -> None:
    from vllm_omni.platforms.interface import OmniPlatform

    assert OmniPlatform.get_graphed_forward_cls() is GraphedForward


def test_xpu_platform_returns_the_xpu_graphed_forward() -> None:
    # vLLM's XPUPlatform imports vllm_xpu_kernels at module scope, and that
    # package is absent on non-XPU images.
    pytest.importorskip("vllm_xpu_kernels")
    from vllm_omni.platforms.xpu.platform import XPUOmniPlatform

    assert XPUOmniPlatform.get_graphed_forward_cls() is XPUGraphedForward


def test_capture_context_wraps_the_capture_forwards(platform) -> None:
    entered = []

    @contextmanager
    def context():
        entered.append(True)
        yield

    graph = XPUGraphedForward(_Doubler(), device=torch.device("cpu"), warmup_iters=1, capture_context=context)
    graph.enabled = True
    graph(x=torch.ones(2))
    graph(x=torch.ones(2))  # replay: no capture, no context

    assert entered == [True] and platform.captures == 1


def test_runs_eager_inside_an_outer_capture(platform) -> None:
    graph = _forced_on()
    platform.outer_capture = True

    torch.testing.assert_close(graph(x=torch.ones(2)), torch.full((2,), 2.0))

    assert graph.enabled and graph.num_graphs == 0 and graph.stats()["eager_calls"] == 1
    assert platform.captures == 0


@pytest.mark.parametrize(("cuda_alike", "stream_capturing", "expected"), [(False, True, False), (True, True, True)])
def test_base_platform_reports_cuda_stream_capture(monkeypatch, cuda_alike, stream_capturing, expected) -> None:
    from vllm_omni.platforms.interface import OmniPlatform

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: stream_capturing)
    platform = SimpleNamespace(is_cuda_alike=lambda: cuda_alike)

    assert OmniPlatform.is_graph_capturing(platform) is expected


@pytest.mark.parametrize(("capturing", "expected_syncs"), [(False, 1), (True, 0)])
def test_profiler_skips_synchronize_during_capture(monkeypatch, capturing, expected_syncs):
    syncs = []
    monkeypatch.setattr(
        profiler_mod,
        "current_omni_platform",
        SimpleNamespace(
            is_available=lambda: True,
            is_graph_capturing=lambda: capturing,
            synchronize=lambda: syncs.append(1),
        ),
    )

    profiler_mod._sync_if_useful()

    assert len(syncs) == expected_syncs
