# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""What is allowed to keep the CFM wrapper from capturing a graph.

These run without a GPU: ``_capture`` is mocked, so nothing here touches a
CUDA context. The point is the decision made *before* capture, not the capture.
"""

from typing import NamedTuple
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.cuda_graph_wrapper import (
    CFMGraphWrapper,
    _format_memory_delta,
    _memory_snapshot,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _InputSignature(NamedTuple):
    """The three properties ``replay`` reads from an input before capturing.

    Everything ``replay`` does ahead of the capture decision -- the device
    check and the cache key -- comes from these, and ``_capture`` is mocked
    out, so the tests need no CUDA tensors and no GPU.
    """

    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


def _cfm_input_signatures(chunk_size: int) -> tuple[_InputSignature, ...]:
    device = torch.device("cuda:0")
    shapes = ((2, 16, chunk_size), (2, 1, 8), (2, 2, 8, 2), (2, 2, 1, 4, 8), (2, 2, 8, 2), (2, 2, 1, 12, 8))
    return tuple(_InputSignature(shape, torch.float32, device) for shape in shapes)


def _wrapper(monkeypatch: pytest.MonkeyPatch, *, max_graphs: int = 4) -> CFMGraphWrapper:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    wrapper = object.__new__(CFMGraphWrapper)
    wrapper.max_graphs = max_graphs
    wrapper.enabled = True
    wrapper.graph_fn = Mock(return_value=torch.tensor([42.0]))
    wrapper.device = torch.device("cuda:0")
    wrapper._cache = {}
    wrapper._unsupported = set()
    wrapper._stats = {"calls": 0, "hits": 0, "captures": 0, "flushes": 0, "eager": 0}
    wrapper._capture = Mock(return_value=None)
    return wrapper


def test_capture_is_not_gated_on_free_device_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whether to capture must not be decided from free device memory.

    A capture is served by the caching allocator out of its private graph
    pool, and the allocator falls back on memory it has already reserved. Free
    device memory is therefore neither an upper nor a lower bound on what a
    capture can draw, and a threshold on it decides the wrong question -- one
    whose answer also silently varies with card size and with whatever else
    shares the device. Anything consulting it here would fail this test.
    """
    import vllm_omni.platforms as platforms_module

    def _saturated(device: torch.device | None = None) -> int:
        raise AssertionError("capture must not depend on device-level free memory")

    # Setting the attribute on the module shadows the lazy ``__getattr__``, so
    # this neither builds nor requires a real platform singleton.
    monkeypatch.setattr(
        platforms_module,
        "current_omni_platform",
        Mock(get_free_memory=_saturated),
        raising=False,
    )

    wrapper = _wrapper(monkeypatch)
    wrapper.replay(*_cfm_input_signatures(10))

    wrapper._capture.assert_called_once()


def test_memory_reporting_is_inert_off_cuda() -> None:
    """Capture logging must not break the platforms that have no such counters."""
    assert _memory_snapshot(torch.device("cpu")) is None
    assert _format_memory_delta(None, (1, 2)) == ""
    assert _format_memory_delta((1, 2), None) == ""
    assert "allocated" in _format_memory_delta((0, 0), (1 << 20, 2 << 20))
