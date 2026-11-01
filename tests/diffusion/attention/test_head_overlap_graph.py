# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for the head-overlap NPUGraph cache.

The capture path itself needs an NPU; what must hold everywhere is the fallback
contract: whenever capture is unavailable or has failed, ``run`` returns None so
the caller keeps its eager path instead of raising mid-request.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention import head_overlap_graph as hog

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


_PLENTY_FREE = (64 * 2**30, 64 * 2**30)


class _RaisingNPU(SimpleNamespace):
    def NPUGraph(self):  # noqa: N802 - mirrors the torch.npu API name
        raise RuntimeError("capture refused")


def _fake_npu(free_bytes):
    return _RaisingNPU(synchronize=lambda: None, mem_get_info=lambda: (free_bytes, 64 * 2**30))


def _increment(inputs):
    return inputs[0] + 1


def test_unsupported_platform_falls_back(monkeypatch):
    monkeypatch.setattr(hog, "_supported", lambda: False)
    cache = hog.HeadOverlapGraphCache("t")

    assert cache.run(("k",), (torch.zeros(4),), _increment) is None


def test_disabled_cache_falls_back():
    cache = hog.HeadOverlapGraphCache("t", max_graphs=0)

    assert cache.run(("k",), (torch.zeros(4),), _increment) is None


def test_nested_capture_falls_back(monkeypatch):
    monkeypatch.setattr(hog, "_supported", lambda: True)
    monkeypatch.setattr(hog, "_stream_is_capturing", lambda: True)

    assert hog.HeadOverlapGraphCache("t").run(("k",), (torch.zeros(4),), _increment) is None


def test_capture_failure_disables_the_cache(monkeypatch):
    monkeypatch.setattr(hog, "_supported", lambda: True)
    monkeypatch.setattr(hog, "_stream_is_capturing", lambda: False)
    monkeypatch.setattr(torch, "npu", _fake_npu(_PLENTY_FREE[0]), raising=False)

    cache = hog.HeadOverlapGraphCache("t")

    assert cache.run(("k",), (torch.zeros(4),), _increment) is None
    assert cache._failed is True
    # And it stays off for every later shape, so a bad capture cannot retry-loop.
    assert cache.run(("other",), (torch.zeros(4),), _increment) is None
    assert cache.captures == 0


def test_budget_gate_skips_capture_when_it_would_not_fit(monkeypatch):
    """The whole point of the gate: refuse before allocating, not after OOM-ing."""
    monkeypatch.setattr(hog, "_supported", lambda: True)
    monkeypatch.setattr(hog, "_stream_is_capturing", lambda: False)
    monkeypatch.setattr(torch, "npu", _fake_npu(free_bytes=1024), raising=False)

    cache = hog.HeadOverlapGraphCache("t")

    assert cache.run(("k",), (torch.zeros(4),), _increment) is None
    assert cache._budget_exhausted is True
    # A budget miss is not a capture failure: the cache is undamaged and a later
    # request on a roomier device may still capture.
    assert cache._failed is False
    assert cache.captures == 0


def test_budget_gate_admits_a_capture_that_fits(monkeypatch):
    monkeypatch.setattr(hog, "_supported", lambda: True)
    monkeypatch.setattr(hog, "_stream_is_capturing", lambda: False)
    monkeypatch.setattr(torch, "npu", _fake_npu(free_bytes=32 * 2**30), raising=False)

    cache = hog.HeadOverlapGraphCache("t")

    # Reaches _capture, which is where the fake NPU refuses.
    assert cache.run(("k",), (torch.zeros(4),), _increment) is None
    assert cache._failed is True
    assert cache._budget_exhausted is False


def test_shared_cache_is_one_instance(monkeypatch):
    monkeypatch.setattr(hog, "_SHARED_CACHE", None)

    first = hog.get_shared_cache()
    second = hog.get_shared_cache()

    assert first is second
