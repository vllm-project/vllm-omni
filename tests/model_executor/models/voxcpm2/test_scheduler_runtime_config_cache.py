# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The unified-decode-graph predicate is resolved once per scheduler."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.voxcpm2 import scheduler as scheduler_module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _scheduler():
    # ``__new__`` keeps the test off the full engine config: the scheduling
    # policy only reads ``vllm_config``.
    sched = scheduler_module.VoxCPM2OmniARAsyncScheduler.__new__(scheduler_module.VoxCPM2OmniARAsyncScheduler)
    sched.vllm_config = SimpleNamespace()
    return sched


def test_unified_decode_graph_state_is_resolved_once(monkeypatch):
    calls = {"count": 0}

    class _FakeRuntimeConfig:
        @classmethod
        def from_vllm_config(cls, _vllm_config):
            calls["count"] += 1
            return SimpleNamespace(unified_decode_graph_available=lambda **_: True)

    monkeypatch.setattr(scheduler_module, "_VoxCPM2RuntimeConfig", _FakeRuntimeConfig)
    monkeypatch.setattr(
        scheduler_module.omni_platform,
        "current_omni_platform",
        SimpleNamespace(is_cuda=lambda: True),
    )
    sched = _scheduler()

    assert sched._unified_decode_graph_enabled() is True
    assert sched._unified_decode_graph_enabled() is True
    assert calls["count"] == 1


def test_false_result_is_cached_too(monkeypatch):
    calls = {"count": 0}

    class _FakeRuntimeConfig:
        @classmethod
        def from_vllm_config(cls, _vllm_config):
            calls["count"] += 1
            return SimpleNamespace(unified_decode_graph_available=lambda **_: False)

    monkeypatch.setattr(scheduler_module, "_VoxCPM2RuntimeConfig", _FakeRuntimeConfig)
    monkeypatch.setattr(
        scheduler_module.omni_platform,
        "current_omni_platform",
        SimpleNamespace(is_cuda=lambda: False),
    )
    sched = _scheduler()

    assert sched._unified_decode_graph_enabled() is False
    assert sched._unified_decode_graph_enabled() is False
    assert calls["count"] == 1
