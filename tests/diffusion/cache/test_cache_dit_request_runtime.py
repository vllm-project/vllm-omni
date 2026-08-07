# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.diffusion.cache.cachedit import runtime as runtime_module
from vllm_omni.diffusion.cache.cachedit.runtime import (
    CacheDiTRequestSpec,
    RequestScopedCacheDiTRuntime,
)
from vllm_omni.diffusion.data import DiffusionCacheConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _spec(key: str, steps: int = 50) -> CacheDiTRequestSpec:
    return CacheDiTRequestSpec(
        installation_key=key,
        cache_config=DiffusionCacheConfig(),
        num_inference_steps=steps,
    )


def _recording_backend(events):
    class FakeBackend:
        def __init__(self, config):
            events.append(("create", config))
            self.enabled = False

        def enable(self, pipeline):
            events.append(("enable", pipeline))
            self.enabled = True

        def refresh(self, pipeline, num_inference_steps):
            events.append(("refresh", pipeline, num_inference_steps))

        def disable(self, pipeline):
            events.append(("disable", pipeline))
            self.enabled = False

        def is_enabled(self):
            return self.enabled

    return FakeBackend


def test_startup_cache_follows_omit_lossless_high_omit_state_machine(monkeypatch):
    events = []
    FakeBackend = _recording_backend(events)
    monkeypatch.setattr(runtime_module, "CacheDiTBackend", FakeBackend)
    pipeline = object()
    config = DiffusionCacheConfig()
    runtime = RequestScopedCacheDiTRuntime(pipeline)
    startup_backend = FakeBackend(config)
    startup_backend.enabled = True

    runtime.adopt(startup_backend, installation_key="generic")
    runtime.prepare(_spec("generic", 50))  # omitted: keep startup cache
    runtime.prepare(None)  # lossless: uninstall
    runtime.prepare(None)  # repeated lossless: no-op
    runtime.prepare(_spec("high", 40))  # high: install H3 profile
    runtime.prepare(_spec("high", 30))  # repeated high: refresh only
    runtime.prepare(_spec("generic", 20))  # omitted: restore startup profile

    assert [event[0] for event in events] == [
        "create",
        "refresh",
        "disable",
        "create",
        "enable",
        "refresh",
        "refresh",
        "disable",
        "create",
        "enable",
        "refresh",
    ]
    assert events[1][-1] == 50
    assert events[5][-1] == 40
    assert events[6][-1] == 30
    assert events[10][-1] == 20
    assert runtime.is_enabled


def test_high_installs_without_startup_cache_and_lossless_uninstalls(monkeypatch):
    events = []
    monkeypatch.setattr(runtime_module, "CacheDiTBackend", _recording_backend(events))
    pipeline = object()
    runtime = RequestScopedCacheDiTRuntime(pipeline)

    runtime.prepare(_spec("high", 40))
    runtime.prepare(None)

    assert [event[0] for event in events] == [
        "create",
        "enable",
        "refresh",
        "disable",
    ]
    assert events[2][-1] == 40
    assert not runtime.is_enabled
