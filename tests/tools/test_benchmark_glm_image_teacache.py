# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for GLM-Image TeaCache benchmark session reuse / warmup order."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.teacache.benchmark_glm_image import OmniSession, warm_and_time

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeOmni:
    constructions = 0
    closes = 0

    def __init__(self, **kwargs):
        type(self).constructions += 1
        self.kwargs = kwargs
        self.closed = False
        self.calls: list[str] = []

    def generate(self, prompt, sampling_params):
        self.calls.append(f"generate:{prompt}")
        return [SimpleNamespace(images=["img"])]

    def close(self):
        self.closed = True
        type(self).closes += 1


@pytest.fixture(autouse=True)
def _reset_fake_omni():
    _FakeOmni.constructions = 0
    _FakeOmni.closes = 0
    yield


def test_omni_session_warms_before_timed_repeats_on_one_engine():
    with OmniSession(_FakeOmni, model="fake", cache_backend="tea_cache") as session:
        assert _FakeOmni.constructions == 1
        times, last = warm_and_time(
            session,
            prompt="p0",
            sampling_params=SimpleNamespace(),
            repeats=3,
        )
        # 1 warm + 3 timed generates on the same instance.
        assert session.generate_calls == 4
        assert len(times) == 3
        assert last[0].images == ["img"]
        assert session.omni.calls[0] == "generate:p0"
        assert len(session.omni.calls) == 4

    assert _FakeOmni.constructions == 1
    assert _FakeOmni.closes == 1
    assert session.omni.closed is True


def test_omni_session_constructs_once_per_config():
    with OmniSession(_FakeOmni, model="fake") as baseline:
        warm_and_time(baseline, prompt="a", sampling_params=None, repeats=2)
    with OmniSession(_FakeOmni, model="fake", cache_backend="tea_cache") as cached:
        warm_and_time(cached, prompt="a", sampling_params=None, repeats=2)

    assert _FakeOmni.constructions == 2
    assert _FakeOmni.closes == 2
