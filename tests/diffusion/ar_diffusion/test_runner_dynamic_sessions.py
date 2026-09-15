# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Dynamic request admission across request and stepwise runner execution.

Use lightweight session doubles to test ordering and failure semantics without
allocating device KV pools. Pool allocation is covered by test_kv_cache.py.
"""

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_omni.experimental.ar_diffusion import runner as runner_module
from vllm_omni.experimental.ar_diffusion.capability import (
    ARDiffusionKVBranchSpec,
    ARDiffusionKVCacheSpec,
    ARDiffusionRequestKVSpec,
    ARDiffusionRequestRejectedError,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.config import ARDiffusionKVConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class DynamicPipeline:
    def __init__(self):
        self.spec = ARDiffusionKVCacheSpec(
            num_layers=1,
            num_kv_heads=1,
            head_size=1,
            tokens_per_frame=1,
            frames_per_block=1,
            window_frames=2,
            session_capacity=2,
            kv_branches=(ARDiffusionKVBranchSpec("main", 0),),
        )
        self.bound = None
        self.closed = []
        self.reset = []

    def ar_diffusion_default_request_spec(self):
        return ARDiffusionRequestKVSpec(self.spec, "small")

    def ar_diffusion_request_spec(self, request):
        return request.geometry

    def ar_diffusion_worst_case_request_specs(self):
        return (self.ar_diffusion_default_request_spec(),)

    def validate_ar_diffusion_effective_spec(self, spec):
        pass

    @contextmanager
    def bind_ar_diffusion_state(self, session_id, state):
        self.bound = state
        try:
            yield
        finally:
            self.bound = None

    def reset_ar_diffusion_session(self, session_id):
        self.reset.append(session_id)

    def close_ar_diffusion_session(self, session_id):
        self.closed.append(session_id)


@pytest.fixture
def runner(monkeypatch):
    pipeline = DynamicPipeline()
    value = object.__new__(runner_module.ARDiffusionModelRunner)
    value.pipeline = pipeline
    value._ar_diffusion_capability = pipeline
    value._ar_diffusion_kv_cache_spec = pipeline.spec
    value.ar_diffusion_kv_config = ARDiffusionKVConfig(enable=True)
    value.kv_cache = object()
    value._sessions = OrderedDict()
    value._session_geometry_keys = {}
    value._session_capacity = 2
    value._perf_e2e_times = []
    value._stepwise_chunk_started = {}
    value.state_cache = {}
    monkeypatch.setattr(value, "_new_session_state", lambda sid: SimpleNamespace(close=Mock()))
    monkeypatch.setattr(runner_module, "supports_step_execution", lambda pipeline: True)
    monkeypatch.setattr(runner_module, "current_omni_platform", SimpleNamespace(synchronize=Mock(), empty_cache=Mock()))
    return value


def request(runner, *, geometry="small", tokens=1, reset=False):
    return SimpleNamespace(
        sampling_params=SimpleNamespace(extra_args={"session_id": "s1", "reset": reset}),
        geometry=ARDiffusionRequestKVSpec(replace(runner.pipeline.spec, tokens_per_frame=tokens), geometry),
    )


def wave(req=None, *, finished=False):
    return SimpleNamespace(
        scheduled_request_ids=[] if finished else ["s1"],
        scheduled_new_reqs=[] if req is None else [SimpleNamespace(request_id="s1", req=req)],
        finished_req_ids={"s1"} if finished else set(),
    )


def test_request_geometry_rejection_preserves_existing_session(runner, monkeypatch):
    original = runner._get_or_create_session("s1", "small")
    forward = Mock(return_value=object())
    monkeypatch.setattr(runner_module.DiffusionModelRunner, "execute_model", forward)
    with pytest.raises(ARDiffusionRequestRejectedError, match="reset is required"):
        runner.execute_model(request(runner, geometry="wide"))
    assert runner._sessions["s1"] is original
    original.close.assert_not_called()
    forward.assert_not_called()
    assert not runner.pipeline.closed


def test_request_reset_validates_before_releasing_history(runner, monkeypatch):
    original = runner._get_or_create_session("s1", "small")
    monkeypatch.setattr(runner.pipeline, "ar_diffusion_request_spec", Mock(side_effect=ValueError("bad geometry")))
    with pytest.raises(ARDiffusionRequestRejectedError, match="bad geometry"):
        runner.execute_model(request(runner, reset=True))
    assert runner._sessions["s1"] is original
    assert not runner.pipeline.reset


def test_request_reset_binds_fresh_state_with_new_geometry(runner, monkeypatch):
    original = runner._get_or_create_session("s1", "small")
    seen = []
    monkeypatch.setattr(
        runner_module.DiffusionModelRunner,
        "execute_model",
        lambda *args, **kwargs: seen.append(runner.pipeline.bound),
    )
    runner.execute_model(request(runner, geometry="wide", reset=True))
    original.close.assert_called_once()
    assert seen == [runner._sessions["s1"]]
    assert seen[0] is not original
    assert runner._session_geometry_keys["s1"] == "wide"
    assert runner.pipeline.bound is None


@pytest.mark.parametrize("rejected", [True, False])
def test_request_admission_rejection_retains_history_but_execution_failure_releases_it(runner, monkeypatch, rejected):
    original = runner._get_or_create_session("s1", "small")
    error = ARDiffusionRequestRejectedError("rejected") if rejected else RuntimeError("forward failed")
    monkeypatch.setattr(runner_module.DiffusionModelRunner, "execute_model", Mock(side_effect=error))
    with pytest.raises(type(error)):
        runner.execute_model(request(runner))
    assert runner.pipeline.bound is None
    assert ("s1" in runner._sessions) is rejected
    assert ("s1" in runner._session_geometry_keys) is rejected
    assert original.close.call_count == (0 if rejected else 1)
    assert not runner._perf_e2e_times


@pytest.mark.parametrize("stepwise", [False, True])
def test_permanent_cache_failure_never_falls_back_to_base_runner(runner, monkeypatch, stepwise):
    runner.kv_cache = None
    runner._cache_failure = RuntimeError("allocation failed")
    method = "execute_stepwise" if stepwise else "execute_model"
    base = Mock()
    monkeypatch.setattr(runner_module.DiffusionModelRunner, method, base)
    with pytest.raises(RuntimeError, match="permanently failed"):
        getattr(runner, method)(wave() if stepwise else request(runner))
    base.assert_not_called()


@pytest.mark.parametrize("stepwise", [False, True])
def test_geometry_rebuild_happens_before_binding(runner, monkeypatch, stepwise):
    events = []

    def rebuild(spec):
        assert runner.pipeline.bound is None
        events.append("rebuild")
        runner._ar_diffusion_kv_cache_spec = spec

    monkeypatch.setattr(runner, "_rebuild_for_spec", rebuild)

    def forward(*args, **kwargs):
        assert runner.pipeline.bound is not None
        assert runner._ar_diffusion_kv_cache_spec.tokens_per_frame == 2
        events.append("forward")
        return SimpleNamespace(get_request_output=lambda rid: None)

    method = "execute_stepwise" if stepwise else "execute_model"
    monkeypatch.setattr(runner_module.DiffusionModelRunner, method, forward)
    req = request(runner, geometry="large", tokens=2)
    getattr(runner, method)(wave(req) if stepwise else req)
    assert events == ["rebuild", "forward"]
    assert runner._session_geometry_keys["s1"] == "large"


def test_stepwise_reuses_admitted_geometry_and_retires_session(runner, monkeypatch):
    seen = []

    def forward(*args):
        seen.append(runner.pipeline.bound)
        return SimpleNamespace(get_request_output=lambda rid: None)

    monkeypatch.setattr(runner_module.DiffusionModelRunner, "execute_stepwise", forward)
    runner.execute_stepwise(wave(request(runner, geometry="wide")))
    runner.execute_stepwise(wave())
    assert seen[0] is seen[1]
    assert runner._session_geometry_keys["s1"] == "wide"
    runner.execute_stepwise(wave(finished=True))
    seen[0].close.assert_called_once()
    assert not runner._sessions
    assert not runner._session_geometry_keys
    assert not runner._stepwise_chunk_started


def test_dynamic_stepwise_cannot_resume_after_losing_kv_session(runner, monkeypatch):
    forward = Mock()
    monkeypatch.setattr(runner_module.DiffusionModelRunner, "execute_stepwise", forward)
    with pytest.raises(RuntimeError, match="no resident KV session"):
        runner.execute_stepwise(wave())
    forward.assert_not_called()


def test_stepwise_chunk_barrier_failure_clears_geometry_and_cached_state(runner, monkeypatch):
    result = SimpleNamespace(finished=False, result=object())
    monkeypatch.setattr(
        runner_module.DiffusionModelRunner,
        "execute_stepwise",
        Mock(return_value=SimpleNamespace(get_request_output=lambda rid: result)),
    )
    runner.state_cache["s1"] = object()
    runner_module.current_omni_platform.synchronize.side_effect = RuntimeError("device failed")
    with pytest.raises(RuntimeError, match="device failed"):
        runner.execute_stepwise(wave(request(runner)))
    assert not runner._sessions
    assert not runner._session_geometry_keys
    assert not runner.state_cache
    assert not runner._stepwise_chunk_started
