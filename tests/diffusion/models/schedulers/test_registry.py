# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the diffusion scheduler registry and construction seam.

Covers ``vllm_omni.diffusion.models.schedulers.registry``: name registration
(direct + decorator), class resolution (registry / entry points / dotted
path), the ``build_pipeline_scheduler`` resolution order, kwargs merging, the
bit-identical default fallback, and the documented injected-scheduler
contract.
"""

import copy
import importlib.metadata
import inspect
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import Mock

import pytest

import vllm_omni.diffusion.models.schedulers.registry as registry_mod
from vllm_omni.diffusion.models.schedulers import (
    FlowMatchEulerDiscreteScheduler,
    build_pipeline_scheduler,
    ensure_scheduler_consumed,
    is_injected_scheduler,
    register_scheduler,
    resolve_scheduler_cls,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class RecordingMockScheduler:
    """Minimal scheduler satisfying the documented injection contract.

    Records every ``from_pretrained`` call in the class-level ``constructed``
    list so tests can inspect the exact construction arguments.
    """

    constructed: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, **config: Any) -> None:
        self.config = dict(config)
        self.timesteps: list[int] = []
        self._begin_index = 0

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        subfolder: str | None = None,
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> "RecordingMockScheduler":
        cls.constructed.append(
            {
                "model": model,
                "subfolder": subfolder,
                "local_files_only": local_files_only,
                "kwargs": kwargs,
            }
        )
        return cls(**kwargs)

    def step(self, noise_pred, t, latents, return_dict=False, generator=None):
        return (latents,)

    def set_timesteps(self, num_inference_steps: int, device=None) -> None:
        self.timesteps = list(range(num_inference_steps))

    def set_begin_index(self, begin_index: int = 0) -> None:
        self._begin_index = begin_index


class OtherMockScheduler(RecordingMockScheduler):
    """Second registrable class, distinguished by type."""


@pytest.fixture(autouse=True)
def _isolate_registry(monkeypatch):
    """Snapshot the module-global registry and skip real entry-point scans."""
    snapshot = dict(registry_mod._SCHEDULER_REGISTRY)
    monkeypatch.setattr(registry_mod, "_entry_points_loaded", True)
    monkeypatch.setattr(registry_mod, "_consumed_scheduler_config_ids", set())
    RecordingMockScheduler.constructed.clear()
    yield
    registry_mod._SCHEDULER_REGISTRY.clear()
    registry_mod._SCHEDULER_REGISTRY.update(snapshot)


def _od_config(**overrides) -> SimpleNamespace:
    """Stub OmniDiffusionConfig (duck-typed; the factory only uses getattr)."""
    base = {
        "model": "stub-model",
        "scheduler": None,
        "scheduler_kwargs": None,
        "local_files_only": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def assert_scheduler_contract(scheduler: Any) -> None:
    """Validate the documented injected-scheduler contract (registry.py docstring)."""
    sig = inspect.signature(scheduler.step)
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert len(positional) >= 3, "step() must accept (noise_pred, t, latents) positionally"
    assert "return_dict" in sig.parameters, "step() must accept return_dict"
    assert "generator" in sig.parameters, "step() must accept generator"
    assert callable(getattr(scheduler, "set_timesteps", None)), "set_timesteps() required"
    assert callable(getattr(scheduler, "set_begin_index", None)), "set_begin_index() required"
    assert hasattr(scheduler, "config"), ".config attribute required"
    copy.deepcopy(scheduler)  # must be deepcopy-safe


class TestRegisterAndResolve:
    def test_register_direct_call(self):
        register_scheduler("mock_direct", RecordingMockScheduler)
        assert resolve_scheduler_cls("mock_direct") is RecordingMockScheduler

    def test_register_decorator(self):
        @register_scheduler("mock_decorated")
        class DecoratedScheduler(RecordingMockScheduler):
            pass

        assert resolve_scheduler_cls("mock_decorated") is DecoratedScheduler

    def test_resolve_none_passthrough(self):
        assert resolve_scheduler_cls(None) is None

    def test_resolve_class_passthrough(self):
        assert resolve_scheduler_cls(RecordingMockScheduler) is RecordingMockScheduler

    def test_resolve_by_dotted_path(self):
        ref = (
            "vllm_omni.diffusion.models.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler"
        )
        assert resolve_scheduler_cls(ref) is FlowMatchEulerDiscreteScheduler

    def test_resolve_unknown_bare_name_raises_keyerror(self):
        register_scheduler("mock_known", RecordingMockScheduler)
        with pytest.raises(KeyError, match="Unknown scheduler 'nope'.*mock_known"):
            resolve_scheduler_cls("nope")

    def test_resolve_unknown_dotted_path_raises(self):
        with pytest.raises(ModuleNotFoundError):
            resolve_scheduler_cls("no.such.module.Scheduler")


class _FakeEntryPoint:
    def __init__(self, name: str, cls: type) -> None:
        self.name = name
        self._cls = cls

    def load(self) -> type:
        return self._cls


class TestEntryPoints:
    def _patch_entry_points(self, monkeypatch, eps: list[_FakeEntryPoint]) -> None:
        monkeypatch.setattr(importlib.metadata, "entry_points", lambda group=None: eps)
        monkeypatch.setattr(registry_mod, "_entry_points_loaded", False)

    def test_entry_point_registered_and_resolvable(self, monkeypatch):
        self._patch_entry_points(monkeypatch, [_FakeEntryPoint("ep_scheduler", RecordingMockScheduler)])
        assert resolve_scheduler_cls("ep_scheduler") is RecordingMockScheduler

    def test_entry_points_loaded_only_once(self, monkeypatch):
        calls = []

        def _fake_entry_points(group=None):
            calls.append(group)
            return [_FakeEntryPoint("ep_scheduler", RecordingMockScheduler)]

        monkeypatch.setattr(importlib.metadata, "entry_points", _fake_entry_points)
        monkeypatch.setattr(registry_mod, "_entry_points_loaded", False)
        resolve_scheduler_cls("ep_scheduler")
        resolve_scheduler_cls("ep_scheduler")
        assert calls == [registry_mod.SCHEDULER_ENTRY_POINT_GROUP]

    def test_direct_registration_wins_over_entry_point(self, monkeypatch):
        register_scheduler("ep_scheduler", OtherMockScheduler)
        self._patch_entry_points(monkeypatch, [_FakeEntryPoint("ep_scheduler", RecordingMockScheduler)])
        assert resolve_scheduler_cls("ep_scheduler") is OtherMockScheduler


class TestBuildPipelineScheduler:
    def test_explicit_arg_beats_config(self):
        register_scheduler("config_sched", RecordingMockScheduler)
        od_config = _od_config(scheduler="config_sched")
        result = build_pipeline_scheduler(od_config, scheduler_cls=OtherMockScheduler)
        assert isinstance(result, OtherMockScheduler)

    def test_config_beats_default_builder(self):
        register_scheduler("config_sched", RecordingMockScheduler)
        default_builder = Mock(return_value=object())
        result = build_pipeline_scheduler(_od_config(scheduler="config_sched"), default_builder=default_builder)
        assert isinstance(result, RecordingMockScheduler)
        default_builder.assert_not_called()

    def test_default_fallback_bit_identical(self):
        sentinel = object()
        default_builder = Mock(return_value=sentinel)
        result = build_pipeline_scheduler(_od_config(), default_builder=default_builder)
        assert result is sentinel, "default path must pass the builder's return value through untouched"
        default_builder.assert_called_once_with()
        assert RecordingMockScheduler.constructed == []

    def test_construction_args_match_stock_sites(self):
        register_scheduler("config_sched", RecordingMockScheduler)
        od_config = _od_config(scheduler="config_sched", model="/local/model")
        build_pipeline_scheduler(od_config, local_files_only=True, revision="abc123")
        assert RecordingMockScheduler.constructed == [
            {
                "model": "/local/model",
                "subfolder": "scheduler",
                "local_files_only": True,
                "kwargs": {"revision": "abc123"},
            }
        ]

    def test_missing_default_builder_raises_valueerror(self):
        with pytest.raises(ValueError, match="No scheduler configured"):
            build_pipeline_scheduler(_od_config())

    def test_od_config_local_files_only_attr_is_ignored(self):
        register_scheduler("config_sched", RecordingMockScheduler)
        od_config = _od_config(scheduler="config_sched", model="hub/model", local_files_only=True)
        build_pipeline_scheduler(od_config, local_files_only=False)
        assert RecordingMockScheduler.constructed[-1]["local_files_only"] is False

    def test_ensure_scheduler_consumed_raises_when_unwired(self):
        od_config = _od_config(scheduler="config_sched")
        with pytest.raises(ValueError, match="does not consume"):
            ensure_scheduler_consumed(od_config, object())

    def test_ensure_scheduler_consumed_passes_after_factory(self):
        register_scheduler("config_sched", RecordingMockScheduler)
        od_config = _od_config(scheduler="config_sched")
        build_pipeline_scheduler(od_config, local_files_only=False)
        ensure_scheduler_consumed(od_config, object())

    def test_is_injected_scheduler(self):
        assert is_injected_scheduler(_od_config(scheduler="x"))
        assert not is_injected_scheduler(_od_config())

    def test_kwargs_merging_explicit_over_config(self):
        register_scheduler("config_sched", RecordingMockScheduler)
        od_config = _od_config(
            scheduler="config_sched",
            scheduler_kwargs={"shift": 1.0, "num_train_timesteps": 1000},
        )
        result = build_pipeline_scheduler(od_config, scheduler_kwargs={"shift": 3.0})
        assert RecordingMockScheduler.constructed[-1]["kwargs"] == {"shift": 3.0, "num_train_timesteps": 1000}
        assert result.config == {"shift": 3.0, "num_train_timesteps": 1000}

    def test_config_accepts_dotted_path(self):
        od_config = _od_config(scheduler=f"{__name__}.RecordingMockScheduler")
        default_builder = Mock()
        result = build_pipeline_scheduler(od_config, default_builder=default_builder)
        assert isinstance(result, RecordingMockScheduler)
        default_builder.assert_not_called()


class TestSchedulerContract:
    def test_mock_satisfies_contract(self):
        scheduler = RecordingMockScheduler(num_train_timesteps=1000)
        assert_scheduler_contract(scheduler)

    def test_mock_step_and_timesteps_behavior(self):
        scheduler = RecordingMockScheduler()
        scheduler.set_timesteps(4)
        assert scheduler.timesteps == [0, 1, 2, 3]
        scheduler.set_begin_index(1)
        assert scheduler._begin_index == 1
        latents = object()
        assert scheduler.step("noise", 0, latents, return_dict=False) == (latents,)

    def test_deepcopy_gives_independent_state(self):
        scheduler = RecordingMockScheduler()
        scheduler.set_timesteps(4)
        clone = copy.deepcopy(scheduler)
        clone.set_timesteps(2)
        assert scheduler.timesteps == [0, 1, 2, 3]
        assert clone.timesteps == [0, 1]

    def test_stock_flow_match_scheduler_satisfies_contract(self):
        # The vendored stock scheduler must keep satisfying the contract that
        # injected schedulers are held to (it is deep-copied per request in
        # step-wise execution).
        assert_scheduler_contract(FlowMatchEulerDiscreteScheduler())

    def test_set_timesteps_advertises_sigmas_for_retrieve_timesteps(self):
        # Qwen-Image always synthesizes a custom sigma schedule, then
        # retrieve_timesteps inspects set_timesteps for a named ``sigmas``.
        scheduler = FlowMatchEulerDiscreteScheduler()
        assert "sigmas" in inspect.signature(scheduler.set_timesteps).parameters

    def test_e2e_helper_preserves_sigmas_on_set_timesteps(self):
        from tests.e2e.features.helpers.custom_scheduler import FlowMatchEulerDiscreteSchedulerForTest

        scheduler = FlowMatchEulerDiscreteSchedulerForTest()
        params = inspect.signature(scheduler.set_timesteps).parameters
        assert "sigmas" in params
        assert "timesteps" in params
