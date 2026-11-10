# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Startup wiring tests for ``mixfusion.setup`` (CPU).

The server-level ``enable_mixfusion`` flag only sets the request-level
default; a request may opt in whenever the pipeline declares
``supports_mixfusion``. Plans are mounted even with the server flag off;
actual capability discovery happens after all runtime wrappers are installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from torch import nn

from vllm_omni.diffusion.hooks.mixfusion import MixFusionRoot
from vllm_omni.diffusion.mixfusion.chunked import ChunkedJointAdapter
from vllm_omni.diffusion.mixfusion.driver import MixFusionDriver
from vllm_omni.diffusion.mixfusion.setup import mount_mixfusion, validate_mixfusion_config

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _FakeTransformer(nn.Module):
    def __init__(self, *, adapter: object | None, plan: dict | None = None) -> None:
        super().__init__()
        # A root-only plan keeps the fake module free of block structure while
        # still exercising the real hook registration path.
        self._mixfusion_plan = {"": MixFusionRoot()} if plan is None else plan
        if adapter is not None:
            self.mixfusion_adapter = adapter


class _FakePipeline(nn.Module):
    def __init__(self, transformer: nn.Module | None) -> None:
        super().__init__()
        if transformer is not None:
            self.transformer = transformer


def _od_config(*, enable: bool):
    return SimpleNamespace(enable_mixfusion=enable)


def _mount(*, supports: bool, adapter: object | None, enable: bool = False, plan: dict | None = None):
    transformer = _FakeTransformer(adapter=adapter, plan=plan)
    pipeline = _FakePipeline(transformer)
    mount_mixfusion(pipeline, _od_config(enable=enable), supported=supports)
    return pipeline, transformer


def test_request_opt_in_mounts_driver_with_server_flag_off():
    adapter = object()
    pipeline, transformer = _mount(supports=True, adapter=adapter)

    driver = pipeline.mixfusion_driver
    assert isinstance(driver, MixFusionDriver)
    assert (driver.adapter, driver.transformer) == (adapter, transformer)
    # Hooks are installed but stay inert until ``armed`` publishes a step.
    assert transformer._mixfusion_hooks_applied is True


def test_unsupported_pipeline_mounts_nothing():
    pipeline, transformer = _mount(supports=False, adapter=object())

    assert not hasattr(pipeline, "mixfusion_driver")
    assert getattr(transformer, "_mixfusion_hooks_applied", False) is False


def test_declarative_plan_mounts_the_framework_adapter():
    pipeline, transformer = _mount(supports=True, adapter=None)

    # No model-side adapter: a MixFusionRoot plan gets the chunked one.
    assert isinstance(pipeline.mixfusion_driver.adapter, ChunkedJointAdapter)
    assert transformer._mixfusion_hooks_applied is True


def test_transformer_without_a_packable_plan_does_not_mount_driver():
    pipeline, transformer = _mount(supports=True, adapter=None, plan={})
    assert not hasattr(pipeline, "mixfusion_driver")
    assert getattr(transformer, "_mixfusion_hooks_applied", False) is False


def test_pipeline_without_transformer_mounts_nothing():
    pipeline = _FakePipeline(None)

    mount_mixfusion(pipeline, _od_config(enable=False), supported=True)

    assert not hasattr(pipeline, "mixfusion_driver")


@pytest.mark.parametrize(
    ("config", "supported", "outcome"),
    [
        ({}, False, False),
        ({"enable_mixfusion": True}, True, True),
        ({"enable_mixfusion": True}, False, "does not support MixFusion"),
        # Scheduler-paged KV is request-level only; MixFusion needs steps.
        ({"enable_mixfusion": True, "diffusion_kv_mode": "PAGED_SCHEDULER"}, True, "paged_scheduler"),
    ],
)
def test_validate_mixfusion_config(config, supported, outcome):
    from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode

    if "diffusion_kv_mode" in config:
        config = {**config, "diffusion_kv_mode": DiffusionKVCacheMode[config["diffusion_kv_mode"]]}
    od_config = SimpleNamespace(model_class_name="Stub", **config)
    lookups: list = []

    def capability(config):
        lookups.append(config)
        return supported

    if isinstance(outcome, str):
        with pytest.raises(ValueError, match=outcome):
            validate_mixfusion_config(od_config, supported=capability)
    else:
        assert validate_mixfusion_config(od_config, supported=capability) is outcome
    # The pipeline class is only resolved when MixFusion is enabled.
    assert len(lookups) == (1 if config.get("enable_mixfusion") else 0)


@dataclass
class _Parallel:
    sequence_parallel_size: int = 1
    cfg_parallel_size: int = 1


def test_capability_is_resolved_after_foreign_hooks_are_installed():
    from vllm_omni.diffusion.mixfusion.setup import inspect_mixfusion_capability

    pipeline, transformer = _mount(supports=True, adapter=None)
    pipeline.prepare_mixfusion_request = lambda *args: None
    pipeline.mixfusion_request_config = lambda: (8, 128, 64)
    assert inspect_mixfusion_capability(pipeline).request_config == (8, 128, 64)
    transformer.register_forward_pre_hook(lambda module, args: args)
    capability = inspect_mixfusion_capability(pipeline)
    assert capability.request_config is None
    assert "foreign hooks" in capability.reason


@pytest.mark.parametrize(
    "parallel,reason", [(_Parallel(2, 1), "sequence parallelism"), (_Parallel(1, 2), "CFG parallel")]
)
def test_parallel_capability_keeps_dense_admission(parallel, reason):
    from vllm_omni.diffusion.mixfusion.setup import inspect_mixfusion_capability

    pipeline, _ = _mount(supports=True, adapter=None)
    pipeline.parallel_config = parallel
    capability = inspect_mixfusion_capability(pipeline)
    assert capability.request_config is None and reason in capability.reason


def test_missing_declared_hooks_is_an_initialization_error():
    from vllm_omni.diffusion.mixfusion.setup import inspect_mixfusion_capability

    pipeline, transformer = _mount(supports=True, adapter=None)
    transformer._mixfusion_hooks_applied = False
    with pytest.raises(RuntimeError, match="not installed"):
        inspect_mixfusion_capability(pipeline)


@pytest.mark.parametrize("consistent", [True, False])
def test_engine_requires_rank_agreement_before_request_preparation(monkeypatch, consistent):
    from unittest.mock import Mock

    import vllm_omni.diffusion.registry as registry
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
    from vllm_omni.diffusion.mixfusion.scheduling import MixFusionCapability
    from vllm_omni.diffusion.mixfusion.setup import initialize_mixfusion_admission

    class Pipeline:
        supports_mixfusion = True
        prepare_mixfusion_request = staticmethod(lambda *args: None)

    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.od_config = _od_config(enable=True)
    engine.execution_mode = "step_batch"
    engine.executor = Mock()
    capability = MixFusionCapability((8, 128, 64), "")
    engine.executor.collective_rpc.return_value = [capability, capability if consistent else MixFusionCapability()]
    monkeypatch.setattr(registry, "resolve_diffusion_pipeline_cls", lambda config: Pipeline)
    if consistent:
        initialize_mixfusion_admission(engine)
        assert engine.mixfusion_capability == capability
        assert engine.mixfusion_prepare_request is Pipeline.prepare_mixfusion_request
    else:
        with pytest.raises(RuntimeError, match="disagree"):
            initialize_mixfusion_admission(engine)
