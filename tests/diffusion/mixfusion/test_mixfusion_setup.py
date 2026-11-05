# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Startup wiring tests for ``mixfusion.setup`` (CPU).

The server-level ``enable_mixfusion`` flag only sets the request-level
default; a request may opt in whenever the pipeline declares
``supports_mixfusion``. That makes the scheduler relax the shape key and turns
``InputBatch.latents`` into a list, so the driver has to be mounted for every
capable pipeline, even with the server flag off.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from torch import nn

from vllm_omni.diffusion.hooks.mixfusion import MixFusionRoot
from vllm_omni.diffusion.mixfusion.driver import MixFusionDriver
from vllm_omni.diffusion.mixfusion.setup import mount_mixfusion, validate_mixfusion_config

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _FakeTransformer(nn.Module):
    def __init__(self, *, adapter: object | None) -> None:
        super().__init__()
        # A root-only plan keeps the fake module free of block structure while
        # still exercising the real hook registration path.
        self._mixfusion_plan = {"": MixFusionRoot()}
        if adapter is not None:
            self.mixfusion_adapter = adapter


class _FakePipeline(nn.Module):
    def __init__(self, transformer: nn.Module | None) -> None:
        super().__init__()
        if transformer is not None:
            self.transformer = transformer


def _od_config(*, enable: bool):
    return SimpleNamespace(enable_mixfusion=enable)


def _mount(*, supports: bool, adapter: object | None, enable: bool = False):
    transformer = _FakeTransformer(adapter=adapter)
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


def test_transformer_without_adapter_gets_a_serial_driver():
    pipeline, transformer = _mount(supports=True, adapter=None)

    # Ragged batches still need the serial fallback instead of reaching the
    # dense path with a list.
    assert pipeline.mixfusion_driver.adapter is None
    assert pipeline.mixfusion_driver.transformer is transformer
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
