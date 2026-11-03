# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Mounting tests for ``_apply_mixfusion_if_enabled`` (CPU).

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

from vllm_omni.diffusion import registry
from vllm_omni.diffusion.hooks.mixfusion import MixFusionRoot
from vllm_omni.diffusion.mixfusion.driver import MixFusionDriver

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


def _od_config(*, enable: bool, sp_size: int = 1):
    return SimpleNamespace(
        enable_mixfusion=enable,
        parallel_config=SimpleNamespace(sequence_parallel_size=sp_size),
    )


def _mount(monkeypatch, *, enable: bool, supports: bool, sp_size: int = 1, adapter: object | None = object()):
    monkeypatch.setattr(registry, "supports_mixfusion", lambda od_config: supports)
    transformer = _FakeTransformer(adapter=adapter)
    pipeline = _FakePipeline(transformer)
    registry._apply_mixfusion_if_enabled(pipeline, _od_config(enable=enable, sp_size=sp_size))
    return pipeline, transformer


def test_request_opt_in_mounts_driver_with_server_flag_off(monkeypatch):
    adapter = object()
    pipeline, transformer = _mount(
        monkeypatch,
        enable=False,
        supports=True,
        adapter=adapter,
    )

    driver = getattr(pipeline, "mixfusion_driver", None)
    assert isinstance(driver, MixFusionDriver)
    assert driver.adapter is adapter
    assert driver.transformer is transformer
    # Hooks are installed but stay inert until ``armed`` publishes a step.
    assert transformer._mixfusion_hooks_applied is True


def test_unsupported_pipeline_mounts_nothing(monkeypatch):
    pipeline, transformer = _mount(monkeypatch, enable=False, supports=False, adapter=object())

    assert not hasattr(pipeline, "mixfusion_driver")
    assert getattr(transformer, "_mixfusion_hooks_applied", False) is False


def test_sequence_parallel_mounts_fallback_driver_only(monkeypatch):
    adapter = object()
    pipeline, transformer = _mount(
        monkeypatch,
        enable=False,
        supports=True,
        sp_size=2,
        adapter=adapter,
    )

    driver = getattr(pipeline, "mixfusion_driver", None)
    assert isinstance(driver, MixFusionDriver)
    # SP cannot be combined with packing, but ragged requests still need the
    # serial fallback instead of reaching the dense path with a list.
    assert driver.adapter is None
    assert driver.transformer is transformer
    assert getattr(transformer, "_mixfusion_hooks_applied", False) is False


def test_server_flag_still_mounts_for_unsupported_pipeline(monkeypatch):
    pipeline, _ = _mount(monkeypatch, enable=True, supports=False, adapter=None)

    driver = getattr(pipeline, "mixfusion_driver", None)
    assert isinstance(driver, MixFusionDriver)
    assert driver.adapter is None


def test_pipeline_without_transformer_mounts_nothing(monkeypatch):
    monkeypatch.setattr(registry, "supports_mixfusion", lambda od_config: True)
    pipeline = _FakePipeline(None)

    registry._apply_mixfusion_if_enabled(pipeline, _od_config(enable=False))

    assert not hasattr(pipeline, "mixfusion_driver")
